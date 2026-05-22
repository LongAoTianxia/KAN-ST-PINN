import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm

from PINNs_util.PINNs_aux import pde_residual, xyt_tensor
from PINNs_util.datasets import (
    _marmousi_velocity,
    _overthrust_velocity,
    _threelayer_velocity,
)
from PINNs_util.td_pinn import TDSinePINN


class Config:
    """Paper-style TD-PINN adaptation for the local npz wavefield datasets."""

    DATA_ROOT = "./datasets"
    MODEL_DIR = "./trained"
    DATASET = "threelayer"

    layers = [3] + 5 * [64] + 3 * [32] + [1]
    n_colloc = 80000
    n_colloc_focus = 30000
    colloc_batch = 8192
    snap_fractions = (0.2, 0.3, 0.7)
    physics_snap_fractions = (0.2, 0.3)

    pretrain_iters = 100000
    full_iters = 60000
    physics_iters = 60000
    physics_rounds = 3
    physics_weights = (1e-4, 1e-3, 1e-1)
    learning_rates = (5e-4, 5e-4, 1e-3, 1e-4, 1e-5)
    lr_decay_steps = 10000
    lr_decay_gamma = 0.9
    lbfgs_max_iter = 20000
    lbfgs_history_size = 50

    save_every = 5000
    tag = "td_pinn_paper"


def set_seed(seed=1111):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_c_func(name):
    if name == "threelayer":
        def c_func(xx, yy, tensor=False):
            return _threelayer_velocity(xx, yy, tensor=tensor)
        return c_func
    if name == "marmousi":
        def c_func(xx, yy, tensor=False):
            return _marmousi_velocity(xx, yy, tensor=tensor) / 4.5
        return c_func
    if name == "overthrust":
        def c_func(xx, yy, tensor=False):
            return _overthrust_velocity(xx, yy, tensor=tensor) / 6.0
        return c_func
    raise ValueError(f"Unknown dataset: {name}")


def load_npz_dataset(name, data_root, device):
    path = os.path.join(data_root, f"{name}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    raw = np.load(path)
    xx = raw["xx"]
    yy = raw["yy"]
    t = raw["t"]
    xy = np.column_stack((xx.reshape(-1, 1), yy.reshape(-1, 1)))
    r_ref = xyt_tensor(xy, t, device)
    c_func = make_c_func(name)
    return {
        "L": float(raw["L"]),
        "T": float(raw["T"]),
        "p_ref": raw["p_ref"],
        "xx": xx,
        "yy": yy,
        "t": t,
        "xy": xy,
        "r_ref": r_ref,
        "n_T": t.shape[0],
        "n_L": xx.shape[0],
        "c_func": c_func,
        "p_max": float(raw["p_max"]) if "p_max" in raw.files else 1.0,
    }


def nearest_time_indices(t_values, fractions):
    t0 = float(t_values[0])
    t1 = float(t_values[-1])
    indices = []
    for frac in fractions:
        target = t0 + float(frac) * (t1 - t0)
        indices.append(int(np.argmin(np.abs(t_values - target))))
    return sorted(set(indices))


def make_snapshots(data, fractions, device):
    idx_t = nearest_time_indices(data["t"], fractions)
    snaps = []
    xy = torch.tensor(data["xy"], dtype=torch.float32, device=device)
    for k in idx_t:
        tt = torch.full((xy.shape[0], 1), float(data["t"][k]), dtype=torch.float32, device=device)
        p = torch.tensor(data["p_ref"][:, :, k].reshape(-1, 1), dtype=torch.float32, device=device)
        snaps.append((torch.cat([xy, tt], dim=1), p))
    return snaps, idx_t


def sample_colloc(config, data, device):
    L = float(data["L"])
    t_min = float(data["t"][0])
    t_max = float(data["t"][-1])
    xy = torch.rand((config.n_colloc, 2), device=device) * L
    tt = torch.rand((config.n_colloc, 1), device=device) * (t_max - t_min) + t_min
    base = torch.cat([xy, tt], dim=1)

    if config.n_colloc_focus <= 0:
        return base

    xy_f = torch.rand((config.n_colloc_focus, 2), device=device) * L
    xy_f[:, 1:2] = 0.25 * L + xy_f[:, 1:2] * 0.5
    tt_f = torch.rand((config.n_colloc_focus, 1), device=device) * max(1e-12, t_max - 0.4 * (t_max - t_min))
    tt_f = tt_f + 0.4 * (t_max - t_min) + t_min
    focus = torch.cat([xy_f, tt_f], dim=1)
    return torch.cat([base, focus], dim=0)


def snapshot_loss(model, snapshots, mse):
    loss = torch.zeros((), device=next(model.parameters()).device)
    for r_snap, p_snap in snapshots:
        loss = loss + mse(model(r_snap), p_snap)
    return loss


def physics_loss(model, data, colloc, batch_size):
    n_total = int(colloc.shape[0])
    if batch_size is not None and 0 < batch_size < n_total:
        idx = torch.randint(0, n_total, (batch_size,), device=colloc.device)
        r0 = colloc[idx]
    else:
        r0 = colloc
    r = r0.detach().clone().requires_grad_(True)
    c = data["c_func"](r[:, 0:1], r[:, 1:2], tensor=True)
    with torch.backends.cudnn.flags(enabled=False):
        p = model(r)
        res = pde_residual(p, r, c)
    return torch.mean(res ** 2)


def stage_loss(model, data, snapshots, colloc, mse, physics_weight, colloc_batch):
    loss_snap = snapshot_loss(model, snapshots, mse)
    if physics_weight <= 0:
        loss_f = torch.zeros((), device=loss_snap.device)
    else:
        loss_f = physics_loss(model, data, colloc, colloc_batch)
    return loss_snap + physics_weight * loss_f, loss_snap, loss_f


def save_model(out_dir, model, history, config, dataset, stage, epoch):
    os.makedirs(out_dir, exist_ok=True)
    state = {
        "model_state": model.state_dict(),
        "model_config": model.get_config(),
        "history": history,
        "dataset": dataset,
        "stage": stage,
        "epoch": epoch,
        "train_time_sec": history.get("train_time_sec", 0.0),
        "td_pinn_config": {
            "n_colloc": config.n_colloc,
            "n_colloc_focus": config.n_colloc_focus,
            "snap_fractions": tuple(config.snap_fractions),
            "physics_snap_fractions": tuple(config.physics_snap_fractions),
            "pretrain_iters": config.pretrain_iters,
            "full_iters": config.full_iters,
            "physics_iters": config.physics_iters,
            "physics_weights": tuple(config.physics_weights),
            "paper_setting": True,
        },
    }
    torch.save(state, os.path.join(out_dir, "model.pt"))
    torch.save(history, os.path.join(out_dir, "history.pt"))


def run_adam_stage(model, data, snapshots, colloc, config, physics_weight, lr, n_iters,
                   stage_name, history, out_dir, dataset, start_time, prior_time):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=config.lr_decay_gamma)
    mse = nn.MSELoss()
    for it in tqdm(range(n_iters), desc=f"{dataset}/{stage_name}", ncols=90, file=sys.stderr):
        optimizer.zero_grad()
        loss, loss_snap, loss_f = stage_loss(
            model, data, snapshots, colloc, mse, physics_weight, config.colloc_batch
        )
        loss.backward()
        optimizer.step()
        if (it + 1) % config.lr_decay_steps == 0:
            scheduler.step()

        history["loss_total"].append(float(loss.detach().cpu()))
        history["loss_snap"].append(float(loss_snap.detach().cpu()))
        history["loss_pde"].append(float(loss_f.detach().cpu()))
        history["stage"].append(stage_name)
        history["train_time_sec"] = prior_time + (time.time() - start_time)

        if (it + 1) % config.save_every == 0:
            save_model(out_dir, model, history, config, dataset, stage_name, it + 1)


def run_lbfgs_stage(model, data, snapshots, colloc, config, physics_weight, stage_name, history,
                    out_dir, dataset, start_time, prior_time):
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        max_iter=config.lbfgs_max_iter,
        history_size=config.lbfgs_history_size,
        line_search_fn="strong_wolfe",
    )
    mse = nn.MSELoss()
    step = {"n": 0}

    def closure():
        optimizer.zero_grad()
        loss, loss_snap, loss_f = stage_loss(
            model, data, snapshots, colloc, mse, physics_weight, config.colloc_batch
        )
        loss.backward()
        step["n"] += 1
        history["loss_total"].append(float(loss.detach().cpu()))
        history["loss_snap"].append(float(loss_snap.detach().cpu()))
        history["loss_pde"].append(float(loss_f.detach().cpu()))
        history["stage"].append(stage_name + "_lbfgs")
        history["train_time_sec"] = prior_time + (time.time() - start_time)
        return loss

    optimizer.step(closure)
    save_model(out_dir, model, history, config, dataset, stage_name + "_lbfgs", step["n"])


def train_one(config, dataset, device, skip_lbfgs=False):
    set_seed()
    data = load_npz_dataset(dataset, config.DATA_ROOT, device)
    lb = [0.0, 0.0, float(data["t"][0])]
    ub = [float(data["L"]), float(data["L"]), float(data["t"][-1])]
    model = TDSinePINN(layers=config.layers, lb=lb, ub=ub).to(device)

    snapshots, snap_idx = make_snapshots(data, config.snap_fractions, device)
    physics_snaps, physics_snap_idx = make_snapshots(data, config.physics_snap_fractions, device)
    colloc = sample_colloc(config, data, device)

    out_dir = os.path.join(config.MODEL_DIR, dataset, config.tag)
    history = {
        "loss_total": [],
        "loss_snap": [],
        "loss_pde": [],
        "stage": [],
        "train_time_sec": 0.0,
        "snap_indices": snap_idx,
        "physics_snap_indices": physics_snap_idx,
    }

    print(f"Training TD-PINN paper setting on {dataset}, device={device}")
    print(f"  layers={config.layers}")
    print(f"  collocation={config.n_colloc}+{config.n_colloc_focus}, snapshots={snap_idx}")
    start = time.time()

    run_adam_stage(model, data, snapshots, colloc, config, 0.0, config.learning_rates[0],
                   config.pretrain_iters, "pretrain", history, out_dir, dataset, start, 0.0)
    if not skip_lbfgs:
        run_lbfgs_stage(model, data, snapshots, colloc, config, 0.0, "pretrain", history,
                        out_dir, dataset, start, 0.0)

    run_adam_stage(model, data, snapshots, colloc, config, 1e-4, config.learning_rates[1],
                   config.full_iters, "full", history, out_dir, dataset, start, 0.0)
    if not skip_lbfgs:
        run_lbfgs_stage(model, data, snapshots, colloc, config, 1e-4, "full", history,
                        out_dir, dataset, start, 0.0)

    for i in range(config.physics_rounds):
        w = config.physics_weights[min(i, len(config.physics_weights) - 1)]
        lr = config.learning_rates[min(2 + i, len(config.learning_rates) - 1)]
        stage = f"physics_{i + 1}"
        run_adam_stage(model, data, physics_snaps, colloc, config, w, lr,
                       config.physics_iters, stage, history, out_dir, dataset, start, 0.0)
        if not skip_lbfgs:
            run_lbfgs_stage(model, data, physics_snaps, colloc, config, w, stage, history,
                            out_dir, dataset, start, 0.0)

    history["train_time_sec"] = time.time() - start
    save_model(out_dir, model, history, config, dataset, "done", 0)
    print(f"[TD-PINN/{dataset}] Train time={history['train_time_sec']:.1f}s ({history['train_time_sec']/3600:.3f}h)")


def parse_args():
    parser = argparse.ArgumentParser(description="Train TD-PINN paper-style baseline on local npz datasets.")
    parser.add_argument("--dataset", choices=["threelayer", "marmousi", "overthrust", "all"], default="threelayer")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--skip-lbfgs", action="store_true", help="Debug only; paper setting includes L-BFGS.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()
    if args.data_root is not None:
        config.DATA_ROOT = args.data_root
    if args.tag is not None:
        config.tag = args.tag
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = ["threelayer", "marmousi", "overthrust"] if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        train_one(config, dataset, device, skip_lbfgs=args.skip_lbfgs)


if __name__ == "__main__":
    main()
