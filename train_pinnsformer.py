import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import LBFGS
from tqdm import tqdm

from PINNs_util.PINNs_aux import bc_residual_absorbing, pde_residual, rand_boundary, xyt_tensor
from PINNs_util.datasets import _marmousi_velocity, _overthrust_velocity, _threelayer_velocity
from PINNs_util.pinnsformer_wave import PINNsFormerWave2D


class Config:
    DATA_ROOT = "./datasets"
    MODEL_DIR = "./trained"
    DATASET = "threelayer"
    tag = "pinnsformer_paper"

    d_model = 32
    d_hidden = 512
    n_layers = 1
    heads = 2
    num_step = 5
    step_size = 1e-4

    n_data = 5000
    n_colloc = 10000
    n_bc = 2000
    w_data = 10.0
    w_pde = 1.0
    w_bc = 0.1

    lbfgs_max_iter = 20000
    lbfgs_history_size = 50
    save_every = 100


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.01)


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


def sample_supervised(data, n_data, device):
    n_total = data["r_ref"].shape[0]
    idx = torch.randint(0, n_total, (n_data,), device=device)
    return data["r_ref"][idx], data["p_ref_gpu"][idx]


def sample_colloc(data, n_colloc, device):
    xy = torch.rand((n_colloc, 2), device=device) * float(data["L"])
    tt = torch.rand((n_colloc, 1), device=device) * float(data["t"][-1])
    return torch.cat([xy, tt], dim=1)


def make_loss(model, data, config, device, mse):
    r_data, p_data = sample_supervised(data, config.n_data, device)
    loss_data = mse(model(r_data), p_data)

    r_colloc = sample_colloc(data, config.n_colloc, device).requires_grad_(True)
    c_colloc = data["c_func"](r_colloc[:, 0:1], r_colloc[:, 1:2], tensor=True)
    with torch.backends.cudnn.flags(enabled=False):
        p_colloc = model(r_colloc)
        loss_pde = torch.mean(pde_residual(p_colloc, r_colloc, c_colloc) ** 2)

    r_bc, side = rand_boundary(config.n_bc, data["L"], float(data["t"][-1]), device)
    r_bc = r_bc.detach().clone().requires_grad_(True)
    c_bc = data["c_func"](r_bc[:, 0:1], r_bc[:, 1:2], tensor=True)
    p_bc = model(r_bc)
    loss_bc = torch.mean(bc_residual_absorbing(p_bc, r_bc, side, c_bc) ** 2)

    loss = config.w_data * loss_data + config.w_pde * loss_pde + config.w_bc * loss_bc
    return loss, loss_data, loss_pde, loss_bc


def save_model(out_dir, model, history, config, dataset, optimizer_state=None):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model.get_config(),
            "optimizer_state": optimizer_state,
            "history": history,
            "dataset": dataset,
            "train_time_sec": history.get("train_time_sec", 0.0),
            "pinnsformer_config": {
                "paper_setting": True,
                "source_repo": "https://github.com/AdityaLab/pinnsformer",
                "adaptation": "2D acoustic wavefield p(x,y,t)",
                "n_data": config.n_data,
                "n_colloc": config.n_colloc,
                "n_bc": config.n_bc,
                "lbfgs_max_iter": config.lbfgs_max_iter,
            },
        },
        os.path.join(out_dir, "model.pt"),
    )
    torch.save(history, os.path.join(out_dir, "history.pt"))


def train_one(config, dataset, device):
    set_seed(0)
    data = load_npz_dataset(dataset, config.DATA_ROOT, device)
    data["p_ref_gpu"] = torch.tensor(data["p_ref"].reshape(-1), dtype=torch.float32, device=device).unsqueeze(1)

    model = PINNsFormerWave2D(
        d_out=1,
        d_model=config.d_model,
        d_hidden=config.d_hidden,
        n_layers=config.n_layers,
        heads=config.heads,
        num_step=config.num_step,
        step_size=config.step_size,
    ).to(device)
    model.apply(init_weights)

    out_dir = os.path.join(config.MODEL_DIR, dataset, config.tag)
    optimizer = LBFGS(
        model.parameters(),
        max_iter=config.lbfgs_max_iter,
        history_size=config.lbfgs_history_size,
        line_search_fn="strong_wolfe",
    )
    mse = nn.MSELoss()
    history = {"loss_total": [], "loss_data": [], "loss_pde": [], "loss_bc": [], "train_time_sec": 0.0}

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Training PINNsFormer paper setting on {dataset}, device={device}")
    print(
        f"  params={n_params:,}, d_model={config.d_model}, hidden={config.d_hidden}, "
        f"N={config.n_layers}, heads={config.heads}, num_step={config.num_step}, step={config.step_size}"
    )
    print(f"  LBFGS max_iter={config.lbfgs_max_iter}, n_data={config.n_data}, n_colloc={config.n_colloc}, n_bc={config.n_bc}")

    start = time.time()
    pbar = tqdm(total=config.lbfgs_max_iter, desc=f"pinnsformer/{dataset}", ncols=90, file=sys.stderr)
    state = {"last_save": 0}

    def closure():
        optimizer.zero_grad()
        loss, loss_data, loss_pde, loss_bc = make_loss(model, data, config, device, mse)
        loss.backward()
        history["loss_total"].append(float(loss.detach().cpu()))
        history["loss_data"].append(float(loss_data.detach().cpu()))
        history["loss_pde"].append(float(loss_pde.detach().cpu()))
        history["loss_bc"].append(float(loss_bc.detach().cpu()))
        history["train_time_sec"] = time.time() - start
        pbar.update(1)
        n = len(history["loss_total"])
        if n - state["last_save"] >= config.save_every:
            save_model(out_dir, model, history, config, dataset, optimizer.state_dict())
            state["last_save"] = n
        return loss

    optimizer.step(closure)
    pbar.close()
    history["train_time_sec"] = time.time() - start
    save_model(out_dir, model, history, config, dataset, optimizer.state_dict())
    print(f"[PINNsFormer/{dataset}] Train time={history['train_time_sec']:.1f}s ({history['train_time_sec']/3600:.3f}h)")


def parse_args():
    parser = argparse.ArgumentParser(description="Train PINNsFormer paper-style baseline on local wavefield datasets.")
    parser.add_argument("--dataset", choices=["threelayer", "marmousi", "overthrust", "all"], default="threelayer")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--lbfgs-max-iter", type=int, default=None)
    parser.add_argument("--n-data", type=int, default=None)
    parser.add_argument("--n-colloc", type=int, default=None)
    parser.add_argument("--n-bc", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()
    if args.data_root is not None:
        config.DATA_ROOT = args.data_root
    if args.tag is not None:
        config.tag = args.tag
    if args.lbfgs_max_iter is not None:
        config.lbfgs_max_iter = args.lbfgs_max_iter
    if args.n_data is not None:
        config.n_data = args.n_data
    if args.n_colloc is not None:
        config.n_colloc = args.n_colloc
    if args.n_bc is not None:
        config.n_bc = args.n_bc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = ["threelayer", "marmousi", "overthrust"] if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        train_one(config, dataset, device)


if __name__ == "__main__":
    main()
