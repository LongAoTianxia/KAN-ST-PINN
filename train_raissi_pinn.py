import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import math
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm

from PINNs_util.PINNs_aux import (
    FCN,
    pde_residual,
    rand_colloc_fixed,
    update_lambda,
    xyt_tensor,
)
from PINNs_util.datasets import (
    _marmousi_velocity,
    _overthrust_velocity,
    _threelayer_velocity,
    generate_marmousi,
    generate_overthrust,
    generate_threelayer,
)


class Config:
    """Raissi-style continuous-time PINN baseline for p(x, y, t)."""

    Nx = 60
    Ny = 60
    n_hidden = 128
    n_layers = 4
    n_ffeatures = 64

    n_epochs = 10000
    auto_epochs = True
    learning_rate = 1e-3
    n_ini = 20
    n_data = 0
    n_colloc = 20000
    n_bc = 800
    ratio_gaussian = 0.85
    causal = True
    n_causal = 2000
    n_lamb_update = 100
    lambda_alpha = 0.9
    mini_batch_ini = 0
    grad_clip = 1.0

    w_ini = 10.0
    w_data = 10.0
    w_pde = 0.01
    w_bc = 0.0
    pde_warmup_epochs = 2000
    pde_rampup_epochs = 3000
    warmup_epochs = 500
    lr_min_ratio = 0.01
    late_data_alpha = 3.0
    late_colloc_uniform = 0.5

    MODEL_DIR = "./trained"
    MODEL_TAG = "raissi_pinn_adaptive_nobc"
    DATA_ROOT = "./datasets"
    DATASET = "threelayer"


def set_seed(seed: int = 0):
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
        return None

    raw = np.load(path)
    xx = raw["xx"]
    yy = raw["yy"]
    t = raw["t"]
    xy = np.column_stack((xx.reshape(-1, 1), yy.reshape(-1, 1)))
    r_ref = xyt_tensor(xy, t, device)
    return {
        "L": float(raw["L"]),
        "T": float(raw["T"]),
        "L_phys": float(raw["L_phys"]) if "L_phys" in raw.files else float(raw["L"]),
        "T_phys": float(raw["T_phys"]) if "T_phys" in raw.files else float(raw["T"]),
        "p_ref": raw["p_ref"],
        "xx": xx,
        "yy": yy,
        "t": t,
        "xy": xy,
        "r_ref": r_ref,
        "n_T": t.shape[0],
        "n_L": xx.shape[0],
        "c": raw["c"] if "c" in raw.files else None,
        "c_func": make_c_func(name),
        "p_max": float(raw["p_max"]) if "p_max" in raw.files else None,
    }


def prepare_dataset(config, device, regenerate=False):
    name = config.DATASET.lower()
    if not regenerate:
        data = load_npz_dataset(name, config.DATA_ROOT, device)
        if data is not None:
            print(f"Loaded dataset from {os.path.join(config.DATA_ROOT, name + '.npz')}")
            return data

    if name == "threelayer":
        return generate_threelayer(device, Nx=config.Nx, Ny=config.Ny)
    if name == "marmousi":
        return generate_marmousi(device, Nx=config.Nx, Ny=config.Ny)
    if name == "overthrust":
        return generate_overthrust(device, Nx=config.Nx, Ny=config.Ny)
    raise ValueError(f"Unknown dataset: {config.DATASET}")


def prepare_initial_condition(data, config, device):
    t_ini = data["t"][:config.n_ini]
    r_ini = xyt_tensor(data["xy"], t_ini, device)
    p_ini = data["p_ref"][:, :, : config.n_ini].reshape(-1, 1)
    return r_ini, torch.tensor(p_ini, dtype=torch.float32, device=device)


def create_model(config, device):
    model = FCN(
        n_in=3,
        n_out=1,
        n_ffeatures=config.n_ffeatures,
        n_hidden=config.n_hidden,
        n_layers=config.n_layers,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Raissi-PINN] Created FCN: {n_params:,} parameters")
    print(
        f"  hidden={config.n_hidden}, layers={config.n_layers}, "
        f"fourier={config.n_ffeatures}"
    )
    return model


def sample_colloc(config, data, device, t_max):
    parts = []
    n_base = int(config.n_colloc * (1.0 - config.late_colloc_uniform))
    n_unif_t = config.n_colloc - n_base
    if n_base > 0:
        parts.append(
            rand_colloc_fixed(
                n_base,
                data["L"],
                t_max,
                device,
                ratio_gaussian=config.ratio_gaussian,
            ).detach()
        )
    if n_unif_t > 0:
        xy = torch.rand((n_unif_t, 2), device=device) * data["L"]
        tt = torch.rand((n_unif_t, 1), device=device) * t_max
        parts.append(torch.cat([xy, tt], dim=1))
    return torch.cat(parts, dim=0).detach()


def pregenerate_colloc_pool(config, data, device, pool_size=100):
    pools = []
    t_max = float(data["t"][-1])
    for _ in range(pool_size):
        pools.append(sample_colloc(config, data, device, t_max))
    return pools


def causal_time_limit(epoch, data, config):
    if not config.causal:
        return float(data["t"][-1]), data["n_T"] - 1
    i_causal = min(epoch // max(1, config.n_causal) + 1, data["n_T"] - 1)
    return float(data["t"][i_causal]), i_causal


def adapt_training_schedule(config, data):
    config.n_causal = max(1, int(config.n_causal))

    if config.causal and config.auto_epochs:
        config.n_epochs = int(config.n_causal * (data["n_T"] - 1))
    else:
        config.n_epochs = max(1, int(config.n_epochs))

    config.warmup_epochs = min(
        int(config.warmup_epochs),
        max(1, int(0.01 * config.n_epochs)),
    )

    if config.causal:
        config.pde_warmup_epochs = min(
            2 * config.n_causal,
            max(0, int(0.05 * config.n_epochs)),
        )
        config.pde_rampup_epochs = min(
            6 * config.n_causal,
            max(1, int(0.15 * config.n_epochs)),
        )
    else:
        config.pde_warmup_epochs = min(
            int(config.pde_warmup_epochs),
            max(0, int(0.2 * config.n_epochs)),
        )
        config.pde_rampup_epochs = min(
            int(config.pde_rampup_epochs),
            max(1, int(0.3 * config.n_epochs)),
        )

    config.n_data = 0


def evaluate_model(model, data, device):
    p_est = []
    model.eval()
    with torch.no_grad():
        for i in range(0, data["r_ref"].shape[0], 4096):
            p_est.append(model(data["r_ref"][i : i + 4096]).cpu())
    p_est = torch.cat(p_est, dim=0).numpy().reshape(-1)
    p_ref = data["p_ref"].reshape(-1)
    rel_l2 = np.linalg.norm(p_est - p_ref) / (np.linalg.norm(p_ref) + 1e-12)
    mae = np.mean(np.abs(p_est - p_ref))
    return rel_l2, mae


def sanitize_history(history):
    clean = {}
    list_keys = [
        "loss_ini",
        "loss_pde",
        "loss_bc",
        "loss_data",
        "loss_total",
        "lamb_ini",
        "lamb_pde",
    ]
    for key in list_keys:
        values = history.get(key, [])
        clean_values = []
        for item in values:
            try:
                if isinstance(item, torch.Tensor):
                    item = item.detach().cpu().item()
                elif isinstance(item, np.generic):
                    item = item.item()
                clean_values.append(float(item))
            except (TypeError, ValueError, RuntimeError):
                continue
        clean[key] = clean_values
    try:
        train_time = history.get("train_time_sec", 0.0)
        if isinstance(train_time, torch.Tensor):
            train_time = train_time.detach().cpu().item()
        elif isinstance(train_time, np.generic):
            train_time = train_time.item()
        clean["train_time_sec"] = float(train_time)
    except (TypeError, ValueError, RuntimeError):
        clean["train_time_sec"] = 0.0
    return clean


def save_checkpoint(out_dir, model, optimizer, history, epoch):
    os.makedirs(out_dir, exist_ok=True)
    history_to_save = sanitize_history(history)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": history_to_save,
            "epoch": epoch,
            "train_time_sec": history_to_save.get("train_time_sec", 0.0),
            "model_config": {
                "n_hidden": model.fcs[0].out_features,
                "n_layers": len(model.fch) + 1,
                "n_ffeatures": model.n_ffeatures,
            },
        },
        os.path.join(out_dir, f"checkpoint_epoch{epoch}.pt"),
    )
    torch.save(history_to_save, os.path.join(out_dir, "history.pt"))


def train(config, device, resume=False):
    set_seed(0)
    data = prepare_dataset(config, device)
    adapt_training_schedule(config, data)
    r_ini, p_ini = prepare_initial_condition(data, config, device)
    model = create_model(config, device)

    out_dir = os.path.join(config.MODEL_DIR, config.DATASET, config.MODEL_TAG)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    warmup = config.warmup_epochs

    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, config.n_epochs - warmup)
        return config.lr_min_ratio + (1 - config.lr_min_ratio) * 0.5 * (
            1 + math.cos(math.pi * progress)
        )

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    start_epoch = 0
    history = {
        "loss_ini": [],
        "loss_pde": [],
        "loss_bc": [],
        "loss_data": [],
        "loss_total": [],
        "lamb_ini": [],
        "lamb_pde": [],
        "train_time_sec": 0.0,
    }

    if resume and os.path.isdir(out_dir):
        ckpts = sorted(
            [f for f in os.listdir(out_dir) if f.startswith("checkpoint_epoch") and f.endswith(".pt")],
            key=lambda f: int(f.replace("checkpoint_epoch", "").replace(".pt", "")),
        )
        if ckpts:
            ckpt = torch.load(os.path.join(out_dir, ckpts[-1]), map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            history = ckpt.get("history", history)
            history.setdefault("train_time_sec", float(ckpt.get("train_time_sec", 0.0)))
            start_epoch = int(ckpt.get("epoch", 0))
            scheduler.last_epoch = start_epoch
            print(f"Resumed from {ckpts[-1]}")

    pool_size = 100
    colloc_pool = None if config.causal else pregenerate_colloc_pool(config, data, device, pool_size)
    t_max = float(data["t"][-1])

    mse = nn.MSELoss()
    mini_bs = config.mini_batch_ini
    n_ini_total = r_ini.shape[0]
    lamb = [
        torch.tensor(1.0, device=device),
        torch.tensor(1.0, device=device),
    ]
    os.makedirs(out_dir, exist_ok=True)

    print(f"Training Raissi-PINN on {config.DATASET} for {config.n_epochs} epochs")
    if config.causal:
        print(
            f"  causal PDE training enabled: advance one time step every "
            f"{max(1, config.n_causal)} epochs"
        )
    print(f"  schedule: lr_warmup={config.warmup_epochs}")
    print(
        f"  training protocol: notebook-style adaptive IC/PDE weighting; "
        f"w_bc={config.w_bc:.1f}; no full-wavefield data loss"
    )
    prior_train_time = float(history.get("train_time_sec", 0.0) or 0.0)
    train_start_time = time.time()
    for epoch in tqdm(range(start_epoch, config.n_epochs), ncols=80, file=sys.stderr):
        model.train()
        optimizer.zero_grad()

        if mini_bs <= 0 or n_ini_total <= mini_bs:
            loss_ini = mse(model(r_ini), p_ini)
        else:
            idx = torch.randperm(n_ini_total, device=device)[:mini_bs]
            loss_ini = mse(model(r_ini[idx]), p_ini[idx])

        t_colloc_max, i_causal = causal_time_limit(epoch, data, config)
        if config.causal:
            r_colloc = sample_colloc(config, data, device, t_colloc_max).requires_grad_(True)
        else:
            r_colloc = colloc_pool[epoch % len(colloc_pool)].clone().requires_grad_(True)
        c_colloc = data["c_func"](r_colloc[:, 0:1], r_colloc[:, 1:2], tensor=True)
        p_colloc = model(r_colloc)
        loss_pde = mse(pde_residual(p_colloc, r_colloc, c_colloc), torch.zeros_like(p_colloc))

        loss_bc = torch.tensor(0.0, device=device)
        loss_data = torch.tensor(0.0, device=device)

        if epoch % config.n_lamb_update == 0:
            lamb = update_lambda(model, [loss_ini, loss_pde], lamb, config.lambda_alpha)

        loss = loss_ini + loss_pde * lamb[1] / (lamb[0] + 1e-12)
        loss.backward()
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()

        history["loss_ini"].append(loss_ini.item())
        history["loss_pde"].append(loss_pde.item())
        history["loss_bc"].append(loss_bc.item())
        history["loss_data"].append(loss_data.item())
        history["loss_total"].append(loss.item())
        history["lamb_ini"].append(float(lamb[0].detach().cpu()))
        history["lamb_pde"].append(float(lamb[1].detach().cpu()))
        history["train_time_sec"] = prior_train_time + (time.time() - train_start_time)

        if epoch % 100 == 99:
            print(
                f"[{epoch+1}] total={loss.item():.3e} ini={loss_ini.item():.3e} "
                f"pde={loss_pde.item():.3e} "
                f"lambda_ratio={(lamb[1] / (lamb[0] + 1e-12)).item():.3e} "
                f"t_pde={t_colloc_max:.3f} i_causal={i_causal}"
            )
            save_checkpoint(out_dir, model, optimizer, history, epoch + 1)

    history["train_time_sec"] = prior_train_time + (time.time() - train_start_time)
    history_to_save = sanitize_history(history)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": {
                "n_hidden": config.n_hidden,
                "n_layers": config.n_layers,
                "n_ffeatures": config.n_ffeatures,
            },
            "history": history_to_save,
            "epoch": config.n_epochs,
            "train_time_sec": history_to_save["train_time_sec"],
        },
        os.path.join(out_dir, "model.pt"),
    )
    torch.save(history_to_save, os.path.join(out_dir, "history.pt"))
    rel_l2, mae = evaluate_model(model, data, device)
    print(f"[Raissi-PINN] Relative L2 = {rel_l2:.6f}  MAE = {mae:.6e}")
    print(f"[Raissi-PINN] Train time = {history['train_time_sec']:.1f}s ({history['train_time_sec']/3600:.3f}h)")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Raissi-style PINN baseline.")
    parser.add_argument("--dataset", choices=["threelayer", "marmousi", "overthrust", "all"], default="threelayer")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--n-colloc", type=int, default=None)
    parser.add_argument("--mini-batch-ini", type=int, default=None)
    parser.add_argument("--causal", dest="causal", action="store_true", default=None)
    parser.add_argument("--no-causal", dest="causal", action="store_false")
    parser.add_argument("--n-causal", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def config_from_args(args, dataset):
    config = Config()
    config.DATASET = dataset
    if args.epochs is not None:
        config.n_epochs = args.epochs
        config.auto_epochs = False
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.nx is not None:
        config.Nx = args.nx
        config.Ny = args.nx
    if args.hidden is not None:
        config.n_hidden = args.hidden
    if args.layers is not None:
        config.n_layers = args.layers
    if args.n_colloc is not None:
        config.n_colloc = args.n_colloc
    if args.mini_batch_ini is not None:
        config.mini_batch_ini = args.mini_batch_ini
    if args.causal is not None:
        config.causal = args.causal
    if args.n_causal is not None:
        config.n_causal = args.n_causal
    return config


def main():
    args = parse_args()
    datasets = ["threelayer", "marmousi", "overthrust"] if args.dataset == "all" else [args.dataset]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    for dataset in datasets:
        print("\n" + "=" * 80)
        print(f"Training dataset: {dataset}")
        print("=" * 80)
        config = config_from_args(args, dataset)
        train(config, device, resume=args.resume)


if __name__ == "__main__":
    main()
