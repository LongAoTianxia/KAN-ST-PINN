import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm

from PINNs_util.PINNs_aux import pde_residual, rand_boundary, rand_colloc_fixed, bc_residual_absorbing
from PINNs_util.PINNs_aux import xyt_tensor
from PINNs_util.datasets import (
    _marmousi_velocity,
    _overthrust_velocity,
    _threelayer_velocity,
    generate_marmousi,
    generate_overthrust,
    generate_threelayer,
)
from PINNs_util.gabor_wave_pinn import GaborWavePINN
from PINNs_util.gabor_wave_pinn import GaborEnhancedWavePINN


class Config:
    Nx = 60
    Ny = 60
    n_epochs = 100000
    learning_rate = 1e-3
    n_data = 5000
    n_colloc = None
    n_bc = 800
    w_data = 10.0
    w_pde = 0.01
    w_bc = 0.01
    pde_warmup_epochs = 0
    pde_rampup_epochs = 0
    grad_clip = 1.0
    ratio_gaussian = 0.85
    batch_size_eval = 4096
    save_every = 5000
    lr_decay_steps = 10000
    lr_decay_gamma = 0.9
    MODEL_DIR = "./trained"
    DATA_ROOT = "./datasets"

    hidden_layers = [128, 128, 128, 128, 64]
    gabor_units = 64
    multires = 4
    freq = 6.0
    alpha = 6.0
    beta = 1.0
    center_hidden = 32
    family = "gabor"
    tag = "gabor_pinn"
    frequency = 10.0
    gepinn_neurons = 64
    gepinn_neurons_final = 64
    gepinn_embed_k = 4
    gepinn_omega = float(2.0 * np.pi * frequency)
    gepinn_v0 = 1.0
    gepinn_delta = 10.0
    gepinn_penultimate_activation = "sigmoid"


ORIGINAL_COLLOC_BY_DATASET = {
    "threelayer": 51 * 51,
    "marmousi": 151 * 101,
    "overthrust": 501 * 161,
}


def set_seed(seed=0):
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
    arr = np.load(path)
    xx = arr["xx"]
    yy = arr["yy"]
    t = arr["t"]
    p_ref = arr["p_ref"]
    xy = np.column_stack((xx.reshape(-1, 1), yy.reshape(-1, 1)))
    r_ref = xyt_tensor(xy, t, device)
    c_func = make_c_func(name)
    return {
        "L": float(arr["L"]),
        "T": float(arr["T"]),
        "L_phys": float(arr["L_phys"]) if "L_phys" in arr else float(arr["L"]),
        "T_phys": float(arr["T_phys"]) if "T_phys" in arr else float(arr["T"]),
        "p_ref": p_ref,
        "xx": xx,
        "yy": yy,
        "t": t,
        "xy": xy,
        "r_ref": r_ref,
        "n_T": t.shape[0],
        "n_L": xx.shape[0],
        "c": arr["c"] if "c" in arr else c_func(xx, yy, tensor=False),
        "c_func": c_func,
        "p_max": float(arr["p_max"]) if "p_max" in arr else float(np.max(np.abs(p_ref))),
    }


def prepare_dataset(name, device, nx, ny, data_root=None, regenerate=False):
    if data_root and not regenerate:
        data = load_npz_dataset(name, data_root, device)
        if data is not None:
            print(f"Loaded dataset: {os.path.join(data_root, name + '.npz')}  "
                  f"(p_ref shape: {data['p_ref'].shape})")
            return data

    if name == "threelayer":
        return generate_threelayer(device, Nx=nx, Ny=ny)
    if name == "marmousi":
        return generate_marmousi(device, Nx=nx, Ny=ny)
    if name == "overthrust":
        return generate_overthrust(device, Nx=nx, Ny=ny)
    raise ValueError(f"Unknown dataset: {name}")


def create_model(config, device):
    if config.family == "gepinn":
        return GaborEnhancedWavePINN(
            neurons=config.gepinn_neurons,
            neurons_final=config.gepinn_neurons_final,
            embed_k=config.gepinn_embed_k,
            omega=config.gepinn_omega,
            v0=config.gepinn_v0,
            delta=config.gepinn_delta,
            penultimate_activation=config.gepinn_penultimate_activation,
        ).to(device)

    return GaborWavePINN(
        hidden_layers=config.hidden_layers,
        gabor_units=config.gabor_units,
        multires=config.multires,
        freq=config.freq,
        alpha=config.alpha,
        beta=config.beta,
        center_hidden=config.center_hidden,
    ).to(device)


def sample_supervised(data, n_data, device):
    n_total = data["r_ref"].shape[0]
    idx = torch.randint(0, n_total, (n_data,), device=device)
    r = data["r_ref"][idx].detach()
    p = data["p_ref_gpu"][idx]
    return r, p


def evaluate_model(model, data, device, batch_size):
    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, data["r_ref"].shape[0], batch_size):
            r = data["r_ref"][i:i + batch_size].detach()
            preds.append(model(r).cpu())
    p_est = torch.cat(preds, dim=0).numpy().reshape(-1)
    p_ref = data["p_ref"].reshape(-1)
    rel_l2 = np.linalg.norm(p_est - p_ref) / (np.linalg.norm(p_ref) + 1e-12)
    mae = np.mean(np.abs(p_est - p_ref))
    return {"rel_l2": float(rel_l2), "mae": float(mae)}


def latest_checkpoint(out_dir):
    if not os.path.isdir(out_dir):
        return None
    ckpts = [
        name for name in os.listdir(out_dir)
        if name.startswith("checkpoint_epoch") and name.endswith(".pt")
    ]
    if not ckpts:
        return None
    ckpts.sort(key=lambda name: int(name.replace("checkpoint_epoch", "").replace(".pt", "")))
    return os.path.join(out_dir, ckpts[-1])


def train_one(dataset, config, device, resume=False):
    set_seed(0)
    n_colloc = config.n_colloc if config.n_colloc is not None else ORIGINAL_COLLOC_BY_DATASET[dataset]
    out_dir = os.path.join(config.MODEL_DIR, dataset, config.tag)
    os.makedirs(out_dir, exist_ok=True)

    data = prepare_dataset(dataset, device, config.Nx, config.Ny, config.DATA_ROOT)
    if "p_ref_gpu" not in data:
        p_flat = data["p_ref"].reshape(-1)
        data["p_ref_gpu"] = torch.tensor(p_flat, dtype=torch.float32, device=device).unsqueeze(1)

    model = create_model(config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=config.lr_decay_gamma)
    mse = nn.MSELoss()
    history = {"loss_total": [], "loss_data": [], "loss_pde": [], "loss_bc": [], "train_time_sec": 0.0}
    start_epoch = 0

    if resume:
        ckpt_path = latest_checkpoint(out_dir)
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            history = ckpt.get("history", history)
            history.setdefault("train_time_sec", float(ckpt.get("train_time_sec", 0.0)))
            start_epoch = int(ckpt["epoch"])
            print(f"Resumed {dataset}/gabor_pinn from epoch {start_epoch}")

    t_max = float(data["t"][-1])
    pool_size = 100
    colloc_pool = [
        rand_colloc_fixed(n_colloc, data["L"], t_max, device,
                          ratio_gaussian=config.ratio_gaussian).detach()
        for _ in range(pool_size)
    ]
    bc_pool = [
        tuple(x.detach() if torch.is_tensor(x) else x for x in rand_boundary(config.n_bc, data["L"], t_max, device))
        for _ in range(pool_size)
    ]

    if config.family == "gepinn":
        label = "Gabor-Enhanced-PINN"
    else:
        label = "Gabor-PINN"
    print(f"Training {label} on {dataset} for {config.n_epochs} epochs, device={device}")
    if config.family == "gepinn":
        print(
            "Original Gabor-Enhanced-PINN paper setting: "
            f"neurons={config.gepinn_neurons}, neurons_final={config.gepinn_neurons_final}, "
            f"K={config.gepinn_embed_k}, frequency={config.frequency}Hz, "
            f"omega={config.gepinn_omega:.6f}, delta={config.gepinn_delta}, "
            f"LR decay gamma={config.lr_decay_gamma}/steps={config.lr_decay_steps}, "
            f"n_colloc={n_colloc}"
        )
    prior_train_time = float(history.get("train_time_sec", 0.0) or 0.0)
    train_start_time = time.time()
    for epoch in tqdm(range(start_epoch, config.n_epochs), desc=f"gabor/{dataset}", ncols=90):
        model.train()
        optimizer.zero_grad()

        r_data, p_data = sample_supervised(data, config.n_data, device)
        p_pred = model(r_data)
        loss_data = mse(p_pred, p_data)

        r_colloc = colloc_pool[epoch % pool_size].clone().requires_grad_(True)
        c_colloc = data["c_func"](r_colloc[:, 0:1], r_colloc[:, 1:2], tensor=True)
        p_colloc = model(r_colloc)
        loss_pde_raw = torch.mean(pde_residual(p_colloc, r_colloc, c_colloc) ** 2)

        r_bc_det, side = bc_pool[epoch % pool_size]
        r_bc = r_bc_det.clone().requires_grad_(True)
        c_bc = data["c_func"](r_bc[:, 0:1], r_bc[:, 1:2], tensor=True)
        p_bc = model(r_bc)
        loss_bc_raw = torch.mean(bc_residual_absorbing(p_bc, r_bc, side, c_bc) ** 2)

        if epoch < config.pde_warmup_epochs:
            w_pde = 0.0
            w_bc = 0.0
        elif epoch < config.pde_warmup_epochs + config.pde_rampup_epochs:
            frac = (epoch - config.pde_warmup_epochs) / max(1, config.pde_rampup_epochs)
            w_pde = config.w_pde * frac
            w_bc = config.w_bc * frac
        else:
            w_pde = config.w_pde
            w_bc = config.w_bc

        loss = config.w_data * loss_data + w_pde * loss_pde_raw + w_bc * loss_bc_raw
        loss.backward()
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        if (epoch + 1) % config.lr_decay_steps == 0:
            scheduler.step()

        history["loss_total"].append(float(loss.detach().cpu()))
        history["loss_data"].append(float(loss_data.detach().cpu()))
        history["loss_pde"].append(float(loss_pde_raw.detach().cpu()))
        history["loss_bc"].append(float(loss_bc_raw.detach().cpu()))
        history["train_time_sec"] = prior_train_time + (time.time() - train_start_time)

        if epoch % 100 == 99:
            print(
                f"  [{epoch + 1}] total={history['loss_total'][-1]:.3e} "
                f"data={history['loss_data'][-1]:.3e} "
                f"pde={history['loss_pde'][-1]:.3e} bc={history['loss_bc'][-1]:.3e}"
            )

        if epoch in (0, 100, 500) or (epoch + 1) % config.save_every == 0:
            save_checkpoint(out_dir, model, optimizer, scheduler, history, epoch + 1)

    history["train_time_sec"] = prior_train_time + (time.time() - train_start_time)
    save_checkpoint(out_dir, model, optimizer, scheduler, history, config.n_epochs)
    model_path = os.path.join(out_dir, "model.pt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model.get_config(),
            "history": history,
            "epoch": config.n_epochs,
            "train_time_sec": history["train_time_sec"],
            "paper_setting": {
                "source_repo": "https://github.com/mahdiabedi/Gabor-Enhanced-PINN",
                "adaptation": "time-domain scalar p(x,y,t) instead of frequency-domain complex scattered Helmholtz",
                "frequency": config.frequency,
                "omega": config.gepinn_omega,
                "neurons": config.gepinn_neurons,
                "neurons_final": config.gepinn_neurons_final,
                "embed_k": config.gepinn_embed_k,
                "delta": config.gepinn_delta,
                "v0": config.gepinn_v0,
                "num_epochs": config.n_epochs,
                "lr_decay_steps": config.lr_decay_steps,
                "lr_decay_gamma": config.lr_decay_gamma,
                "n_colloc": n_colloc,
            },
        },
        model_path,
    )
    torch.save(history, os.path.join(out_dir, "history.pt"))
    metrics = evaluate_model(model, data, device, config.batch_size_eval)
    print(f"[{label}/{dataset}] Rel L2={metrics['rel_l2']:.6f}  MAE={metrics['mae']:.6e}")
    print(f"[{label}/{dataset}] Train time={history['train_time_sec']:.1f}s ({history['train_time_sec']/3600:.3f}h)")
    return metrics


def save_checkpoint(out_dir, model, optimizer, scheduler, history, epoch):
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model.get_config(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "history": history,
            "epoch": epoch,
            "train_time_sec": history.get("train_time_sec", 0.0),
        },
        os.path.join(out_dir, f"checkpoint_epoch{epoch}.pt"),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train Gabor-PINN on PINN wavefield datasets.")
    parser.add_argument("--dataset", choices=["threelayer", "marmousi", "overthrust", "all"],
                        default="all")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--n-data", type=int, default=None)
    parser.add_argument("--n-colloc", type=int, default=None)
    parser.add_argument("--n-bc", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--w-data", type=float, default=None)
    parser.add_argument("--w-pde", type=float, default=None)
    parser.add_argument("--w-bc", type=float, default=None)
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=None)
    parser.add_argument("--gabor-units", type=int, default=None)
    parser.add_argument("--multires", type=int, default=None)
    parser.add_argument("--freq", type=float, default=None)
    parser.add_argument("--frequency", type=float, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--family", choices=["gabor", "gepinn"], default="gabor")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--gepinn-neurons", type=int, default=None)
    parser.add_argument("--gepinn-neurons-final", type=int, default=None)
    parser.add_argument("--gepinn-embed-k", type=int, default=None)
    parser.add_argument("--gepinn-omega", type=float, default=None)
    parser.add_argument("--gepinn-v0", type=float, default=None)
    parser.add_argument("--gepinn-delta", type=float, default=None)
    parser.add_argument("--gepinn-penultimate-activation", choices=["sigmoid", "sin"], default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()
    config.family = args.family
    default_tag = {
        "gabor": "gabor_pinn",
        "gepinn": "gabor_enhanced_pinn",
    }[args.family]
    config.tag = args.tag or default_tag
    if args.quick:
        config.n_epochs = 600
        config.n_data = 2000
        config.n_colloc = 1000
        config.n_bc = 300
        config.pde_warmup_epochs = 100
        config.pde_rampup_epochs = 200
    if args.epochs is not None:
        config.n_epochs = args.epochs
    if args.nx is not None:
        config.Nx = args.nx
        config.Ny = args.nx
    if args.n_data is not None:
        config.n_data = args.n_data
    if args.n_colloc is not None:
        config.n_colloc = args.n_colloc
    if args.n_bc is not None:
        config.n_bc = args.n_bc
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.w_data is not None:
        config.w_data = args.w_data
    if args.w_pde is not None:
        config.w_pde = args.w_pde
    if args.w_bc is not None:
        config.w_bc = args.w_bc
    if args.hidden_layers is not None:
        config.hidden_layers = args.hidden_layers
    if args.gabor_units is not None:
        config.gabor_units = args.gabor_units
    if args.multires is not None:
        config.multires = args.multires
    if args.freq is not None:
        config.freq = args.freq
    if args.frequency is not None:
        config.frequency = args.frequency
        config.gepinn_omega = float(2.0 * np.pi * args.frequency)
    if args.data_root is not None:
        config.DATA_ROOT = args.data_root
    if args.gepinn_neurons is not None:
        config.gepinn_neurons = args.gepinn_neurons
    if args.gepinn_neurons_final is not None:
        config.gepinn_neurons_final = args.gepinn_neurons_final
    if args.gepinn_embed_k is not None:
        config.gepinn_embed_k = args.gepinn_embed_k
    if args.gepinn_omega is not None:
        config.gepinn_omega = args.gepinn_omega
    if args.gepinn_v0 is not None:
        config.gepinn_v0 = args.gepinn_v0
    if args.gepinn_delta is not None:
        config.gepinn_delta = args.gepinn_delta
    if args.gepinn_penultimate_activation is not None:
        config.gepinn_penultimate_activation = args.gepinn_penultimate_activation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = ["threelayer", "marmousi", "overthrust"] if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        train_one(dataset, config, device, resume=args.resume)


if __name__ == "__main__":
    main()
