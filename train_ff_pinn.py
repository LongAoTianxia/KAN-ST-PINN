import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
import sys
import time

import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm

from PINNs_util.PINNs_aux import xyt_tensor
from PINNs_util.datasets import (
    _marmousi_velocity,
    _overthrust_velocity,
    _threelayer_velocity,
    generate_marmousi,
    generate_overthrust,
    generate_threelayer,
)
from PINNs_util.ff_pinn import HardInitialFFPINN


class Config:
    """FF-PINN paper-style reproduction adapted to the local datasets.

    Defaults follow arXiv:2409.03536, Section C, as closely as possible:
    Gaussian Fourier features, Swish MLP, Adam-only SGD sampling, second-order
    Clayton-Engquist ABCs, and staged time-domain decomposition.
    """

    Nx = 60
    Ny = 60

    # Section C uses 5k + 10k + 10k + 20k staged Adam epochs.
    stage_epochs = (5000, 10000, 10000, 20000)
    stage_fractions = (0.3, 0.4, 0.5, 1.0)
    n_epochs = sum(stage_epochs)
    learning_rate = 5e-3
    lr_decay = 0.9
    lr_decay_steps = 1000
    lr_reset_each_stage = False

    # Paper: fixed Gaussian B, m=256, sigma=1.5, [128] x 5 Swish MLP.
    n_frequencies = 256
    sigma = 1.5
    hidden_dim = 128
    n_hidden_layers = 5
    include_input = False
    feature_order = "cos_sin"

    # Paper C: Nr=80,000 and Nce=2,000 points on each boundary.
    n_colloc = 80000
    n_bc_per_side = 2000
    batch_size_eval = 4096
    grad_clip = 0.0

    # Paper losses: PDE residual plus ABC residual. Appendix Table V shows
    # lambda_ce=1 is near the best fixed setting when lambda_r=1.
    w_pde = 1.0
    w_bc = 1.0
    w_data = 0.0
    n_data = 0
    data_sampling = "energy"
    data_power = 1.0
    data_eps = 1e-4
    bc_order = 2
    pde_form = "standard"

    # "source" is the paper formulation u=t^2 f_theta with Ricker forcing.
    # "initial" keeps the local datasets' initial-Gaussian problem for ablation.
    problem_form = "source"
    source_amplitude = 1.0

    save_every = 1000
    MODEL_DIR = "./trained"
    MODEL_TAG = "ff_pinn_paper"
    DATA_ROOT = "./datasets"
    DATASET = "threelayer"


SOURCE_PARAMS = {
    # f0 and alpha follow the paper cases where available; source centers follow
    # the local datasets so the trained model remains comparable with compute_l2.
    "threelayer": {"f0": 5.0, "alpha": 0.01, "center": (0.5, 0.5)},
    "marmousi": {"f0": 10.0, "alpha": 0.01, "center": (0.5, 0.15)},
    "overthrust": {"f0": 15.0, "alpha": 0.02, "center": (0.5, 0.12)},
}


def apply_initial_ablation_tuned_defaults(config):
    """Tuned initial-Gaussian setting for the local npz datasets.

    This intentionally diverges from the strict paper source-term setting and
    should be reported as ff_pinn_initial_ablation/tuned.
    """
    config.problem_form = "initial"
    config.pde_form = "flux"
    config.stage_epochs = (5000, 5000, 5000, 5000, 5000, 5000, 40000)
    config.stage_fractions = (0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0)
    config.n_epochs = sum(config.stage_epochs)
    config.learning_rate = 1e-3
    config.lr_decay = 0.95
    config.lr_decay_steps = 1000
    config.lr_reset_each_stage = True
    config.n_data = 5000
    config.w_data = 10.0
    config.data_sampling = "energy"
    config.data_power = 1.0
    config.data_eps = 1e-4


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
    raw = np.load(path)
    xx = raw["xx"]
    yy = raw["yy"]
    t = raw["t"]
    p_ref = raw["p_ref"].astype(np.float32)
    xy = np.column_stack((xx.reshape(-1, 1), yy.reshape(-1, 1)))
    r_ref = xyt_tensor(xy, t, device)
    return {
        "L": float(raw["L"]),
        "T": float(raw["T"]),
        "L_phys": float(raw["L_phys"]) if "L_phys" in raw.files else float(raw["L"]),
        "T_phys": float(raw["T_phys"]) if "T_phys" in raw.files else float(raw["T"]),
        "p_ref": p_ref,
        "xx": xx,
        "yy": yy,
        "t": t,
        "xy": xy,
        "r_ref": r_ref,
        "n_T": t.shape[0],
        "n_L": xx.shape[0],
        "c": raw["c"] if "c" in raw.files else None,
        "c_func": make_c_func(name),
        "p_max": float(raw["p_max"]) if "p_max" in raw.files else 1.0,
    }


def prepare_dataset(config, device):
    name = config.DATASET.lower()
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
    raise ValueError(f"Unknown dataset: {name}")


def prepare_supervised_tensors(data, device):
    p_ref = data["p_ref"].reshape(-1)
    p_gpu = torch.tensor(p_ref, dtype=torch.float32, device=device).unsqueeze(1)
    weights = np.abs(data["p_ref"].astype(np.float32)) ** 1.0
    data["p_ref_gpu"] = p_gpu
    data["p_abs_weight"] = torch.tensor(weights.reshape(-1), dtype=torch.float32, device=device)


def sample_supervised(data, n_data, device, t_limit, sampling="energy", power=1.0, eps=1e-4):
    n_t = int(data["n_T"])
    n_space = int(data["n_L"] * data["n_L"])
    max_t_idx = int(np.searchsorted(data["t"], t_limit, side="right"))
    max_t_idx = max(1, min(max_t_idx, n_t))

    if sampling == "uniform":
        spatial_idx = torch.randint(0, n_space, (n_data,), device=device)
        time_idx = torch.randint(0, max_t_idx, (n_data,), device=device)
    else:
        weights = data["p_abs_weight"].view(n_space, n_t)[:, :max_t_idx]
        if power != 1.0:
            weights = torch.pow(weights + 1e-12, power)
        weights = weights.reshape(-1) + float(eps)
        local_idx = torch.multinomial(weights, n_data, replacement=True)
        spatial_idx = torch.div(local_idx, max_t_idx, rounding_mode="floor")
        time_idx = local_idx - spatial_idx * max_t_idx

    idx = spatial_idx * n_t + time_idx
    return data["r_ref"][idx], data["p_ref_gpu"][idx]


def create_model(config, data, device):
    initial_mode = "zero" if config.problem_form == "source" else "initial_pressure"
    model = HardInitialFFPINN(
        dataset=config.DATASET,
        p_max=data.get("p_max", 1.0),
        initial_mode=initial_mode,
        input_dim=3,
        output_dim=1,
        n_frequencies=config.n_frequencies,
        sigma=config.sigma,
        hidden_dim=config.hidden_dim,
        n_hidden_layers=config.n_hidden_layers,
        include_input=config.include_input,
        feature_order=config.feature_order,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[FF-PINN] Created model: {n_params:,} parameters")
    print(
        f"  Gaussian FF m={config.n_frequencies}, sigma={config.sigma}, "
        f"hidden={config.hidden_dim}, layers={config.n_hidden_layers}, "
        f"include_input={config.include_input}, initial_mode={initial_mode}"
    )
    return model


def sample_colloc(n, L, t_limit, device):
    xy = torch.rand((n, 2), device=device) * L
    tt = torch.rand((n, 1), device=device) * t_limit
    return torch.cat([xy, tt], dim=1).requires_grad_(True)


def sample_boundary_per_side(n_per_side, L, t_limit, device):
    n = int(n_per_side)
    tt = torch.rand((4 * n, 1), device=device) * t_limit
    coord = torch.rand((4 * n, 1), device=device) * L
    zeros = torch.zeros((n, 1), device=device)
    full = torch.full((n, 1), float(L), device=device)

    r0 = torch.cat([zeros, coord[0:n], tt[0:n]], dim=1)
    r1 = torch.cat([full, coord[n:2 * n], tt[n:2 * n]], dim=1)
    r2 = torch.cat([coord[2 * n:3 * n], zeros, tt[2 * n:3 * n]], dim=1)
    r3 = torch.cat([coord[3 * n:4 * n], full, tt[3 * n:4 * n]], dim=1)
    r = torch.cat([r0, r1, r2, r3], dim=0).requires_grad_(True)
    side = torch.cat(
        [torch.full((n, 1), i, device=device, dtype=torch.long) for i in range(4)],
        dim=0,
    )
    return r, side


def ricker_source(r, config):
    params = SOURCE_PARAMS[config.DATASET]
    f0 = float(params["f0"])
    alpha = float(params["alpha"])
    xs, ys = params["center"]
    x = r[:, 0:1]
    y = r[:, 1:2]
    t = r[:, 2:3]
    tau = np.pi * f0 * (t - 1.0 / f0)
    wavelet = config.source_amplitude * (1.0 - 2.0 * tau ** 2) * torch.exp(-(tau ** 2))
    spatial = torch.exp(-0.5 * (((x - xs) / alpha) ** 2 + ((y - ys) / alpha) ** 2))
    return wavelet * spatial


def pde_residual_with_optional_source(p, r, c, config):
    p_r = torch.autograd.grad(p, r, torch.ones_like(p), create_graph=True)[0]
    p_x = p_r[:, 0:1]
    p_y = p_r[:, 1:2]
    p_xx = torch.autograd.grad(p_r[:, 0], r, torch.ones_like(p_r[:, 0]), create_graph=True)[0][:, 0:1]
    p_yy = torch.autograd.grad(p_r[:, 1], r, torch.ones_like(p_r[:, 1]), create_graph=True)[0][:, 1:2]
    p_tt = torch.autograd.grad(p_r[:, 2], r, torch.ones_like(p_r[:, 2]), create_graph=True)[0][:, 2:3]
    if config.pde_form == "flux":
        q = c ** 2
        if q.requires_grad:
            q_r = torch.autograd.grad(
                q, r, torch.ones_like(q), create_graph=True, allow_unused=True
            )[0]
        else:
            q_r = None
        if q_r is None:
            q_x = torch.zeros_like(p_x)
            q_y = torch.zeros_like(p_y)
        else:
            q_x = q_r[:, 0:1]
            q_y = q_r[:, 1:2]
        flux_div = q * (p_xx + p_yy) + q_x * p_x + q_y * p_y
        residual = p_tt - flux_div
    else:
        residual = p_tt - c ** 2 * (p_xx + p_yy)
    if config.problem_form == "source":
        residual = residual - ricker_source(r, config)
    return residual


def first_order_abc_residual(p, r, side, c):
    p_r = torch.autograd.grad(p, r, torch.ones_like(p), create_graph=True)[0]
    p_x = p_r[:, 0:1]
    p_y = p_r[:, 1:2]
    p_t = p_r[:, 2:3]
    res = torch.zeros_like(p)
    masks = [(side == i).squeeze() for i in range(4)]
    if masks[0].any():
        res[masks[0]] = p_x[masks[0]] - p_t[masks[0]] / c[masks[0]]
    if masks[1].any():
        res[masks[1]] = p_x[masks[1]] + p_t[masks[1]] / c[masks[1]]
    if masks[2].any():
        res[masks[2]] = p_y[masks[2]] - p_t[masks[2]] / c[masks[2]]
    if masks[3].any():
        res[masks[3]] = p_y[masks[3]] + p_t[masks[3]] / c[masks[3]]
    return res


def second_order_paraxial_abc_residual(p, r, side, c):
    p_r = torch.autograd.grad(p, r, torch.ones_like(p), create_graph=True)[0]
    p_x = p_r[:, 0:1]
    p_y = p_r[:, 1:2]
    p_t = p_r[:, 2:3]
    p_xt = torch.autograd.grad(p_x, r, torch.ones_like(p_x), create_graph=True)[0][:, 2:3]
    p_yt = torch.autograd.grad(p_y, r, torch.ones_like(p_y), create_graph=True)[0][:, 2:3]
    p_tt = torch.autograd.grad(p_t, r, torch.ones_like(p_t), create_graph=True)[0][:, 2:3]
    p_xx = torch.autograd.grad(p_x, r, torch.ones_like(p_x), create_graph=True)[0][:, 0:1]
    p_yy = torch.autograd.grad(p_y, r, torch.ones_like(p_y), create_graph=True)[0][:, 1:2]

    res = torch.zeros_like(p)
    masks = [(side == i).squeeze() for i in range(4)]
    if masks[0].any():
        res[masks[0]] = p_xt[masks[0]] - p_tt[masks[0]] / c[masks[0]] + 0.5 * c[masks[0]] * p_yy[masks[0]]
    if masks[1].any():
        res[masks[1]] = p_xt[masks[1]] + p_tt[masks[1]] / c[masks[1]] - 0.5 * c[masks[1]] * p_yy[masks[1]]
    if masks[2].any():
        res[masks[2]] = p_yt[masks[2]] - p_tt[masks[2]] / c[masks[2]] + 0.5 * c[masks[2]] * p_xx[masks[2]]
    if masks[3].any():
        res[masks[3]] = p_yt[masks[3]] + p_tt[masks[3]] / c[masks[3]] - 0.5 * c[masks[3]] * p_xx[masks[3]]
    return res


def compute_bc_residual(p, r, side, c, order):
    if order == 2:
        return second_order_paraxial_abc_residual(p, r, side, c)
    return first_order_abc_residual(p, r, side, c)


def current_t_limit(config, data, epoch):
    t_max = float(data["t"][-1])
    elapsed = 0
    for stage_epochs, frac in zip(config.stage_epochs, config.stage_fractions):
        elapsed += int(stage_epochs)
        if epoch < elapsed:
            return max(float(data["t"][1]), min(t_max, t_max * float(frac)))
    return t_max


def current_stage_info(config, epoch):
    start = 0
    for i, stage_epochs in enumerate(config.stage_epochs):
        end = start + int(stage_epochs)
        if epoch < end:
            return i, start, end
        start = end
    return len(config.stage_epochs) - 1, max(0, start - int(config.stage_epochs[-1])), start


def current_learning_rate(config, epoch):
    if config.lr_reset_each_stage:
        _, stage_start, _ = current_stage_info(config, epoch)
        local_epoch = epoch - stage_start
        decay_count = (local_epoch + 1) // max(1, int(config.lr_decay_steps))
    else:
        decay_count = (epoch + 1) // max(1, int(config.lr_decay_steps))
    return float(config.learning_rate) * (float(config.lr_decay) ** int(decay_count))


def set_optimizer_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


def evaluate_model(model, data, device, batch_size):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, data["r_ref"].shape[0], batch_size):
            preds.append(model(data["r_ref"][i:i + batch_size]).cpu())
    p_est = torch.cat(preds, dim=0).numpy().reshape(-1)
    p_ref = data["p_ref"].reshape(-1)
    rel_l2 = np.linalg.norm(p_est - p_ref) / (np.linalg.norm(p_ref) + 1e-12)
    mae = np.mean(np.abs(p_est - p_ref))
    return {"rel_l2": rel_l2, "mae": mae}


def latest_checkpoint(out_dir):
    if not os.path.isdir(out_dir):
        return None
    ckpts = [f for f in os.listdir(out_dir) if f.startswith("checkpoint_epoch") and f.endswith(".pt")]
    if not ckpts:
        return None
    ckpts.sort(key=lambda x: int(x.replace("checkpoint_epoch", "").replace(".pt", "")))
    return os.path.join(out_dir, ckpts[-1])


def training_config_dict(config):
    return {
        "paper_reference": "arXiv:2409.03536v2 Section C",
        "problem_form": config.problem_form,
        "stage_epochs": tuple(config.stage_epochs),
        "stage_fractions": tuple(config.stage_fractions),
        "n_colloc": config.n_colloc,
        "n_bc_per_side": config.n_bc_per_side,
        "learning_rate": config.learning_rate,
        "lr_decay": config.lr_decay,
        "lr_decay_steps": config.lr_decay_steps,
        "lr_reset_each_stage": config.lr_reset_each_stage,
        "w_pde": config.w_pde,
        "w_bc": config.w_bc,
        "w_data": config.w_data,
        "n_data": config.n_data,
        "data_sampling": config.data_sampling,
        "data_power": config.data_power,
        "data_eps": config.data_eps,
        "bc_order": config.bc_order,
        "pde_form": config.pde_form,
        "source_amplitude": config.source_amplitude,
        "source_params": SOURCE_PARAMS.get(config.DATASET, {}),
    }


def save_checkpoint(out_dir, model, optimizer, scheduler, history, epoch, config):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model.get_config(),
            "training_config": training_config_dict(config),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "history": history,
            "epoch": epoch,
            "train_time_sec": history.get("train_time_sec", 0.0),
        },
        os.path.join(out_dir, f"checkpoint_epoch{epoch}.pt"),
    )
    torch.save(history, os.path.join(out_dir, "history.pt"))


def train_one(config, device, resume=False):
    set_seed(0)
    data = prepare_dataset(config, device)
    if config.w_data > 0 and config.n_data > 0:
        prepare_supervised_tensors(data, device)

    model = create_model(config, data, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = None
    history = {
        "loss_total": [],
        "loss_pde": [],
        "loss_bc": [],
        "loss_data": [],
        "t_limit": [],
        "lr": [],
        "train_time_sec": 0.0,
    }
    start_epoch = 0

    out_dir = os.path.join(config.MODEL_DIR, config.DATASET, config.MODEL_TAG)
    if resume:
        ckpt_path = latest_checkpoint(out_dir)
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            history = ckpt.get("history", history)
            history.setdefault("loss_data", [])
            history.setdefault("train_time_sec", float(ckpt.get("train_time_sec", 0.0)))
            start_epoch = int(ckpt.get("epoch", 0))
            print(f"Resumed from {ckpt_path}")

    print(f"Training FF-PINN on {config.DATASET} for {config.n_epochs} epochs, device={device}")
    print(
        f"Paper C params: Adam lr={config.learning_rate:g}, exp_decay={config.lr_decay}/"
        f"{config.lr_decay_steps} epochs, Nr={config.n_colloc}, "
        f"Nce={config.n_bc_per_side} per side, second_order_ABC={config.bc_order == 2}, "
        f"problem_form={config.problem_form}, pde_form={config.pde_form}, "
        f"n_data={config.n_data}, w_data={config.w_data}, lr_reset={config.lr_reset_each_stage}"
    )
    prior_train_time = float(history.get("train_time_sec", 0.0) or 0.0)
    train_start = time.time()

    for epoch in tqdm(range(start_epoch, config.n_epochs), desc=f"ff/{config.DATASET}", ncols=90, file=sys.stderr):
        model.train()
        optimizer.zero_grad()
        t_limit = current_t_limit(config, data, epoch)
        lr = current_learning_rate(config, epoch)
        set_optimizer_lr(optimizer, lr)

        r_colloc = sample_colloc(config.n_colloc, data["L"], t_limit, device)
        c_colloc = data["c_func"](r_colloc[:, 0:1], r_colloc[:, 1:2], tensor=True)
        p_colloc = model(r_colloc)
        loss_pde = torch.mean(pde_residual_with_optional_source(p_colloc, r_colloc, c_colloc, config) ** 2)

        r_bc, side = sample_boundary_per_side(config.n_bc_per_side, data["L"], t_limit, device)
        c_bc = data["c_func"](r_bc[:, 0:1], r_bc[:, 1:2], tensor=True)
        p_bc = model(r_bc)
        loss_bc = torch.mean(compute_bc_residual(p_bc, r_bc, side, c_bc, config.bc_order) ** 2)

        if config.w_data > 0 and config.n_data > 0:
            r_data, p_data = sample_supervised(
                data,
                config.n_data,
                device,
                t_limit,
                sampling=config.data_sampling,
                power=config.data_power,
                eps=config.data_eps,
            )
            loss_data = torch.mean((model(r_data) - p_data) ** 2)
        else:
            loss_data = torch.zeros((), device=device)

        loss = config.w_pde * loss_pde + config.w_bc * loss_bc + config.w_data * loss_data
        loss.backward()
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        history["loss_total"].append(float(loss.detach().cpu()))
        history["loss_pde"].append(float(loss_pde.detach().cpu()))
        history["loss_bc"].append(float(loss_bc.detach().cpu()))
        history["loss_data"].append(float(loss_data.detach().cpu()))
        history["t_limit"].append(float(t_limit))
        history["lr"].append(float(lr))
        history["train_time_sec"] = prior_train_time + (time.time() - train_start)

        if epoch % 100 == 99:
            print(
                f"[{epoch + 1}] total={history['loss_total'][-1]:.3e} "
                f"pde={history['loss_pde'][-1]:.3e} bc={history['loss_bc'][-1]:.3e} "
                f"data={history['loss_data'][-1]:.3e} "
                f"t_max={t_limit:.3f} lr={history['lr'][-1]:.3e}"
            )
        if (epoch + 1) % config.save_every == 0:
            save_checkpoint(out_dir, model, optimizer, scheduler, history, epoch + 1, config)

    history["train_time_sec"] = prior_train_time + (time.time() - train_start)
    save_checkpoint(out_dir, model, optimizer, scheduler, history, config.n_epochs, config)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model.get_config(),
            "training_config": training_config_dict(config),
            "history": history,
            "epoch": config.n_epochs,
            "train_time_sec": history["train_time_sec"],
        },
        os.path.join(out_dir, "model.pt"),
    )
    torch.save(history, os.path.join(out_dir, "history.pt"))
    metrics = evaluate_model(model, data, device, config.batch_size_eval)
    print(f"[FF-PINN/{config.DATASET}] Rel L2={metrics['rel_l2']:.6f}  MAE={metrics['mae']:.6e}")
    print(f"[FF-PINN/{config.DATASET}] Train time={history['train_time_sec']:.1f}s ({history['train_time_sec']/3600:.3f}h)")


def parse_stage_epochs(raw):
    values = tuple(int(x.strip()) for x in raw.split(",") if x.strip())
    if not values:
        raise argparse.ArgumentTypeError("stage epochs cannot be empty")
    return values


def parse_stage_fractions(raw):
    values = tuple(float(x.strip()) for x in raw.split(",") if x.strip())
    if not values:
        raise argparse.ArgumentTypeError("stage fractions cannot be empty")
    if any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("stage fractions must be positive")
    return values


def parse_args():
    parser = argparse.ArgumentParser(description="Train paper-style FF-PINN on wavefield datasets.")
    parser.add_argument("--dataset", choices=["threelayer", "marmousi", "overthrust", "all"], default="threelayer")
    parser.add_argument("--epochs", type=int, default=None, help="Override total epochs; disables staged epoch sum only.")
    parser.add_argument("--stage-epochs", type=parse_stage_epochs, default=None, help="Comma list, default 5000,10000,10000,20000.")
    parser.add_argument("--stage-fractions", type=parse_stage_fractions, default=None, help="Comma list of time-window fractions, default 0.3,0.4,0.5,1.0.")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr-decay", type=float, default=None)
    parser.add_argument("--lr-decay-steps", type=int, default=None)
    parser.add_argument("--lr-reset-each-stage", action="store_true")
    parser.add_argument("--n-colloc", type=int, default=None)
    parser.add_argument("--n-bc", type=int, default=None, help="Boundary points per side, matching paper Nce.")
    parser.add_argument("--n-data", type=int, default=None)
    parser.add_argument("--w-data", type=float, default=None)
    parser.add_argument("--data-sampling", choices=["uniform", "energy"], default=None)
    parser.add_argument("--data-power", type=float, default=None)
    parser.add_argument("--data-eps", type=float, default=None)
    parser.add_argument("--frequencies", type=int, default=None)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--bc-order", type=int, choices=[1, 2], default=None)
    parser.add_argument("--pde-form", choices=["standard", "flux"], default=None)
    parser.add_argument("--problem-form", choices=["source", "initial"], default=None)
    parser.add_argument("--model-tag", default=None, help="Output tag under trained/<dataset>/; default ff_pinn_paper.")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    base = Config()
    if args.model_tag is not None:
        base.MODEL_TAG = args.model_tag
    if base.MODEL_TAG in ("ff_pinn_initial_ablation", "ff_pinn_initial_tuned"):
        apply_initial_ablation_tuned_defaults(base)
    if args.stage_epochs is not None:
        base.stage_epochs = args.stage_epochs
        base.n_epochs = sum(base.stage_epochs)
    if args.stage_fractions is not None:
        base.stage_fractions = args.stage_fractions
    if len(base.stage_epochs) != len(base.stage_fractions):
        raise ValueError("--stage-epochs and --stage-fractions must have the same length")
    if args.epochs is not None:
        base.n_epochs = args.epochs
    if args.lr is not None:
        base.learning_rate = args.lr
    if args.lr_decay is not None:
        base.lr_decay = args.lr_decay
    if args.lr_decay_steps is not None:
        base.lr_decay_steps = args.lr_decay_steps
    if args.lr_reset_each_stage:
        base.lr_reset_each_stage = True
    if args.n_colloc is not None:
        base.n_colloc = args.n_colloc
    if args.n_bc is not None:
        base.n_bc_per_side = args.n_bc
    if args.n_data is not None:
        base.n_data = args.n_data
    if args.w_data is not None:
        base.w_data = args.w_data
    if args.data_sampling is not None:
        base.data_sampling = args.data_sampling
    if args.data_power is not None:
        base.data_power = args.data_power
    if args.data_eps is not None:
        base.data_eps = args.data_eps
    if args.frequencies is not None:
        base.n_frequencies = args.frequencies
    if args.sigma is not None:
        base.sigma = args.sigma
    if args.hidden is not None:
        base.hidden_dim = args.hidden
    if args.layers is not None:
        base.n_hidden_layers = args.layers
    if args.bc_order is not None:
        base.bc_order = args.bc_order
    if args.pde_form is not None:
        base.pde_form = args.pde_form
    if args.problem_form is not None:
        base.problem_form = args.problem_form
    if args.model_tag is not None:
        base.MODEL_TAG = args.model_tag

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    datasets = ["threelayer", "marmousi", "overthrust"] if args.dataset == "all" else [args.dataset]
    for name in datasets:
        config = Config()
        config.__dict__.update(base.__dict__)
        config.DATASET = name
        train_one(config, device, resume=args.resume)


if __name__ == "__main__":
    main()
