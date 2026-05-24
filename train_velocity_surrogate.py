import argparse
import json
import os
import random
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
from tqdm import tqdm

from PINNs_util.ff_pinn import FFPINN


PAPER_DEFAULTS = {
    "n_frequencies": 256,
    "sigma": 15.0,
    "hidden_dim": 20,
    "n_hidden_layers": 5,
    "learning_rate": 5e-3,
    "epochs": 100000,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_velocity_npz(dataset, data_root):
    path = os.path.join(data_root, f"{dataset}_extracted.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing extracted velocity file: {path}. "
            "Run `python PINNs_util/datasets.py --extracted` first."
        )
    raw = np.load(path, allow_pickle=False)
    if "velocity" not in raw.files:
        raise KeyError(f"{path} does not contain a `velocity` array")
    velocity = raw["velocity"].astype(np.float32)
    dx_m = float(raw["dx_m"]) if "dx_m" in raw.files else 1.0
    meta = {key: raw[key].item() if raw[key].shape == () else raw[key].tolist() for key in raw.files if key != "velocity"}
    meta["path"] = path
    meta["dx_m"] = dx_m
    return velocity, meta


def make_coordinate_labels(velocity):
    nz, nx = velocity.shape
    x = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    z = np.linspace(0.0, 1.0, nz, dtype=np.float32)
    xx, zz = np.meshgrid(x, z)
    coords = np.column_stack((xx.reshape(-1), zz.reshape(-1))).astype(np.float32)
    labels = velocity.reshape(-1, 1).astype(np.float32)
    return coords, labels


def create_model(args, device):
    model = FFPINN(
        input_dim=2,
        output_dim=1,
        n_frequencies=args.n_frequencies,
        sigma=args.sigma,
        hidden_dim=args.hidden_dim,
        n_hidden_layers=args.n_hidden_layers,
        include_input=args.include_input,
        feature_order=args.feature_order,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[velocity surrogate] params={n_params:,}, "
        f"m={args.n_frequencies}, sigma={args.sigma}, "
        f"hidden={args.hidden_dim}, layers={args.n_hidden_layers}"
    )
    return model


def evaluate_full_grid(model, coords, labels, device, batch_size):
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, coords.shape[0], batch_size):
            batch = torch.from_numpy(coords[start:start + batch_size]).to(device)
            preds.append(model(batch).cpu().numpy())
    pred = np.concatenate(preds, axis=0).astype(np.float32)
    err = pred - labels
    rel_l2 = float(np.linalg.norm(err) / (np.linalg.norm(labels) + 1e-12))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return pred, {"rel_l2": rel_l2, "mae": mae, "rmse": rmse}


def save_diagnostics(dataset, velocity, pred, metrics, history, args, meta, out_dir, fig_dir):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    pred_img = pred.reshape(velocity.shape)
    abs_err = np.abs(pred_img - velocity)
    np.savez_compressed(
        os.path.join(out_dir, "prediction.npz"),
        velocity=velocity,
        prediction=pred_img.astype(np.float32),
        abs_error=abs_err.astype(np.float32),
        **{k: v for k, v in meta.items() if isinstance(v, (int, float, str, bool))},
    )

    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dx_km = float(meta.get("dx_m", 1.0)) / 1000.0
    extent = [0.0, velocity.shape[1] * dx_km, velocity.shape[0] * dx_km, 0.0]
    vmin = float(np.min(velocity))
    vmax = float(np.max(velocity))

    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    im0 = axs[0].imshow(velocity, cmap="viridis", origin="upper", extent=extent, vmin=vmin, vmax=vmax)
    axs[0].set_title("True velocity")
    im1 = axs[1].imshow(pred_img, cmap="viridis", origin="upper", extent=extent, vmin=vmin, vmax=vmax)
    axs[1].set_title("Surrogate prediction")
    im2 = axs[2].imshow(abs_err, cmap="magma", origin="upper", extent=extent)
    axs[2].set_title("Absolute error")
    for ax in axs:
        ax.set_xlabel("x (km)")
        ax.set_ylabel("z (km)")
        ax.set_aspect("equal")
    fig.colorbar(im0, ax=axs[:2], shrink=0.85, label="Velocity (km/s)")
    fig.colorbar(im2, ax=axs[2], shrink=0.85, label="Abs error (km/s)")
    fig.suptitle(f"{dataset} velocity surrogate, rel_l2={metrics['rel_l2']:.4e}")
    fig.savefig(os.path.join(fig_dir, f"{dataset}_velocity_surrogate.png"), dpi=180)
    plt.close(fig)

    if history["epoch"]:
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        ax.semilogy(history["epoch"], history["loss"], label="train MSE")
        if history["rel_l2"]:
            ax.semilogy(history["epoch"], history["rel_l2"], label="full-grid rel L2")
        ax.set_xlabel("epoch")
        ax.set_ylabel("metric")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"{dataset}_velocity_surrogate_history.png"), dpi=180)
        plt.close(fig)


def checkpoint_payload(model, optimizer, args, meta, velocity_shape, metrics, history):
    return {
        "model_state": model.state_dict(),
        "model_config": model.get_config(),
        "optimizer_state": optimizer.state_dict(),
        "dataset": args.dataset,
        "velocity_shape": tuple(int(v) for v in velocity_shape),
        "metadata": meta,
        "metrics": metrics,
        "history": history,
        "paper_reference": {
            "description": "FF-PINN velocity surrogate: coordinate-velocity label pairs, Gaussian Fourier feature MLP.",
            "sigma": 15.0,
            "hidden_layers": 5,
            "hidden_dim": 20,
            "optimizer": "Adam",
            "learning_rate": 5e-3,
            "epochs": 100000,
        },
    }


def train_dataset(dataset, base_args, device):
    args = argparse.Namespace(**vars(base_args))
    args.dataset = dataset
    velocity, meta = load_velocity_npz(dataset, args.data_root)
    coords, labels = make_coordinate_labels(velocity)
    coords_t = torch.from_numpy(coords).to(device)
    labels_t = torch.from_numpy(labels).to(device)

    out_dir = os.path.join(args.model_dir, "velocity_surrogate", dataset)
    fig_dir = os.path.join(args.figure_dir, "velocity_surrogate")
    os.makedirs(out_dir, exist_ok=True)

    model = create_model(args, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history = {"epoch": [], "loss": [], "rel_l2": [], "mae": [], "rmse": [], "lr": []}
    n = coords_t.shape[0]
    best_rel_l2 = float("inf")
    start_time = time.time()

    pbar = tqdm(range(1, args.epochs + 1), desc=f"{dataset} velocity", dynamic_ncols=True)
    for epoch in pbar:
        model.train()
        idx = torch.randint(0, n, (min(args.batch_size, n),), device=device)
        pred = model(coords_t[idx])
        loss = torch.mean((pred - labels_t[idx]) ** 2)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if epoch % args.eval_every == 0 or epoch == 1 or epoch == args.epochs:
            full_pred, metrics = evaluate_full_grid(model, coords, labels, device, args.eval_batch_size)
            best_rel_l2 = min(best_rel_l2, metrics["rel_l2"])
            history["epoch"].append(epoch)
            history["loss"].append(float(loss.detach().cpu()))
            history["rel_l2"].append(metrics["rel_l2"])
            history["mae"].append(metrics["mae"])
            history["rmse"].append(metrics["rmse"])
            history["lr"].append(float(optimizer.param_groups[0]["lr"]))
            pbar.set_postfix(loss=f"{loss.item():.3e}", rel_l2=f"{metrics['rel_l2']:.3e}")

            if epoch % args.save_every == 0 or epoch == args.epochs:
                torch.save(
                    checkpoint_payload(model, optimizer, args, meta, velocity.shape, metrics, history),
                    os.path.join(out_dir, f"checkpoint_epoch{epoch}.pt"),
                )

    pred, metrics = evaluate_full_grid(model, coords, labels, device, args.eval_batch_size)
    metrics["best_rel_l2"] = best_rel_l2
    metrics["train_time_sec"] = time.time() - start_time
    torch.save(
        checkpoint_payload(model, optimizer, args, meta, velocity.shape, metrics, history),
        os.path.join(out_dir, "velocity_surrogate.pt"),
    )
    save_diagnostics(dataset, velocity, pred, metrics, history, args, meta, out_dir, fig_dir)
    print(
        f"[{dataset}] saved {os.path.join(out_dir, 'velocity_surrogate.pt')} "
        f"rel_l2={metrics['rel_l2']:.4e}, mae={metrics['mae']:.4e}, rmse={metrics['rmse']:.4e}"
    )
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train FF-PINN paper-style velocity surrogate models.")
    parser.add_argument("--dataset", choices=("marmousi", "overthrust", "all"), default="all")
    parser.add_argument("--data-root", default="./datasets")
    parser.add_argument("--model-dir", default="./trained")
    parser.add_argument("--figure-dir", default="./figures")
    parser.add_argument("--epochs", type=int, default=PAPER_DEFAULTS["epochs"])
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=16384)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=10000)
    parser.add_argument("--learning-rate", type=float, default=PAPER_DEFAULTS["learning_rate"])
    parser.add_argument("--n-frequencies", type=int, default=PAPER_DEFAULTS["n_frequencies"])
    parser.add_argument("--sigma", type=float, default=PAPER_DEFAULTS["sigma"])
    parser.add_argument("--hidden-dim", type=int, default=PAPER_DEFAULTS["hidden_dim"])
    parser.add_argument("--n-hidden-layers", type=int, default=PAPER_DEFAULTS["n_hidden_layers"])
    parser.add_argument("--include-input", action="store_true")
    parser.add_argument("--feature-order", choices=("sin_cos", "cos_sin"), default="cos_sin")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    datasets = ("marmousi", "overthrust") if args.dataset == "all" else (args.dataset,)
    print(f"Using device: {device}")
    print(
        "Paper-style defaults: Gaussian Fourier features, "
        f"sigma={args.sigma}, hidden=[{args.hidden_dim}]x{args.n_hidden_layers}, "
        f"Adam lr={args.learning_rate}, epochs={args.epochs}"
    )
    all_metrics = {}
    for dataset in datasets:
        all_metrics[dataset] = train_dataset(dataset, args, device)
    print(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()
