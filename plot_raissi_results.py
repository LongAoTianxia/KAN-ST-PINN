import argparse
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from train_raissi_pinn import Config, create_model, prepare_dataset


plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
    }
)


def smooth_curve(values, max_points=2500):
    values = np.asarray(values, dtype=np.float64)
    if values.size <= max_points:
        return np.arange(values.size), values

    window = max(1, values.size // max_points)
    trim = (values.size // window) * window
    y = values[:trim].reshape(-1, window).mean(axis=1)
    x = np.arange(y.size) * window + window // 2
    return x, y


def load_history(model_dir):
    history_path = os.path.join(model_dir, "history.pt")
    model_path = os.path.join(model_dir, "model.pt")
    if os.path.exists(history_path):
        return torch.load(history_path, map_location="cpu", weights_only=False)
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    return state.get("history", {})


def load_raissi_model(model_dir, device):
    model_path = os.path.join(model_dir, "model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    state = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg = Config()
    model_cfg = state.get("model_config", {})
    cfg.n_hidden = int(model_cfg.get("n_hidden", cfg.n_hidden))
    cfg.n_layers = int(model_cfg.get("n_layers", cfg.n_layers))
    cfg.n_ffeatures = int(model_cfg.get("n_ffeatures", cfg.n_ffeatures))

    model = create_model(cfg, device)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model, state


def predict(model, data, device, batch_size=4096):
    outputs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, data["r_ref"].shape[0], batch_size):
            outputs.append(model(data["r_ref"][i : i + batch_size]).cpu())
    return torch.cat(outputs, dim=0).numpy().reshape(-1)


def plot_loss_curves(history, out_path, dataset_name):
    keys = [
        ("loss_ini", "Initial Condition"),
        ("loss_data", "Data Supervision"),
        ("loss_pde", "PDE Residual"),
        ("loss_bc", "Boundary"),
        ("loss_total", "Total"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    axes = axes.flatten()

    for ax, (key, title) in zip(axes, keys):
        vals = history.get(key, [])
        if len(vals) == 0:
            ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue
        x, y = smooth_curve(vals)
        y = np.maximum(y, 1e-16)
        ax.plot(x, y, linewidth=1.4)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)

    axes[-1].axis("off")
    fig.suptitle(f"Raissi-PINN Loss Curves - {dataset_name}", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def plot_wavefield_comparison(p_est, data, out_path, dataset_name):
    n_l = int(data["n_L"])
    n_t = int(data["n_T"])
    p_ref = data["p_ref"]
    p_pred = p_est.reshape(n_l, n_l, n_t)
    err = p_pred - p_ref

    p_scale = max(float(np.max(np.abs(p_ref))), float(np.max(np.abs(p_pred))), 1e-12)
    e_scale = max(float(np.max(np.abs(err))), 1e-12)

    l_phys = float(data.get("L_phys", data["L"]))
    t_phys_end = float(data.get("T_phys", float(data["t"][-1])))
    t_phys = np.linspace(0.0, t_phys_end, n_t)

    time_indices = sorted(set([0, n_t // 4, n_t // 2, 3 * n_t // 4, n_t - 1]))
    n_cols = len(time_indices)
    row_labels = ["Reference", "Raissi-PINN", "Error"]

    fig = plt.figure(figsize=(4.0 * n_cols + 1.2, 10.5))
    gs_main = GridSpec(3, 1, figure=fig, hspace=0.34, top=0.92, bottom=0.05, left=0.07, right=0.88)

    for row in range(3):
        gs_row = gs_main[row].subgridspec(1, n_cols + 1, width_ratios=[1] * n_cols + [0.05], wspace=0.22)
        im = None
        for col, tidx in enumerate(time_indices):
            ax = fig.add_subplot(gs_row[0, col])
            if row == 0:
                img = p_ref[:, :, tidx]
                vmin, vmax = -p_scale, p_scale
                cb_label = "Pressure"
            elif row == 1:
                img = p_pred[:, :, tidx]
                vmin, vmax = -p_scale, p_scale
                cb_label = "Pressure"
            else:
                img = err[:, :, tidx]
                vmin, vmax = -e_scale, e_scale
                cb_label = "Error"

            im = ax.imshow(
                img,
                extent=[0, l_phys, 0, l_phys],
                origin="lower",
                cmap="seismic",
                vmin=vmin,
                vmax=vmax,
                aspect="equal",
            )

            if row == 0:
                ax.set_title(f"t = {t_phys[tidx]:.2f} s")
            if row == 2:
                local_ref = p_ref[:, :, tidx]
                local_err = err[:, :, tidx]
                rel = np.linalg.norm(local_err) / (np.linalg.norm(local_ref) + 1e-12)
                ax.set_title(f"Rel L2 = {rel:.4f}")
            ax.set_xlabel("x (km)")
            if col == 0:
                ax.set_ylabel("y (km)")
            else:
                ax.set_yticklabels([])

        first_ax = fig.axes[-n_cols]
        first_ax.text(
            -0.30,
            0.5,
            row_labels[row],
            transform=first_ax.transAxes,
            fontsize=13,
            va="center",
            ha="center",
            rotation=90,
            fontweight="bold",
        )
        cax = fig.add_subplot(gs_row[0, n_cols])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(cb_label)

    rel_l2 = np.linalg.norm(p_pred.reshape(-1) - p_ref.reshape(-1)) / (np.linalg.norm(p_ref.reshape(-1)) + 1e-12)
    mae = np.mean(np.abs(p_pred.reshape(-1) - p_ref.reshape(-1)))
    fig.suptitle(
        f"Wavefield Comparison - {dataset_name}  (Rel L2={rel_l2:.6f}, MAE={mae:.3e})",
        fontsize=15,
        fontweight="bold",
    )
    plt.savefig(out_path)
    plt.close(fig)
    return rel_l2, mae


def visualize_dataset(dataset_name, args, device):
    model_dir = os.path.join(args.model_dir, dataset_name, "raissi_pinn_adaptive_nobc")
    fig_dir = os.path.join(args.fig_dir, "raissi_pinn_adaptive_nobc", dataset_name)
    os.makedirs(fig_dir, exist_ok=True)

    if not os.path.exists(os.path.join(model_dir, "model.pt")):
        print(f"[SKIP] {dataset_name}: missing {model_dir}/model.pt")
        return

    print(f"[{dataset_name}] loading data/model...")
    cfg = Config()
    cfg.DATASET = dataset_name
    cfg.DATA_ROOT = args.data_root
    data = prepare_dataset(cfg, device)
    history = load_history(model_dir)
    model, state = load_raissi_model(model_dir, device)

    loss_path = os.path.join(fig_dir, "loss_curves.png")
    plot_loss_curves(history, loss_path, dataset_name)
    print(f"  saved loss curves: {loss_path}")

    p_est = predict(model, data, device, batch_size=args.batch_size)
    wave_path = os.path.join(fig_dir, "wavefield_comparison.png")
    rel_l2, mae = plot_wavefield_comparison(p_est, data, wave_path, dataset_name)
    print(f"  saved wavefield comparison: {wave_path}")
    print(f"  metrics: Rel L2={rel_l2:.6f}, MAE={mae:.6e}, epoch={state.get('epoch')}")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Raissi-PINN loss curves and wavefield comparison.")
    parser.add_argument("--dataset", choices=["threelayer", "marmousi", "overthrust", "all"], default="all")
    parser.add_argument("--model-dir", default="./trained")
    parser.add_argument("--data-root", default="./datasets")
    parser.add_argument("--fig-dir", default="./figures")
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = ["threelayer", "marmousi", "overthrust"] if args.dataset == "all" else [args.dataset]
    print(f"Device: {device}")
    for dataset_name in datasets:
        visualize_dataset(dataset_name, args, device)


if __name__ == "__main__":
    main()
