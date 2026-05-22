import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compute_l2 import (
    prepare_dataset,
    align_temporal,
    create_model,
    load_best_checkpoint,
)


MODEL_DIR = "./trained"
OUT_DIR = "./figures/ablation"
DATASETS = ["threelayer", "marmousi", "overthrust"]


def predict_full(model, r_ref, batch_size=4096):
    outputs = []
    with torch.no_grad():
        for i in range(0, r_ref.shape[0], batch_size):
            outputs.append(model(r_ref[i:i + batch_size]).cpu())
    return torch.cat(outputs, dim=0).numpy().flatten()


def rel_l2(ref, est):
    ref_flat = ref.reshape(-1)
    est_flat = est.reshape(-1)
    return float(np.linalg.norm(est_flat - ref_flat) / (np.linalg.norm(ref_flat) + 1e-12))


def build_selected_time_errors(p_ref, p_est, time_indices):
    values = []
    for idx in time_indices:
        values.append(rel_l2(p_ref[:, :, idx], p_est[:, :, idx]))
    return values


def load_prediction(dataset, pure_kan, data, device):
    tag = "kan_pinn" if pure_kan else "kan_st_pinn"
    model_dir = os.path.join(MODEL_DIR, dataset, tag)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"missing model directory: {model_dir}")

    if pure_kan:
        model = create_model(True, device)
    else:
        dt_hist, seq_len = align_temporal(data)
        model = create_model(False, device, dt_hist=dt_hist, seq_len=seq_len)

    ckpt_name = load_best_checkpoint(model, model_dir)
    model.eval()

    n_l = data["n_L"]
    n_t = data["n_T"]
    pred = predict_full(model, data["r_ref"]).reshape(n_l, n_l, n_t)
    return pred, ckpt_name


def make_ablation_figure(dataset, data, pred_st, pred_kan):
    os.makedirs(OUT_DIR, exist_ok=True)

    p_ref = data["p_ref"]
    n_t = data["n_T"]
    t_phys = np.linspace(0.0, data.get("T_phys", float(data["t"][-1])), n_t)
    time_indices = [0, n_t // 4, n_t // 2, 3 * n_t // 4, n_t - 1]
    time_labels = [f"{t_phys[idx]:.2f}s" for idx in time_indices]

    global_st = rel_l2(p_ref, pred_st)
    global_kan = rel_l2(p_ref, pred_kan)
    per_t_st = build_selected_time_errors(p_ref, pred_st, time_indices)
    per_t_kan = build_selected_time_errors(p_ref, pred_kan, time_indices)

    p_max = float(np.max(np.abs(p_ref)))
    extent = [0, data.get("L_phys", data["L"]), 0, data.get("L_phys", data["L"])]

    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, len(time_indices) + 1, width_ratios=[1] * len(time_indices) + [0.06],
                          height_ratios=[1, 1, 1, 0.55], hspace=0.28, wspace=0.18)

    row_titles = [
        "Reference",
        f"KAN-ST-PINN  (Global Rel L2 = {global_st:.4f})",
        f"Pure-KAN  (Global Rel L2 = {global_kan:.4f})",
    ]
    row_arrays = [p_ref, pred_st, pred_kan]

    for row, arr_3d in enumerate(row_arrays):
        for col, tidx in enumerate(time_indices):
            ax = fig.add_subplot(gs[row, col])
            im = ax.imshow(
                arr_3d[:, :, tidx],
                cmap="seismic",
                origin="lower",
                extent=extent,
                vmin=-p_max,
                vmax=p_max,
                aspect="equal",
            )
            if row == 0:
                ax.set_title(f"t = {time_labels[col]}", fontsize=11)
            ax.set_xlabel("x (km)")
            if col == 0:
                ax.set_ylabel("y (km)")
                ax.text(
                    -0.32,
                    0.5,
                    row_titles[row],
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=12,
                    fontweight="bold",
                )
            else:
                ax.set_yticklabels([])

        cax = fig.add_subplot(gs[row, -1])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("Normalized pressure")

    ax_bar = fig.add_subplot(gs[3, :-1])
    x = np.arange(len(time_indices))
    width = 0.36
    bars_st = ax_bar.bar(x - width / 2, per_t_st, width, label="KAN-ST-PINN", color="#1f77b4", alpha=0.9)
    bars_kan = ax_bar.bar(x + width / 2, per_t_kan, width, label="Pure-KAN", color="#d62728", alpha=0.9)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"t = {label}" for label in time_labels])
    ax_bar.set_ylabel("Rel L2")
    ax_bar.set_title("Per-Timestep Error Comparison")
    ax_bar.grid(axis="y", alpha=0.3)
    ax_bar.legend()

    for bar in list(bars_st) + list(bars_kan):
        height = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width() / 2, height + 0.001, f"{height:.4f}",
                    ha="center", va="bottom", fontsize=9)

    fig.suptitle(f"Ablation Study - {dataset}", fontsize=18, fontweight="bold", y=0.98)

    out_path = os.path.join(OUT_DIR, f"ablation_{dataset}.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved {out_path}")

    return {
        "global_kan_st": global_st,
        "global_pure_kan": global_kan,
        "selected_times": time_labels,
        "per_t_kan_st": per_t_st,
        "per_t_pure_kan": per_t_kan,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    summary_lines = []

    for dataset in DATASETS:
        print(f"\nProcessing dataset: {dataset}")
        data = prepare_dataset(dataset, device)

        try:
            pred_st, ckpt_st = load_prediction(dataset, pure_kan=False, data=data, device=device)
            pred_kan, ckpt_kan = load_prediction(dataset, pure_kan=True, data=data, device=device)
        except FileNotFoundError as exc:
            print(f"[SKIP] {exc}")
            continue

        metrics = make_ablation_figure(dataset, data, pred_st, pred_kan)
        summary_lines.append(f"[{dataset}]")
        summary_lines.append(f"kan_st_checkpoint={ckpt_st}")
        summary_lines.append(f"pure_kan_checkpoint={ckpt_kan}")
        summary_lines.append(f"global_kan_st={metrics['global_kan_st']:.6f}")
        summary_lines.append(f"global_pure_kan={metrics['global_pure_kan']:.6f}")
        for label, st_val, kan_val in zip(metrics["selected_times"], metrics["per_t_kan_st"], metrics["per_t_pure_kan"]):
            summary_lines.append(f"t={label}, kan_st={st_val:.6f}, pure_kan={kan_val:.6f}")
        summary_lines.append("")

    if summary_lines:
        summary_path = os.path.join(OUT_DIR, "ablation_metrics.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines))
        print(f"\n[OK] saved {summary_path}")
    else:
        print("\nNo ablation figures were generated. Missing trained models.")


if __name__ == "__main__":
    main()
