"""Plot per-timestep relative L2 curves for KAN-ST-PINN and Pure-KAN."""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from compute_l2 import (
    MODEL_DIR,
    align_temporal,
    create_model,
    evaluate,
    load_best_checkpoint,
    prepare_dataset,
)


DATASETS = ("threelayer", "marmousi", "overthrust")
DATASET_TITLES = {
    "threelayer": "Three-layer",
    "marmousi": "Marmousi-inspired",
    "overthrust": "Overthrust-inspired",
}
MODELS = {
    "kan_st_pinn": {
        "label": "KAN-ST-PINN",
        "color": "#0072B2",
        "linestyle": "-",
        "linewidth": 2.6,
    },
    "kan_pinn": {
        "label": "Pure-KAN",
        "color": "#D55E00",
        "linestyle": "--",
        "linewidth": 2.4,
    },
}


def physical_time_axis(data):
    t = np.asarray(data["t"], dtype=float)
    if t.size == 0:
        return t
    t_span = t[-1] - t[0]
    if abs(t_span) < 1e-12:
        return t
    t_phys = float(data.get("T_phys", t[-1]))
    return (t - t[0]) / t_span * t_phys


def evaluate_temporal_curve(model_tag, dataset, data, device):
    model_dir = os.path.join(MODEL_DIR, dataset, model_tag)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    if model_tag == "kan_pinn":
        model = create_model(True, device)
    elif model_tag == "kan_st_pinn":
        dt_hist, seq_len = align_temporal(data)
        model = create_model(
            False,
            device,
            dt_hist=dt_hist,
            seq_len=seq_len,
            temporal_mode="bilstm_attention",
        )
    else:
        raise ValueError(f"Unsupported model tag: {model_tag}")

    ckpt_name = load_best_checkpoint(model, model_dir)
    model.eval()
    metrics = evaluate(model, data, device)
    return np.asarray(metrics["per_timestep"], dtype=float), ckpt_name, metrics


def save_csv(curves, out_path):
    rows = ["dataset,time_s,KAN-ST-PINN,Pure-KAN"]
    for dataset, values in curves.items():
        t = values["time"]
        kan_st = values["kan_st_pinn"]
        pure_kan = values["kan_pinn"]
        for i in range(len(t)):
            rows.append(f"{dataset},{t[i]:.10g},{kan_st[i]:.10g},{pure_kan[i]:.10g}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")


def plot_curves(curves, out_path, dataset_order):
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 0.9,
        "figure.dpi": 160,
        "savefig.dpi": 300,
    })

    n_cols = len(dataset_order)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.55 * n_cols, 3.7), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, dataset in zip(axes, dataset_order):
        values = curves[dataset]
        for model_tag, style in MODELS.items():
            ax.plot(
                values["time"],
                values[model_tag],
                label=style["label"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
            )

        ax.set_title(DATASET_TITLES[dataset])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Relative $L_2$ error")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=1.6)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot temporal relative L2 curves for KAN-ST-PINN and Pure-KAN."
    )
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--out-dir", default="./figures/ablation_temporal")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    curves = {}
    for dataset in args.datasets:
        print(f"Preparing dataset: {dataset}")
        data = prepare_dataset(dataset, device)
        curves[dataset] = {"time": physical_time_axis(data)}

        for model_tag in MODELS:
            curve, ckpt_name, metrics = evaluate_temporal_curve(model_tag, dataset, data, device)
            curves[dataset][model_tag] = curve
            print(
                f"  {MODELS[model_tag]['label']}: loaded {ckpt_name}, "
                f"global Rel L2={metrics['rel_l2']:.6f}, mean temporal Rel L2={metrics['mean_per_t']:.6f}"
            )

    fig_path = os.path.join(args.out_dir, "kan_st_vs_pure_kan_temporal_rel_l2.png")
    csv_path = os.path.join(args.out_dir, "kan_st_vs_pure_kan_temporal_rel_l2.csv")
    plot_curves(curves, fig_path, args.datasets)
    save_csv(curves, csv_path)
    print(f"Saved: {fig_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
