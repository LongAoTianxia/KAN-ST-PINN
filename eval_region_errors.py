# 计算速度梯度分区，输出区域误差统计表和补充图片
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import csv
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from PINNs_util.kan_pinn import KANPinn, KANSTPinn
from train_kan_pinn import Config, prepare_dataset, align_temporal_config


MODEL_SPECS = {
    "kan_st_pinn": {
        "label": "KAN-ST-PINN",
        "pure_kan": False,
        "temporal_mode": "bilstm_attention",
    },
    "kan_pinn": {
        "label": "Pure-KAN",
        "pure_kan": True,
        "temporal_mode": None,
    },
    "kan_bilstm_no_attention": {
        "label": "KAN-BiLSTM w/o Attention",
        "pure_kan": False,
        "temporal_mode": "bilstm_no_attention",
    },
    "kan_attention_no_bilstm": {
        "label": "KAN-Attention w/o BiLSTM",
        "pure_kan": False,
        "temporal_mode": "attention_no_bilstm",
    },
}


def create_model(config, spec, device):
    if spec["pure_kan"]:
        return KANPinn(
            d_hidden=config.d_model,
            n_layers=config.spatial_kan_layers,
            n_fourier=config.n_fourier,
            grid_size=config.grid_size,
            spline_order=config.spline_order,
        ).to(device)

    return KANSTPinn(
        d_model=config.d_model,
        seq_len=config.seq_len,
        dt_hist=config.dt_hist,
        n_fourier=config.n_fourier,
        spatial_kan_layers=config.spatial_kan_layers,
        grid_size=config.grid_size,
        spline_order=config.spline_order,
        lstm_layers=config.lstm_layers,
        num_heads=config.num_heads,
        n_attn_layers=config.n_attn_layers,
        dropout=config.dropout,
        temporal_mode=spec["temporal_mode"],
    ).to(device)


def load_model(model, model_dir, device):
    model_path = os.path.join(model_dir, "model.pt")
    if not os.path.exists(model_path):
        ckpts = sorted(
            [
                name for name in os.listdir(model_dir)
                if name.startswith("checkpoint_epoch") and name.endswith(".pt")
            ],
            key=lambda name: int(name.replace("checkpoint_epoch", "").replace(".pt", "")),
        )
        if not ckpts:
            raise FileNotFoundError(f"No model.pt or checkpoint_epoch*.pt found in {model_dir}")
        model_path = os.path.join(model_dir, ckpts[-1])

    state = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    return model_path


def predict(model, r_ref, device, batch_size=4096):
    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, r_ref.shape[0], batch_size):
            preds.append(model(r_ref[i:i + batch_size]).detach().cpu())
    return torch.cat(preds, dim=0).numpy().reshape(-1)


def compute_velocity_and_gradient(data, device):
    xx = data["xx"]
    yy = data["yy"]
    with torch.no_grad():
        x_t = torch.tensor(xx, dtype=torch.float32, device=device)
        y_t = torch.tensor(yy, dtype=torch.float32, device=device)
        c = data["c_func"](x_t, y_t, tensor=True).detach().cpu().numpy()

    def grid_spacing(axis):
        dx_axis = np.diff(xx, axis=axis)
        dy_axis = np.diff(yy, axis=axis)
        dist = np.sqrt(dx_axis ** 2 + dy_axis ** 2)
        valid = dist[np.isfinite(dist) & (dist > 1e-12)]
        if valid.size == 0:
            return 1.0
        return float(np.mean(valid))

    h0 = grid_spacing(axis=0)
    h1 = grid_spacing(axis=1)
    dc_d0, dc_d1 = np.gradient(c, h0, h1, edge_order=1)
    dc_dx, dc_dy = dc_d0, dc_d1
    grad = np.sqrt(dc_dx ** 2 + dc_dy ** 2)
    grad = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    return c, grad


def build_region_masks(grad, high_q=0.80, low_q=0.40):
    finite_grad = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    high_thr = np.quantile(finite_grad, high_q)
    low_thr = np.quantile(finite_grad, low_q)
    masks = {
        "low_gradient": finite_grad <= low_thr,
        "mid_gradient": (finite_grad > low_thr) & (finite_grad < high_thr),
        "high_gradient": finite_grad >= high_thr,
    }

    if any(mask.sum() == 0 for mask in masks.values()):
        order = np.argsort(finite_grad.reshape(-1))
        n = order.size
        low_end = int(round(n * low_q))
        high_start = int(round(n * high_q))
        flat_masks = {
            "low_gradient": np.zeros(n, dtype=bool),
            "mid_gradient": np.zeros(n, dtype=bool),
            "high_gradient": np.zeros(n, dtype=bool),
        }
        flat_masks["low_gradient"][order[:low_end]] = True
        flat_masks["mid_gradient"][order[low_end:high_start]] = True
        flat_masks["high_gradient"][order[high_start:]] = True
        masks = {name: mask.reshape(finite_grad.shape) for name, mask in flat_masks.items()}

    return masks, {"low_threshold": low_thr, "high_threshold": high_thr}


def region_metrics(error_3d, ref_3d, masks, time_slice=None):
    rows = []
    if time_slice is None:
        time_indices = np.arange(ref_3d.shape[2])
        time_label = "all_time"
    else:
        time_indices = np.asarray(time_slice, dtype=int)
        time_label = "late_time"

    for region_name, mask in masks.items():
        err_vals = error_3d[:, :, time_indices][mask, :].reshape(-1)
        ref_vals = ref_3d[:, :, time_indices][mask, :].reshape(-1)
        mae = float(np.mean(np.abs(err_vals)))
        rel_l2 = float(np.linalg.norm(err_vals) / (np.linalg.norm(ref_vals) + 1e-12))
        rows.append({
            "time_group": time_label,
            "region": region_name,
            "rel_l2": rel_l2,
            "mae": mae,
            "num_points": int(mask.sum()),
            "num_values": int(err_vals.size),
        })
    return rows


def save_region_csv(dataset, metrics_by_model, out_dir):
    path = os.path.join(out_dir, "region_error_stats.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "dataset", "model_tag", "model", "time_group", "region",
            "rel_l2", "mae", "num_points", "num_values",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model_tag, item in metrics_by_model.items():
            for row in item["metrics"]:
                writer.writerow({
                    "dataset": dataset,
                    "model_tag": model_tag,
                    "model": item["label"],
                    **row,
                })
    print(f"Saved: {path}")


def save_comparison_csv(dataset, metrics_by_model, baseline_tag, target_tag, out_dir):
    baseline = metrics_by_model[baseline_tag]
    target = metrics_by_model[target_tag]
    base_map = {
        (row["time_group"], row["region"]): row
        for row in baseline["metrics"]
    }

    path = os.path.join(out_dir, "region_error_comparison.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "dataset", "time_group", "region",
            "baseline_model", "target_model",
            "baseline_rel_l2", "target_rel_l2", "rel_l2_improvement_pct",
            "baseline_mae", "target_mae", "mae_improvement_pct",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in target["metrics"]:
            key = (row["time_group"], row["region"])
            if key not in base_map:
                continue
            b = base_map[key]
            rel_imp = (b["rel_l2"] - row["rel_l2"]) / (b["rel_l2"] + 1e-12) * 100.0
            mae_imp = (b["mae"] - row["mae"]) / (b["mae"] + 1e-12) * 100.0
            writer.writerow({
                "dataset": dataset,
                "time_group": row["time_group"],
                "region": row["region"],
                "baseline_model": baseline["label"],
                "target_model": target["label"],
                "baseline_rel_l2": f"{b['rel_l2']:.8f}",
                "target_rel_l2": f"{row['rel_l2']:.8f}",
                "rel_l2_improvement_pct": f"{rel_imp:.4f}",
                "baseline_mae": f"{b['mae']:.8e}",
                "target_mae": f"{row['mae']:.8e}",
                "mae_improvement_pct": f"{mae_imp:.4f}",
            })
    print(f"Saved: {path}")


def save_region_barplot(metrics_by_model, out_dir, metric="rel_l2", time_group="all_time"):
    regions = ["low_gradient", "mid_gradient", "high_gradient"]
    model_tags = list(metrics_by_model.keys())
    width = 0.8 / len(model_tags)
    x = np.arange(len(regions))

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for i, model_tag in enumerate(model_tags):
        rows = {
            row["region"]: row
            for row in metrics_by_model[model_tag]["metrics"]
            if row["time_group"] == time_group
        }
        values = [rows[region][metric] for region in regions]
        offset = (i - (len(model_tags) - 1) / 2.0) * width
        ax.bar(x + offset, values, width=width, label=metrics_by_model[model_tag]["label"])

    ax.set_xticks(x)
    ax.set_xticklabels(["low gradient", "mid gradient", "high gradient"])
    ax.set_ylabel("Rel L2" if metric == "rel_l2" else "MAE")
    ax.set_title(f"Region-wise {ax.get_ylabel()} ({time_group.replace('_', ' ')})")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = os.path.join(out_dir, f"region_{metric}_{time_group}.png")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"Saved: {path}")


def save_gradient_region_map(grad, masks, out_dir):
    region_id = np.zeros_like(grad, dtype=float)
    region_id[masks["mid_gradient"]] = 1.0
    region_id[masks["high_gradient"]] = 2.0

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    im0 = axes[0].imshow(grad.T, origin="lower", cmap="viridis", aspect="equal")
    axes[0].set_title("Velocity-gradient magnitude")
    axes[0].set_xlabel("x index")
    axes[0].set_ylabel("y index")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(region_id.T, origin="lower", cmap="Set2", aspect="equal", vmin=0, vmax=2)
    axes[1].set_title("Gradient regions")
    axes[1].set_xlabel("x index")
    axes[1].set_ylabel("y index")
    cbar = fig.colorbar(im1, ax=axes[1], ticks=[0, 1, 2], fraction=0.046)
    cbar.ax.set_yticklabels(["low", "mid", "high"])
    fig.tight_layout()

    path = os.path.join(out_dir, "velocity_gradient_regions.png")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"Saved: {path}")


def save_local_error_maps(data, predictions, metrics_by_model, out_dir, time_indices):
    ref = data["p_ref"]
    n_t = data["n_T"]
    t_phys = data["t"] / (data["t"][-1] + 1e-12) * float(data.get("T_phys", data["t"][-1]))

    for idx in time_indices:
        if idx < 0:
            idx = n_t + idx
        if idx < 0 or idx >= n_t:
            continue

        model_tags = list(predictions.keys())
        fig, axes = plt.subplots(2, len(model_tags), figsize=(4.3 * len(model_tags), 7.2))
        if len(model_tags) == 1:
            axes = np.asarray(axes).reshape(2, 1)

        vmax_ref = np.max(np.abs(ref[:, :, idx]))
        err_max = max(np.max(np.abs(predictions[tag][:, :, idx] - ref[:, :, idx])) for tag in model_tags)

        for col, tag in enumerate(model_tags):
            label = metrics_by_model[tag]["label"]
            pred = predictions[tag][:, :, idx]
            err = pred - ref[:, :, idx]

            im0 = axes[0, col].imshow(pred.T, origin="lower", cmap="seismic",
                                      vmin=-vmax_ref, vmax=vmax_ref, aspect="equal")
            axes[0, col].set_title(f"{label}\nprediction")
            axes[0, col].set_xlabel("x index")
            axes[0, col].set_ylabel("y index")
            fig.colorbar(im0, ax=axes[0, col], fraction=0.046)

            im1 = axes[1, col].imshow(np.abs(err).T, origin="lower", cmap="inferno",
                                      vmin=0.0, vmax=err_max, aspect="equal")
            axes[1, col].set_title(f"{label}\nabsolute error")
            axes[1, col].set_xlabel("x index")
            axes[1, col].set_ylabel("y index")
            fig.colorbar(im1, ax=axes[1, col], fraction=0.046)

        fig.suptitle(f"Local prediction and error maps at t={t_phys[idx]:.3f}s", y=1.02)
        fig.tight_layout()
        path = os.path.join(out_dir, f"local_error_maps_t{idx:03d}.png")
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate region-wise errors in velocity-gradient regions.")
    parser.add_argument("--dataset", choices=["threelayer", "marmousi", "overthrust"], default="marmousi")
    parser.add_argument("--models", nargs="+", default=["kan_pinn", "kan_st_pinn"],
                        choices=list(MODEL_SPECS.keys()))
    parser.add_argument("--baseline", default="kan_pinn", choices=list(MODEL_SPECS.keys()))
    parser.add_argument("--target", default="kan_st_pinn", choices=list(MODEL_SPECS.keys()))
    parser.add_argument("--model-dir", default="./trained")
    parser.add_argument("--figures-dir", default="./figures")
    parser.add_argument("--high-quantile", type=float, default=0.80) # 高梯度  top 20%
    parser.add_argument("--low-quantile", type=float, default=0.40)  # 低梯度 bottom 40%
    parser.add_argument("--late-frac", type=float, default=0.25,
                        help="Last fraction of time steps used for late-time statistics.")
    parser.add_argument("--time-indices", nargs="*", type=int, default=None,
                        help="Time indices for local error maps. Defaults to middle and final steps.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = Config()
    config.DATASET = args.dataset
    data = prepare_dataset(config, device)
    align_temporal_config(config, data)

    c_field, grad = compute_velocity_and_gradient(data, device)
    masks, thresholds = build_region_masks(grad, high_q=args.high_quantile, low_q=args.low_quantile)

    out_dir = os.path.join(args.figures_dir, args.dataset, "region_error_analysis")
    os.makedirs(out_dir, exist_ok=True)

    save_gradient_region_map(grad, masks, out_dir)

    late_start = int(np.floor(data["n_T"] * (1.0 - args.late_frac)))
    late_indices = np.arange(max(0, late_start), data["n_T"])

    metrics_by_model = {}
    predictions = {}

    for model_tag in args.models:
        spec = MODEL_SPECS[model_tag]
        config.temporal_mode = spec["temporal_mode"] or "bilstm_attention"
        model = create_model(config, spec, device)

        model_path = load_model(model, os.path.join(args.model_dir, args.dataset, model_tag), device)
        print(f"Loaded {spec['label']}: {model_path}")

        pred = predict(model, data["r_ref"], device).reshape(data["n_L"], data["n_L"], data["n_T"])
        predictions[model_tag] = pred
        err = pred - data["p_ref"]

        metrics = []
        metrics.extend(region_metrics(err, data["p_ref"], masks, time_slice=None))
        metrics.extend(region_metrics(err, data["p_ref"], masks, time_slice=late_indices))

        metrics_by_model[model_tag] = {
            "label": spec["label"],
            "metrics": metrics,
            "model_path": model_path,
        }

    save_region_csv(args.dataset, metrics_by_model, out_dir)
    if args.baseline in metrics_by_model and args.target in metrics_by_model:
        save_comparison_csv(args.dataset, metrics_by_model, args.baseline, args.target, out_dir)

    save_region_barplot(metrics_by_model, out_dir, metric="rel_l2", time_group="all_time")
    save_region_barplot(metrics_by_model, out_dir, metric="mae", time_group="all_time")
    save_region_barplot(metrics_by_model, out_dir, metric="rel_l2", time_group="late_time")
    save_region_barplot(metrics_by_model, out_dir, metric="mae", time_group="late_time")

    if args.time_indices is None:
        time_indices = [data["n_T"] // 2, data["n_T"] - 1]
    else:
        time_indices = args.time_indices
    save_local_error_maps(data, predictions, metrics_by_model, out_dir, time_indices)


if __name__ == "__main__":
    main()
