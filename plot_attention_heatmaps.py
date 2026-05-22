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

from PINNs_util.kan_pinn import KANSTPinn
from train_kan_pinn import Config, prepare_dataset, align_temporal_config


TEMPORAL_MODE_TO_TAG = {
    "bilstm_attention": "kan_st_pinn",
    "attention_no_bilstm": "kan_attention_no_bilstm",
}


def create_model(config, device):
    if config.temporal_mode == "bilstm_no_attention":
        raise ValueError("bilstm_no_attention has no cross-attention weights to plot.")

    model = KANSTPinn(
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
        temporal_mode=config.temporal_mode,
    ).to(device)
    return model


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
    print(f"Loaded: {model_path}")


def get_xy(data):
    if "xy" in data:
        return data["xy"]
    if "xx" in data and "yy" in data:
        return np.column_stack((data["xx"].reshape(-1, 1), data["yy"].reshape(-1, 1)))
    raise KeyError("Dataset must provide either 'xy' or both 'xx' and 'yy'.")


def select_active_points(data, time_index, n_points):
    p_ref = data["p_ref"]
    n_l = data["n_L"]
    xy = get_xy(data)

    active = np.abs(p_ref[:, :, time_index]).reshape(-1)
    if np.all(active <= 1e-12):
        selected = np.linspace(0, n_l * n_l - 1, n_points, dtype=int)
    else:
        selected = np.argsort(active)[-n_points:]
        selected = selected[np.argsort(xy[selected, 1])]

    return selected, xy[selected]


def build_query_points(data, selected_xy, time_index, device):
    t_value = float(data["t"][time_index])
    t_col = np.full((selected_xy.shape[0], 1), t_value, dtype=np.float32)
    r = np.concatenate([selected_xy.astype(np.float32), t_col], axis=1)
    return torch.tensor(r, dtype=torch.float32, device=device)


def physical_lags(data, time_offsets):
    t_end = float(data["t"][-1])
    t_phys = float(data.get("T_phys", t_end))
    return (-time_offsets.detach().cpu().numpy()) / (t_end + 1e-12) * t_phys


def save_attention_over_time(attn, lags, selected_xy, out_dir):
    # [B, layers, heads, seq] -> [B, seq]
    heat = attn.mean(axis=(1, 2))

    fig, ax = plt.subplots(figsize=(8.5, max(4.0, 0.28 * heat.shape[0] + 1.5)))
    im = ax.imshow(heat, aspect="auto", cmap="viridis", vmin=0.0)
    ax.set_title("Attention over Historical Time")
    ax.set_xlabel("History lag (s)")
    ax.set_ylabel("Selected spatial point")
    ax.set_xticks(np.arange(len(lags)))
    ax.set_xticklabels([f"{v:.3f}" for v in lags], rotation=45, ha="right")

    step = max(1, heat.shape[0] // 12)
    yticks = np.arange(0, heat.shape[0], step)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"({selected_xy[i,0]:.2f},{selected_xy[i,1]:.2f})" for i in yticks])
    fig.colorbar(im, ax=ax, label="attention weight")
    fig.tight_layout()

    path = os.path.join(out_dir, "attention_over_historical_time.png")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"Saved: {path}")


def save_headwise_attention(attn, lags, out_dir):
    # [B, layers, heads, seq] -> [heads, seq]
    heat = attn.mean(axis=(0, 1))

    fig, ax = plt.subplots(figsize=(8.0, max(3.2, 0.55 * heat.shape[0] + 1.5)))
    im = ax.imshow(heat, aspect="auto", cmap="magma", vmin=0.0)
    ax.set_title("Head-wise Attention")
    ax.set_xlabel("History lag (s)")
    ax.set_ylabel("Attention head")
    ax.set_xticks(np.arange(len(lags)))
    ax.set_xticklabels([f"{v:.3f}" for v in lags], rotation=45, ha="right")
    ax.set_yticks(np.arange(heat.shape[0]))
    ax.set_yticklabels([f"head {i}" for i in range(heat.shape[0])])
    fig.colorbar(im, ax=ax, label="attention weight")
    fig.tight_layout()

    path = os.path.join(out_dir, "attention_headwise.png")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"Saved: {path}")


def save_layerwise_attention(attn, lags, out_dir):
    # [B, layers, heads, seq] -> [layers, seq]
    heat = attn.mean(axis=(0, 2))

    fig, ax = plt.subplots(figsize=(8.0, max(3.0, 0.7 * heat.shape[0] + 1.5)))
    im = ax.imshow(heat, aspect="auto", cmap="cividis", vmin=0.0)
    ax.set_title("Layer-wise Attention")
    ax.set_xlabel("History lag (s)")
    ax.set_ylabel("Attention layer")
    ax.set_xticks(np.arange(len(lags)))
    ax.set_xticklabels([f"{v:.3f}" for v in lags], rotation=45, ha="right")
    ax.set_yticks(np.arange(heat.shape[0]))
    ax.set_yticklabels([f"layer {i}" for i in range(heat.shape[0])])
    fig.colorbar(im, ax=ax, label="attention weight")
    fig.tight_layout()

    path = os.path.join(out_dir, "attention_layerwise.png")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"Saved: {path}")


def save_stats(attn, lags, out_dir):
    flat = attn.reshape(-1, attn.shape[-1])
    mean_w = flat.mean(axis=0)
    peak_idx = int(np.argmax(mean_w))
    entropy = -np.sum(mean_w * np.log(mean_w + 1e-12))

    path = os.path.join(out_dir, "attention_summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["peak_lag_s", f"{lags[peak_idx]:.8f}"])
        writer.writerow(["peak_weight", f"{mean_w[peak_idx]:.8f}"])
        writer.writerow(["entropy", f"{entropy:.8f}"])
        writer.writerow([])
        writer.writerow(["lag_s", "mean_attention"])
        for lag, value in zip(lags, mean_w):
            writer.writerow([f"{lag:.8f}", f"{value:.8f}"])
    print(f"Saved: {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot KAN-ST-PINN attention heatmaps.")
    parser.add_argument("--dataset", choices=["threelayer", "marmousi", "overthrust"], default="threelayer")
    parser.add_argument(
        "--temporal-mode",
        choices=["bilstm_attention", "attention_no_bilstm"],
        default="bilstm_attention",
        help="Only attention-enabled modes can produce attention heatmaps.",
    )
    parser.add_argument("--model-tag", default=None, help="Defaults to tag matching --temporal-mode.")
    parser.add_argument("--time-index", type=int, default=-1, help="Reference time index for selected query points.")
    parser.add_argument("--n-points", type=int, default=24, help="Number of active spatial points to visualize.")
    parser.add_argument("--model-dir", default="./trained")
    parser.add_argument("--figures-dir", default="./figures")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = Config()
    config.DATASET = args.dataset
    config.temporal_mode = args.temporal_mode

    data = prepare_dataset(config, device)
    align_temporal_config(config, data)

    time_index = args.time_index
    if time_index < 0:
        time_index = data["n_T"] + time_index
    if time_index < 0 or time_index >= data["n_T"]:
        raise ValueError(f"time_index must be in [0, {data['n_T'] - 1}], got {args.time_index}")

    model_tag = args.model_tag or TEMPORAL_MODE_TO_TAG[args.temporal_mode]
    model_dir = os.path.join(args.model_dir, args.dataset, model_tag)
    out_dir = os.path.join(args.figures_dir, args.dataset, model_tag, "attention_heatmaps")
    os.makedirs(out_dir, exist_ok=True)

    model = create_model(config, device)
    load_model(model, model_dir, device)
    model.eval()

    selected_idx, selected_xy = select_active_points(data, time_index, args.n_points)
    query = build_query_points(data, selected_xy, time_index, device)

    with torch.no_grad():
        _, info = model(query, return_attention=True)

    attn = info["attention"]
    if attn is None:
        raise RuntimeError(f"Model temporal_mode={args.temporal_mode} returned no attention maps.")

    attn_np = attn.detach().cpu().numpy()
    lags = physical_lags(data, info["time_offsets"])

    selected_path = os.path.join(out_dir, "selected_points.csv")
    with open(selected_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["flat_index", "x", "y", "time_index", "time_value"])
        for idx, xy in zip(selected_idx, selected_xy):
            writer.writerow([int(idx), f"{xy[0]:.8f}", f"{xy[1]:.8f}", time_index, f"{float(data['t'][time_index]):.8f}"])
    print(f"Saved: {selected_path}")

    save_attention_over_time(attn_np, lags, selected_xy, out_dir)
    save_headwise_attention(attn_np, lags, out_dir)
    save_layerwise_attention(attn_np, lags, out_dir)
    save_stats(attn_np, lags, out_dir)


if __name__ == "__main__":
    main()
