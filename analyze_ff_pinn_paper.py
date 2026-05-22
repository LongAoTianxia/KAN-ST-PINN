"""
Analyze FF-PINN training checkpoints.

The script reads trained/<dataset>/<model_tag>/model.pt, checkpoint_epoch*.pt
and history.pt, then creates figures and a Markdown report focused on why the
FF-PINN variant may underfit the current npz reference wavefields.

Example:
    conda run -n d2l python analyze_ff_pinn_paper.py --dataset all
"""
import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from PINNs_util.ff_pinn import HardInitialFFPINN
from compute_l2 import evaluate_pde_residual, load_npz_dataset


DATASETS = ("threelayer", "marmousi", "overthrust")
DEFAULT_MODEL_ROOT = "./trained"
DEFAULT_DATA_ROOT = "./datasets"
DEFAULT_OUT_DIR = "./figures/ff_pinn_paper_analysis"


@dataclass
class CheckpointEval:
    epoch: int
    path: str
    stage: int
    t_limit: float
    rel_l2: float
    active_rel_l2: float
    mae: float
    amp_ratio: float
    rms_ratio: float
    corr: float
    pde_rmse: float
    pde_mae: float
    pde_max_abs: float


def checkpoint_epoch(path):
    name = os.path.basename(path)
    match = re.search(r"checkpoint_epoch(\d+)\.pt$", name)
    if not match:
        return None
    return int(match.group(1))


def find_checkpoints(model_dir):
    if not os.path.isdir(model_dir):
        return []
    paths = []
    for name in os.listdir(model_dir):
        if name.startswith("checkpoint_epoch") and name.endswith(".pt"):
            paths.append(os.path.join(model_dir, name))
    return sorted(paths, key=lambda p: checkpoint_epoch(p) or -1)


def load_state(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def load_model_from_state(state, device):
    if not isinstance(state, dict) or "model_config" not in state:
        raise ValueError("FF-PINN checkpoint must contain model_config")
    model = HardInitialFFPINN(**state["model_config"]).to(device)
    model.load_state_dict(state["model_state"] if "model_state" in state else state)
    model._training_config = state.get("training_config", {})
    model.eval()
    return model


def load_model(path, device):
    state = load_state(path)
    return load_model_from_state(state, device), state


def safe_history(model_dir):
    history_path = os.path.join(model_dir, "history.pt")
    if not os.path.exists(history_path):
        return {}
    history = torch.load(history_path, map_location="cpu", weights_only=False)
    return history if isinstance(history, dict) else {}


def stage_spec(training_config, final_epoch):
    stage_epochs = training_config.get("stage_epochs") or (5000, 10000, 10000, 20000)
    stage_fractions = training_config.get("stage_fractions") or (0.3, 0.4, 0.5, 1.0)
    stage_epochs = tuple(int(x) for x in stage_epochs)
    stage_fractions = tuple(float(x) for x in stage_fractions)
    if len(stage_epochs) != len(stage_fractions):
        stage_epochs = (final_epoch,)
        stage_fractions = (1.0,)
    boundaries = np.cumsum(stage_epochs)
    if final_epoch and boundaries[-1] != final_epoch:
        boundaries[-1] = final_epoch
    return stage_epochs, stage_fractions, boundaries


def stage_for_epoch(epoch, boundaries):
    for i, boundary in enumerate(boundaries, start=1):
        if epoch <= boundary:
            return i
    return len(boundaries)


def t_limit_for_epoch(epoch, data, stage_fractions, boundaries):
    stage = stage_for_epoch(epoch, boundaries)
    frac = stage_fractions[stage - 1]
    t_max = float(data["t"][-1])
    return min(t_max, max(float(data["t"][1]), t_max * frac))


def choose_checkpoints(paths, boundaries, per_stage):
    if not paths:
        return []
    epochs = np.array([checkpoint_epoch(p) for p in paths], dtype=int)
    selected = set()
    start = 0
    for boundary in boundaries:
        in_stage = np.where((epochs > start) & (epochs <= boundary))[0]
        if in_stage.size:
            count = min(per_stage, in_stage.size)
            pick = np.linspace(0, in_stage.size - 1, count).round().astype(int)
            for idx in in_stage[pick]:
                selected.add(int(idx))
            selected.add(int(in_stage[-1]))
        start = boundary
    return [paths[i] for i in sorted(selected)]


def predict_flat(model, r_ref, batch_size):
    preds = []
    with torch.no_grad():
        for start in range(0, r_ref.shape[0], batch_size):
            preds.append(model(r_ref[start:start + batch_size]).detach().cpu())
    return torch.cat(preds, dim=0).numpy().reshape(-1)


def checkpoint_metrics(model, state, path, data, device, args, stage_fractions, boundaries):
    epoch = int(state.get("epoch", checkpoint_epoch(path) or 0))
    pred = predict_flat(model, data["r_ref"], args.batch_size)
    ref = data["p_ref"].reshape(-1)
    err = pred - ref
    ref_norm = np.linalg.norm(ref) + 1e-12

    t_limit = t_limit_for_epoch(epoch, data, stage_fractions, boundaries)
    n_space = int(data["n_L"] * data["n_L"])
    t_mask = data["t"] <= t_limit + 1e-12
    active_mask = np.repeat(t_mask.reshape(1, -1), n_space, axis=0).reshape(-1)
    active_ref = ref[active_mask]
    active_pred = pred[active_mask]
    active_rel = np.linalg.norm(active_pred - active_ref) / (np.linalg.norm(active_ref) + 1e-12)

    pred_centered = pred - pred.mean()
    ref_centered = ref - ref.mean()
    corr = float(
        np.dot(pred_centered, ref_centered) /
        ((np.linalg.norm(pred_centered) * np.linalg.norm(ref_centered)) + 1e-12)
    )

    pde_rmse = pde_mae = pde_max_abs = float("nan")
    if not args.skip_pde:
        pde = evaluate_pde_residual(
            model,
            data,
            device,
            max_points=args.pde_points,
            batch_size=args.pde_batch_size,
        )
        pde_rmse = float(pde["pde_rmse"])
        pde_mae = float(pde["pde_mae"])
        pde_max_abs = float(pde["pde_max_abs"])

    return CheckpointEval(
        epoch=epoch,
        path=path,
        stage=stage_for_epoch(epoch, boundaries),
        t_limit=t_limit,
        rel_l2=float(np.linalg.norm(err) / ref_norm),
        active_rel_l2=float(active_rel),
        mae=float(np.mean(np.abs(err))),
        amp_ratio=float((np.max(np.abs(pred)) + 1e-12) / (np.max(np.abs(ref)) + 1e-12)),
        rms_ratio=float((np.sqrt(np.mean(pred ** 2)) + 1e-12) / (np.sqrt(np.mean(ref ** 2)) + 1e-12)),
        corr=corr,
        pde_rmse=pde_rmse,
        pde_mae=pde_mae,
        pde_max_abs=pde_max_abs,
    )


def final_temporal_metrics(model, data, args):
    pred = predict_flat(model, data["r_ref"], args.batch_size)
    n_l = int(data["n_L"])
    n_t = int(data["n_T"])
    pred_3d = pred.reshape(n_l, n_l, n_t)
    ref_3d = data["p_ref"]

    rel = []
    mae = []
    amp_ratio = []
    energy_ratio = []
    for i in range(n_t):
        p = pred_3d[:, :, i].reshape(-1)
        r = ref_3d[:, :, i].reshape(-1)
        rel.append(float(np.linalg.norm(p - r) / (np.linalg.norm(r) + 1e-12)))
        mae.append(float(np.mean(np.abs(p - r))))
        amp_ratio.append(float((np.max(np.abs(p)) + 1e-12) / (np.max(np.abs(r)) + 1e-12)))
        energy_ratio.append(float((np.linalg.norm(p) + 1e-12) / (np.linalg.norm(r) + 1e-12)))
    return {
        "pred_3d": pred_3d,
        "rel_l2_t": np.array(rel),
        "mae_t": np.array(mae),
        "amp_ratio_t": np.array(amp_ratio),
        "energy_ratio_t": np.array(energy_ratio),
    }


def source_coverage(training_config, dataset, n=200000, seed=0):
    if training_config.get("problem_form") != "source":
        return None
    params = training_config.get("source_params") or {}
    if not params:
        return None
    rng = np.random.default_rng(seed)
    xy = rng.random((n, 2))
    xs, ys = params.get("center", (0.5, 0.5))
    alpha = float(params.get("alpha", 0.01))
    spatial = np.exp(-0.5 * (((xy[:, 0] - xs) / alpha) ** 2 + ((xy[:, 1] - ys) / alpha) ** 2))
    return {
        "dataset": dataset,
        "alpha": alpha,
        "frac_source_gt_1e-2": float(np.mean(spatial > 1e-2)),
        "frac_source_gt_1e-3": float(np.mean(spatial > 1e-3)),
        "expected_points_gt_1e-3_per_Nr": float(np.mean(spatial > 1e-3) * 80000),
    }


def plot_history(dataset, model_tag, history, boundaries, out_dir):
    if not history:
        return None
    epochs = np.arange(1, len(history.get("loss_total", [])) + 1)
    if epochs.size == 0:
        return None

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for key, label in [("loss_total", "total"), ("loss_pde", "PDE"), ("loss_bc", "ABC")]:
        values = np.asarray(history.get(key, []), dtype=float)
        if values.size:
            axes[0].semilogy(epochs[:values.size], values, label=label, linewidth=1.2)
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    t_limit = np.asarray(history.get("t_limit", []), dtype=float)
    if t_limit.size:
        axes[1].plot(epochs[:t_limit.size], t_limit, color="tab:green", linewidth=1.2)
    axes[1].set_ylabel("t_limit")
    axes[1].grid(True, alpha=0.25)

    lr = np.asarray(history.get("lr", []), dtype=float)
    if lr.size:
        axes[2].semilogy(epochs[:lr.size], lr, color="tab:orange", linewidth=1.2)
    axes[2].set_ylabel("learning rate")
    axes[2].set_xlabel("epoch")
    axes[2].grid(True, alpha=0.25)

    for ax in axes:
        for boundary in boundaries:
            ax.axvline(boundary, color="k", linestyle="--", linewidth=0.8, alpha=0.35)
    fig.suptitle(f"{dataset} {model_tag} training history")
    fig.tight_layout()
    path = os.path.join(out_dir, f"{dataset}_history.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_checkpoint_metrics(dataset, model_tag, rows, boundaries, out_dir):
    if not rows:
        return None
    x = np.array([r.epoch for r in rows])
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)

    axes[0].plot(x, [r.rel_l2 for r in rows], marker="o", label="global Rel L2")
    axes[0].plot(x, [r.active_rel_l2 for r in rows], marker="s", label="active-window Rel L2")
    axes[0].set_ylabel("Rel L2")
    axes[0].legend()

    axes[1].plot(x, [r.amp_ratio for r in rows], marker="o", label="max amplitude ratio")
    axes[1].plot(x, [r.rms_ratio for r in rows], marker="s", label="RMS ratio")
    axes[1].axhline(1.0, color="k", linewidth=0.8, alpha=0.4)
    axes[1].set_ylabel("pred/ref")
    axes[1].legend()

    axes[2].plot(x, [r.corr for r in rows], marker="o", color="tab:purple")
    axes[2].set_ylabel("correlation")

    axes[3].plot(x, [r.pde_rmse for r in rows], marker="o", color="tab:red")
    axes[3].set_ylabel("PDE RMSE")
    axes[3].set_xlabel("epoch")
    if np.isfinite([r.pde_rmse for r in rows]).any():
        axes[3].set_yscale("log")

    for ax in axes:
        ax.grid(True, alpha=0.25)
        for boundary in boundaries:
            ax.axvline(boundary, color="k", linestyle="--", linewidth=0.8, alpha=0.35)
    fig.suptitle(f"{dataset} {model_tag} checkpoint diagnostics")
    fig.tight_layout()
    path = os.path.join(out_dir, f"{dataset}_checkpoint_diagnostics.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_temporal(dataset, data, temporal, boundaries, stage_fractions, out_dir):
    t = data["t"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(t, temporal["rel_l2_t"], color="tab:blue")
    axes[0].set_ylabel("Rel L2(t)")
    axes[1].plot(t, temporal["amp_ratio_t"], color="tab:orange")
    axes[1].axhline(1.0, color="k", linewidth=0.8, alpha=0.4)
    axes[1].set_ylabel("max amp ratio")
    axes[2].plot(t, temporal["energy_ratio_t"], color="tab:green")
    axes[2].axhline(1.0, color="k", linewidth=0.8, alpha=0.4)
    axes[2].set_ylabel("energy ratio")
    axes[2].set_xlabel("normalized time")

    t_max = float(t[-1])
    for ax in axes:
        for frac in stage_fractions:
            ax.axvline(t_max * frac, color="k", linestyle="--", linewidth=0.8, alpha=0.25)
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"{dataset} final temporal error")
    fig.tight_layout()
    path = os.path.join(out_dir, f"{dataset}_temporal_error.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_snapshots(dataset, data, temporal, stage_fractions, out_dir):
    pred = temporal["pred_3d"]
    ref = data["p_ref"]
    t = data["t"]
    t_max = float(t[-1])
    target_times = sorted(set([0.0] + [t_max * f for f in stage_fractions]))
    idxs = [int(np.argmin(np.abs(t - tt))) for tt in target_times]
    idxs = sorted(set(idxs))
    n_cols = len(idxs)
    fig, axes = plt.subplots(3, n_cols, figsize=(3.2 * n_cols, 8.0), squeeze=False)
    vmax = max(float(np.max(np.abs(ref))), float(np.max(np.abs(pred))), 1e-9)
    err_vmax = max(float(np.max(np.abs(pred - ref))), 1e-9)
    for col, idx in enumerate(idxs):
        axes[0, col].imshow(ref[:, :, idx], origin="lower", cmap="seismic", vmin=-vmax, vmax=vmax)
        axes[0, col].set_title(f"ref t={t[idx]:.3f}")
        axes[1, col].imshow(pred[:, :, idx], origin="lower", cmap="seismic", vmin=-vmax, vmax=vmax)
        axes[1, col].set_title("pred")
        axes[2, col].imshow(pred[:, :, idx] - ref[:, :, idx], origin="lower", cmap="seismic", vmin=-err_vmax, vmax=err_vmax)
        axes[2, col].set_title("pred-ref")
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{dataset} final checkpoint snapshots")
    fig.tight_layout()
    path = os.path.join(out_dir, f"{dataset}_snapshots.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def trend(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return 0.0
    return float(arr[-1] - arr[0])


def diagnose(dataset, final_row, rows, history, training_config, coverage):
    findings = []
    recommendations = []

    problem_form = training_config.get("problem_form")
    if problem_form == "source":
        findings.append(
            "当前 ff_pinn_paper 按论文的零初值 + Ricker 源项 PDE 训练；但本项目 npz 参考场来自初始高斯位移问题。"
            " 这会造成训练目标与 compute_l2 的参考解不完全同构，是 Rel L2 偏大的首要根因。"
        )
        recommendations.append(
            "若目标是评估本项目 npz 数据精度，应训练 `--problem-form initial --model-tag ff_pinn_initial_ablation`；"
            "若目标是严格论文复现，应重新生成零初值 + Ricker 源项 + PML/FDM 参考数据后再用 compute_l2 对齐评估。"
        )

    if final_row.amp_ratio < 0.5 or final_row.rms_ratio < 0.5:
        findings.append(
            f"最终预测振幅明显不足：max amp ratio={final_row.amp_ratio:.3f}, RMS ratio={final_row.rms_ratio:.3f}，"
            "说明网络趋向低能量/近零波场。"
        )
        recommendations.append(
            "增加源区有效约束：对源附近 collocation 进行分层采样，或提高 Ricker 源项幅值/归一化到与 p_ref 相同尺度；"
            "同时监控预测能量，避免只靠 PDE/ABC 残差收敛到低振幅解。"
        )

    if final_row.corr < 0.5:
        findings.append(
            f"最终预测与参考场相关系数较低：corr={final_row.corr:.3f}，不仅是尺度问题，也存在相位/波形路径不一致。"
        )
        recommendations.append(
            "优先统一 PDE 形式、源函数、初值和边界条件；这些物理设定不一致时，继续加 epoch 或加宽网络通常不能根治。"
        )

    if coverage and coverage["expected_points_gt_1e-3_per_Nr"] < 200:
        findings.append(
            "Ricker 空间源非常窄，随机采样中真正落入源区的点偏少："
            f"预计每 80000 个点只有约 {coverage['expected_points_gt_1e-3_per_Nr']:.1f} 个满足 G>1e-3。"
        )
        recommendations.append(
            "把 collocation 拆成全域点 + 源区点两部分，例如 70% 全域、30% 在源中心附近高斯/均匀采样，仍可标注为 paper+tuned。"
        )

    rel_slope = trend([r.rel_l2 for r in rows[-min(5, len(rows)):]])
    if rel_slope < -1e-3:
        findings.append("最后若干 checkpoint 的 Rel L2 仍在下降，训练可能尚未充分收敛。")
        recommendations.append(
            "延长最后阶段训练，或减慢学习率指数衰减；当前 45000 epoch 后 lr 已被衰减很多，后期更新可能过弱。"
        )
    elif abs(rel_slope) <= 1e-3:
        findings.append("最后若干 checkpoint 的 Rel L2 基本平台化，单纯延长训练的收益可能有限。")

    if history:
        pde = np.asarray(history.get("loss_pde", []), dtype=float)
        bc = np.asarray(history.get("loss_bc", []), dtype=float)
        if pde.size and bc.size:
            ratio = float(np.nanmedian(bc[-min(1000, bc.size):] / (pde[-min(1000, pde.size):] + 1e-12)))
            if ratio > 100 or ratio < 0.01:
                findings.append(f"后期 PDE/ABC 损失尺度明显失衡，median(BC/PDE)≈{ratio:.3e}。")
                recommendations.append(
                    "实现论文附录提到的 NTK/adaptive loss weighting，或至少按 running RMS 对 PDE 与 ABC 残差归一化。"
                )

    if not findings:
        findings.append("未触发明显的单点异常；建议结合图中的阶段曲线继续检查误差从哪个时间窗开始放大。")
    if not recommendations:
        recommendations.append("优先对齐训练 PDE 与评估参考数据，再考虑增加网络容量或训练轮数。")
    return findings, recommendations


def write_dataset_report(path, dataset, model_tag, model_dir, training_config, rows, temporal_summary, coverage, findings, recommendations, figures):
    final = rows[-1]
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# FF-PINN Analysis: {model_tag} / {dataset}\n\n")
        f.write(f"- model_dir: `{model_dir}`\n")
        f.write(f"- problem_form: `{training_config.get('problem_form', 'unknown')}`\n")
        f.write(f"- final_epoch: `{final.epoch}`\n")
        f.write(f"- final Rel L2: `{final.rel_l2:.6f}`\n")
        f.write(f"- final MAE: `{final.mae:.6e}`\n")
        f.write(f"- final amp ratio: `{final.amp_ratio:.6f}`\n")
        f.write(f"- final RMS ratio: `{final.rms_ratio:.6f}`\n")
        f.write(f"- final corr: `{final.corr:.6f}`\n")
        f.write(f"- final PDE RMSE: `{final.pde_rmse:.6e}`\n\n")

        f.write("## Figures\n\n")
        for label, fig_path in figures.items():
            if fig_path:
                f.write(f"- {label}: `{fig_path}`\n")
        f.write("\n## Checkpoint Metrics\n\n")
        f.write("| epoch | stage | t_limit | rel_l2 | active_rel_l2 | amp_ratio | rms_ratio | corr | pde_rmse |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r.epoch} | {r.stage} | {r.t_limit:.4f} | {r.rel_l2:.6f} | "
                f"{r.active_rel_l2:.6f} | {r.amp_ratio:.6f} | {r.rms_ratio:.6f} | "
                f"{r.corr:.6f} | {r.pde_rmse:.6e} |\n"
            )
        f.write("\n## Temporal Summary\n\n")
        for key, value in temporal_summary.items():
            f.write(f"- {key}: `{value}`\n")
        if coverage:
            f.write("\n## Source Coverage\n\n")
            for key, value in coverage.items():
                f.write(f"- {key}: `{value}`\n")
        f.write("\n## Root-Cause Findings\n\n")
        for item in findings:
            f.write(f"- {item}\n")
        f.write("\n## Improvement Measures\n\n")
        for item in recommendations:
            f.write(f"- {item}\n")


def save_rows_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(CheckpointEval.__annotations__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def analyze_dataset(dataset, args, device):
    model_dir = os.path.join(args.model_root, dataset, args.model_tag)
    final_path = os.path.join(model_dir, "model.pt")
    if not os.path.exists(final_path):
        print(f"[SKIP] {dataset}: missing {final_path}")
        return None

    out_dir = os.path.abspath(os.path.join(args.out_dir, dataset))
    os.makedirs(out_dir, exist_ok=True)

    data = load_npz_dataset(dataset, device, args.data_root)
    if data is None:
        raise FileNotFoundError(f"Missing dataset: {dataset}")

    final_model, final_state = load_model(final_path, device)
    training_config = final_state.get("training_config", {})
    final_epoch = int(final_state.get("epoch", 0))
    _, stage_fractions, boundaries = stage_spec(training_config, final_epoch)

    checkpoints = find_checkpoints(model_dir)
    selected = choose_checkpoints(checkpoints, boundaries, args.max_checkpoints_per_stage)
    # Prefer model.pt for the final epoch so the report does not duplicate the
    # last checkpoint and the deployed weight file is the one being diagnosed.
    selected = [p for p in selected if checkpoint_epoch(p) != final_epoch]
    selected.append(final_path)

    rows = []
    print(f"[{dataset}] evaluating {len(selected)} checkpoints from {model_dir}")
    for path in selected:
        model, state = load_model(path, device)
        rows.append(checkpoint_metrics(model, state, path, data, device, args, stage_fractions, boundaries))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    rows = sorted(rows, key=lambda r: r.epoch)

    history = safe_history(model_dir)
    temporal = final_temporal_metrics(final_model, data, args)
    temporal_summary = {
        "mean_rel_l2_t": float(np.mean(temporal["rel_l2_t"])),
        "max_rel_l2_t": float(np.max(temporal["rel_l2_t"])),
        "worst_time": float(data["t"][int(np.argmax(temporal["rel_l2_t"]))]),
        "mean_amp_ratio_t": float(np.mean(temporal["amp_ratio_t"])),
        "mean_energy_ratio_t": float(np.mean(temporal["energy_ratio_t"])),
    }
    coverage = source_coverage(training_config, dataset, seed=args.seed)

    figures = {
        "history": plot_history(dataset, args.model_tag, history, boundaries, out_dir),
        "checkpoint_diagnostics": plot_checkpoint_metrics(dataset, args.model_tag, rows, boundaries, out_dir),
        "temporal_error": plot_temporal(dataset, data, temporal, boundaries, stage_fractions, out_dir),
        "snapshots": None if args.no_snapshots else plot_snapshots(dataset, data, temporal, stage_fractions, out_dir),
    }

    findings, recommendations = diagnose(dataset, rows[-1], rows, history, training_config, coverage)

    csv_path = os.path.join(out_dir, f"{dataset}_checkpoint_metrics.csv")
    json_path = os.path.join(out_dir, f"{dataset}_summary.json")
    md_path = os.path.join(out_dir, f"{dataset}_diagnosis.md")
    save_rows_csv(csv_path, rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": dataset,
            "model_dir": model_dir,
            "training_config": training_config,
            "final_metrics": rows[-1].__dict__,
            "temporal_summary": temporal_summary,
            "source_coverage": coverage,
            "findings": findings,
            "recommendations": recommendations,
            "figures": figures,
        }, f, indent=2, ensure_ascii=False)
    write_dataset_report(md_path, dataset, args.model_tag, model_dir, training_config, rows, temporal_summary, coverage, findings, recommendations, figures)

    return {
        "dataset": dataset,
        "report": md_path,
        "csv": csv_path,
        "summary": json_path,
        "final_rel_l2": rows[-1].rel_l2,
        "final_amp_ratio": rows[-1].amp_ratio,
        "final_corr": rows[-1].corr,
    }


def write_global_report(out_dir, model_tag, summaries):
    path = os.path.join(out_dir, f"{model_tag}_global_diagnosis.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# FF-PINN Global Diagnosis: {model_tag}\n\n")
        f.write("| dataset | final_rel_l2 | final_amp_ratio | final_corr | report |\n")
        f.write("|---|---:|---:|---:|---|\n")
        for s in summaries:
            rel_report = os.path.relpath(s["report"], out_dir).replace("\\", "/")
            f.write(
                f"| {s['dataset']} | {s['final_rel_l2']:.6f} | {s['final_amp_ratio']:.6f} | "
                f"{s['final_corr']:.6f} | `{rel_report}` |\n"
            )
        problem_forms = set()
        for s in summaries:
            summary_path = s.get("summary")
            if not summary_path or not os.path.exists(summary_path):
                continue
            with open(summary_path, "r", encoding="utf-8") as sf:
                payload = json.load(sf)
            problem_forms.add((payload.get("training_config") or {}).get("problem_form", "unknown"))
        f.write("\n## Primary Interpretation\n\n")
        if problem_forms == {"source"}:
            f.write(
                "This run follows the paper-style source-term PDE `u=t^2 f_theta` with a Ricker source. "
                "If it is evaluated against the current local `.npz` datasets, remember those references "
                "were generated as initial-Gaussian displacement wavefields; physics-definition mismatch "
                "is therefore a first-order explanation for high Rel L2.\n"
            )
        elif problem_forms == {"initial"}:
            f.write(
                "This run uses the local initial-Gaussian formulation, so Rel L2 is directly comparable "
                "with the current `.npz` reference wavefields. Remaining errors should be interpreted as "
                "optimization, sampling, loss-balance, or model-capacity issues rather than source/initial "
                "condition mismatch.\n"
            )
        else:
            f.write(
                "The analyzed checkpoints contain mixed or missing `problem_form` metadata. Interpret each "
                "dataset-level report individually before comparing Rel L2 values.\n"
            )
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Plot and diagnose FF-PINN paper checkpoint training.")
    parser.add_argument("--dataset", choices=["all", *DATASETS], default="all")
    parser.add_argument("--model-tag", default="ff_pinn_paper")
    parser.add_argument("--model-root", default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-checkpoints-per-stage", type=int, default=4)
    parser.add_argument("--skip-pde", action="store_true", help="Skip expensive PDE residual diagnostics.")
    parser.add_argument("--pde-points", type=int, default=1024)
    parser.add_argument("--pde-batch-size", type=int, default=256)
    parser.add_argument("--no-snapshots", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    summaries = []
    for dataset in datasets:
        result = analyze_dataset(dataset, args, device)
        if result is not None:
            summaries.append(result)
            print(f"[DONE] {dataset}: {result['report']}")
    if summaries:
        global_report = write_global_report(os.path.abspath(args.out_dir), args.model_tag, summaries)
        print(f"[DONE] global report: {global_report}")


if __name__ == "__main__":
    main()
