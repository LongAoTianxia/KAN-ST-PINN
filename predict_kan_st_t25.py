import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from train_kan_pinn import (
    Config,
    set_seed,
    prepare_dataset,
    align_temporal_config,
    create_model,
)


def load_model_weights(model, model_path, device):
    state = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and 'model_state' in state:
        model.load_state_dict(state['model_state'])
    else:
        model.load_state_dict(state)


def predict_full_field(model, r_ref, batch_size=4096):
    outputs = []
    with torch.no_grad():
        for i in range(0, r_ref.shape[0], batch_size):
            outputs.append(model(r_ref[i:i + batch_size]).cpu())
    return torch.cat(outputs, dim=0).numpy().flatten()


def compute_per_timestep_rel_l2(p_est_3d, p_ref_3d):
    n_t = p_ref_3d.shape[2]
    rel_l2 = []
    for k in range(n_t):
        ref_k = p_ref_3d[:, :, k].reshape(-1)
        est_k = p_est_3d[:, :, k].reshape(-1)
        denom = np.linalg.norm(ref_k)
        if denom > 1e-12:
            rel_l2.append(np.linalg.norm(est_k - ref_k) / denom)
        else:
            rel_l2.append(np.linalg.norm(est_k - ref_k))
    return np.asarray(rel_l2)


def build_snapshot_indices(n_t, n_snapshots):
    idx = np.linspace(0, n_t - 1, n_snapshots)
    idx = np.clip(np.rint(idx).astype(int), 0, n_t - 1)
    return list(dict.fromkeys(idx.tolist()))


def save_comparison_figure(p_est, data, save_path, title, n_snapshots=6):
    n_l = data['n_L']
    n_t = data['n_T']
    p_ref = data['p_ref']
    p_est_3d = p_est.reshape(n_l, n_l, n_t)

    l_phys = data.get('L_phys', data['L'])
    t_phys = data['t'] / (data['t'][-1] + 1e-12) * data.get('T_phys', float(data['t'][-1]))

    time_indices = build_snapshot_indices(n_t, n_snapshots)
    n_cols = len(time_indices)

    p_max = float(np.max(np.abs(p_ref)))
    e_global = float(np.max(np.abs(p_est_3d - p_ref)))

    fig = plt.figure(figsize=(4.0 * n_cols + 1.7, 10.2))
    gs_main = GridSpec(
        3, 1, figure=fig, hspace=0.42, top=0.93, bottom=0.06, left=0.06, right=0.90
    )
    row_labels = ['Reference', 'PINN', 'Error']

    for row_idx in range(3):
        gs_row = gs_main[row_idx].subgridspec(
            1, n_cols + 1, width_ratios=[1] * n_cols + [0.05], wspace=0.24
        )
        for col_idx, t_idx in enumerate(time_indices):
            ax = fig.add_subplot(gs_row[0, col_idx])
            if row_idx == 0:
                img = p_ref[:, :, t_idx]
                vmin, vmax = -p_max, p_max
                cmap = 'seismic'
            elif row_idx == 1:
                img = p_est_3d[:, :, t_idx]
                vmin, vmax = -p_max, p_max
                cmap = 'seismic'
            else:
                img = p_est_3d[:, :, t_idx] - p_ref[:, :, t_idx]
                vmin, vmax = -e_global, e_global
                cmap = 'seismic'

            im = ax.imshow(
                img,
                extent=[0, l_phys, 0, l_phys],
                origin='lower',
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect='equal',
            )
            if row_idx == 0:
                ax.set_title(f"t={t_phys[t_idx]:.2f}s", fontsize=11)
            elif row_idx == 2:
                err = p_est_3d[:, :, t_idx] - p_ref[:, :, t_idx]
                rel_l2_t = np.linalg.norm(err) / (np.linalg.norm(p_ref[:, :, t_idx]) + 1e-12)
                ax.set_title(f"Rel L2={rel_l2_t:.4f}", fontsize=10)

            ax.set_xlabel('x (km)')
            if col_idx == 0:
                ax.set_ylabel('y (km)')
            else:
                ax.set_yticklabels([])

        first_ax = fig.axes[-n_cols]
        first_ax.text(
            -0.33,
            0.5,
            row_labels[row_idx],
            transform=first_ax.transAxes,
            fontsize=12,
            va='center',
            ha='center',
            rotation=90,
            fontweight='bold',
        )

        cax = fig.add_subplot(gs_row[0, n_cols])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label('normalized pressure' if row_idx < 2 else 'Error')

    fig.suptitle(title, fontsize=15)
    plt.savefig(save_path, dpi=220, bbox_inches='tight')
    plt.close()


def save_error_curve(rel_l2, t_phys, train_t_sec, save_path):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(t_phys, rel_l2, color='blue', linewidth=1.8, label='Rel L2 per timestep')
    ax.axvline(train_t_sec, color='red', linestyle='--', linewidth=1.8, label=f'Train T={train_t_sec:.1f}s')

    if t_phys[-1] > train_t_sec:
        ax.axvspan(train_t_sec, t_phys[-1], color='red', alpha=0.10, label='Extrapolation zone')

    ax.set_title('Per-Timestep Error: Training vs Extrapolation', fontsize=16)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Relative L2 Error', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xlim(float(t_phys[0]), float(t_phys[-1]))
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Use a threelayer KAN-ST-PINN model trained at T=2s to evaluate and visualize T=2.5s.'
    )
    parser.add_argument('--model-path', type=str, default='./trained/threelayer/kan_st_pinn/model.pt')
    parser.add_argument('--eval-T', type=float, default=2.5, help='Evaluation horizon in seconds.')
    parser.add_argument('--train-T', type=float, default=2.0, help='Training horizon in seconds for the red boundary line.')
    parser.add_argument('--dataset', type=str, default='threelayer', choices=['threelayer'])
    parser.add_argument('--out-dir', type=str, default='./figures/threelayer/full_T2s')
    parser.add_argument('--comparison-name', type=str, default='comparison_T2p5s.png')
    parser.add_argument('--error-name', type=str, default='error_vs_time_T2p5s.png')
    parser.add_argument('--metrics-name', type=str, default='metrics_T2p5s.txt')
    parser.add_argument('--snapshots', type=int, default=6)
    args = parser.parse_args()

    set_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = Config()
    config.DATASET = args.dataset
    config.T = args.eval_T

    print(f'Device: {device}')
    print(f'Loading model from: {args.model_path}')
    print(f'Generating {args.dataset} reference up to T={args.eval_T:.2f}s')

    data = prepare_dataset(config, device)
    align_temporal_config(config, data)

    model = create_model(config, device)
    load_model_weights(model, args.model_path, device)
    model.eval()

    p_est = predict_full_field(model, data['r_ref'])
    n_l = data['n_L']
    n_t = data['n_T']
    p_est_3d = p_est.reshape(n_l, n_l, n_t)
    rel_l2 = compute_per_timestep_rel_l2(p_est_3d, data['p_ref'])
    global_rel_l2 = float(np.linalg.norm(p_est - data['p_ref'].reshape(-1)) /
                          (np.linalg.norm(data['p_ref'].reshape(-1)) + 1e-12))
    mae = float(np.mean(np.abs(p_est - data['p_ref'].reshape(-1))))
    t_phys = data['t'] / (data['t'][-1] + 1e-12) * data['T_phys']

    os.makedirs(args.out_dir, exist_ok=True)
    comparison_path = os.path.join(args.out_dir, args.comparison_name)
    error_path = os.path.join(args.out_dir, args.error_name)
    metrics_path = os.path.join(args.out_dir, args.metrics_name)

    save_comparison_figure(
        p_est,
        data,
        comparison_path,
        title=f'T={args.eval_T:.1f}s Validation',
        n_snapshots=args.snapshots,
    )
    save_error_curve(rel_l2, t_phys, args.train_T, error_path)

    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write(f'model_path={args.model_path}\n')
        f.write(f'dataset={args.dataset}\n')
        f.write(f'train_T={args.train_T:.6f}\n')
        f.write(f'eval_T={args.eval_T:.6f}\n')
        f.write(f'global_rel_l2={global_rel_l2:.8f}\n')
        f.write(f'mae={mae:.8e}\n')
        f.write(f'max_per_timestep_rel_l2={float(np.max(rel_l2)):.8f}\n')
        f.write(f'mean_per_timestep_rel_l2={float(np.mean(rel_l2)):.8f}\n')

    print(f'Global Rel L2: {global_rel_l2:.6f}')
    print(f'MAE: {mae:.6e}')
    print(f'Saved: {comparison_path}')
    print(f'Saved: {error_path}')
    print(f'Saved: {metrics_path}')


if __name__ == '__main__':
    main()
