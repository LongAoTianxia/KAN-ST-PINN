"""消融实验可视化: 损失曲线对比、波场图、Rel L2柱状图等"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker

from PINNs_util.PINNs_fdiff import solver
from PINNs_util.PINNs_aux import xyt_tensor
from PINNs_util.kan_pinn import KANPinn, KANSTPinn
from PINNs_util.datasets import generate_marmousi, generate_overthrust

# 画图样式
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})

# 配置（需要和训练一致）
D_MODEL = 128
N_FOURIER = 32
SPATIAL_KAN_LAYERS = 3
GRID_SIZE = 5
SPLINE_ORDER = 3
SEQ_LEN = 8
DT_HIST = 0.02
LSTM_LAYERS = 2
NUM_HEADS = 4
N_ATTN_LAYERS = 2
DROPOUT = 0.0
L_KM = 5.0; T_SEC = 2.0; C0 = 3.0; NX = 60; NY = 60

MODEL_DIR = "./trained"
FIG_DIR = "./figures/paper"
os.makedirs(FIG_DIR, exist_ok=True)

OLD_REL_L2 = 0.113  # old ST-PINN threelayer Rel L2 baseline


# --- Data preparation (same as eval_rel_l2.py) ---
def get_wave_speed_threelayer(xx, yy, device, tensor=False):
    steepness = 40.0
    if tensor:
        sig = torch.sigmoid; ones = torch.ones_like(xx, device=device)
    else:
        sig = lambda x: 1.0 / (1.0 + np.exp(-x)); ones = np.ones_like(xx)
    return 0.5 * ones + 0.25 * sig(steepness * (yy - 0.33)) + 0.25 * sig(steepness * (yy - 0.66))

def prepare_threelayer(device):
    L = L_KM / L_KM; T = T_SEC * C0 / L_KM
    def c_func(xx, yy, tensor=False):
        return get_wave_speed_threelayer(xx, yy, device, tensor)
    gpulse_std = 5e-2; r_pulse = np.array([0.5, 0.5])
    def I_func(xx, yy):
        return np.exp(-0.5*(((xx-r_pulse[0])/gpulse_std)**2+((yy-r_pulse[1])/gpulse_std)**2))
    print("  Solving FD reference (threelayer)...")
    p_ref, xx, yy, t, dt = solver(I_func, c_func, L, L, NX, NY, -1, T)
    p_max = np.max(np.abs(p_ref))
    if p_max > 1e-12: p_ref = p_ref / p_max
    xy = np.column_stack((xx.reshape(-1,1), yy.reshape(-1,1)))
    r_ref = xyt_tensor(xy, t, device)
    return {'p_ref':p_ref, 'r_ref':r_ref, 'n_T':t.shape[0], 'n_L':xx.shape[0],
            't':t, 'L':L, 'T':T, 'L_phys':L_KM, 'T_phys':T_SEC, 'xx':xx, 'yy':yy}

def prepare_dataset(name, device):
    if name == 'threelayer': return prepare_threelayer(device)
    elif name == 'marmousi': return generate_marmousi(device, Nx=NX, Ny=NY)
    elif name == 'overthrust': return generate_overthrust(device, Nx=NX, Ny=NY)

def align_temporal(data):
    dt = DT_HIST; sl = SEQ_LEN
    if data['n_T'] > 1:
        dt_data = float(data['t'][1] - data['t'][0])
        if dt_data > 0: dt = dt_data
    t_span = float(data['t'][-1] - data['t'][0]) if data['n_T'] > 1 else 0.0
    max_seq = max(1, int(np.floor((t_span + 1e-12) / dt)) + 1)
    if sl > max_seq: sl = max_seq
    return dt, sl

def create_model(pure_kan, device, dt_hist=DT_HIST, seq_len=SEQ_LEN):
    if pure_kan:
        return KANPinn(d_hidden=D_MODEL, n_layers=SPATIAL_KAN_LAYERS,
                       n_fourier=N_FOURIER, grid_size=GRID_SIZE, spline_order=SPLINE_ORDER).to(device)
    else:
        return KANSTPinn(d_model=D_MODEL, seq_len=seq_len, dt_hist=dt_hist,
                         n_fourier=N_FOURIER, spatial_kan_layers=SPATIAL_KAN_LAYERS,
                         grid_size=GRID_SIZE, spline_order=SPLINE_ORDER,
                         lstm_layers=LSTM_LAYERS, num_heads=NUM_HEADS,
                         n_attn_layers=N_ATTN_LAYERS, dropout=DROPOUT).to(device)

def load_model(model, model_dir):
    path = os.path.join(model_dir, 'model.pt')
    state = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'model_state' in state:
        model.load_state_dict(state['model_state'])
    else:
        model.load_state_dict(state)

def predict(model, data, device):
    p_est_list = []
    with torch.no_grad():
        for i in range(0, data['r_ref'].shape[0], 4096):
            p_est_list.append(model(data['r_ref'][i:i+4096]).cpu())
    return torch.cat(p_est_list, 0).numpy().flatten()

def compute_rel_l2(p_est, p_ref_flat):
    return np.linalg.norm(p_est - p_ref_flat) / (np.linalg.norm(p_ref_flat) + 1e-12)

def per_timestep_rel_l2(p_est, p_ref_3d, n_L, n_T):
    p_est_3d = p_est.reshape(n_L, n_L, n_T)
    errs = []
    for k in range(n_T):
        ref_k = p_ref_3d[:,:,k].flatten()
        est_k = p_est_3d[:,:,k].flatten()
        norm = np.linalg.norm(ref_k)
        errs.append(np.linalg.norm(est_k - ref_k) / (norm + 1e-12) if norm > 1e-12 else np.linalg.norm(est_k - ref_k))
    return np.array(errs)


# --- 图1: Loss curves comparison (threelayer) ---
def fig1_loss_curves():
    print("[图1] Loss curves comparison...")
    # 加载训练历史
    hist_old_path = os.path.join(MODEL_DIR, 'threelayer/full/history_6000.pt')
    if not os.path.exists(hist_old_path):
        hist_old_path = os.path.join(MODEL_DIR, 'threelayer/full/history.pt')
    h_old = torch.load(hist_old_path, weights_only=False)
    h_kan = torch.load(os.path.join(MODEL_DIR, 'threelayer/kan_st_pinn/history.pt'), weights_only=False)
    h_pure = torch.load(os.path.join(MODEL_DIR, 'threelayer/kan_pinn/history.pt'), weights_only=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    loss_keys = [('loss_ini', 'Initial Condition Loss'),
                 ('loss_data', 'Data Supervision Loss'),
                 ('loss_pde', 'PDE Residual Loss'),
                 ('loss_total', 'Total Loss')]

    colors = {'Old ST-PINN': '#2196F3', 'KAN-ST-PINN': '#E91E63', 'Pure-KAN': '#4CAF50'}

    for idx, (key, title) in enumerate(loss_keys):
        ax = axes[idx // 2, idx % 2]
        for label, h, c in [('Old ST-PINN', h_old, colors['Old ST-PINN']),
                             ('KAN-ST-PINN', h_kan, colors['KAN-ST-PINN']),
                             ('Pure-KAN', h_pure, colors['Pure-KAN'])]:
            vals = h.get(key, [])
            if vals and any(v > 0 for v in vals):
                # 滑动平均平滑
                vals_arr = np.array(vals, dtype=float)
                win = min(50, len(vals_arr) // 10)
                if win > 1:
                    smoothed = np.convolve(vals_arr, np.ones(win)/win, mode='valid')
                    x = np.arange(win-1, len(vals_arr))
                else:
                    smoothed = vals_arr
                    x = np.arange(len(vals_arr))
                ax.plot(x, smoothed, label=label, color=c, linewidth=1.5, alpha=0.85)

        # 标记PDE预热阶段
        ax.axvline(x=2000, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        if idx == 0:
            ax.text(2050, ax.get_ylim()[1]*0.7, 'PDE\nwarmup', fontsize=9, color='gray')

        ax.set_yscale('log')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Training Loss Curves — Three-Layer Model', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig1_loss_curves.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# --- 图2: Wavefield comparison (per dataset, KAN-ST-PINN) ---
def fig2_wavefield(dataset_name, data, model, device):
    print(f"[图2] Wavefield comparison ({dataset_name})...")
    model.eval()
    p_est = predict(model, data, device)

    n_L = data['n_L']; n_T = data['n_T']
    p_ref = data['p_ref']
    p_est_3d = p_est.reshape(n_L, n_L, n_T)

    L_phys = data.get('L_phys', data['L'])
    T_phys = data.get('T_phys', float(data['t'][-1]))
    t_phys = data['t'] / (data['t'][-1] + 1e-12) * T_phys

    p_max = np.max(np.abs(p_ref))
    time_indices = [0, n_T // 4, n_T // 2, 3 * n_T // 4, n_T - 1]
    n_cols = len(time_indices)

    fig = plt.figure(figsize=(4.0 * n_cols + 1.2, 11))
    gs_main = GridSpec(3, 1, figure=fig, hspace=0.35, top=0.93, bottom=0.04, left=0.07, right=0.88)
    row_labels = ['Reference', 'KAN-ST-PINN', 'Error']
    e_global = np.max(np.abs(p_est_3d - p_ref))

    for row in range(3):
        gs_row = gs_main[row].subgridspec(1, n_cols + 1, width_ratios=[1]*n_cols+[0.05], wspace=0.22)
        for col, tidx in enumerate(time_indices):
            ax = fig.add_subplot(gs_row[0, col])
            if row == 0:
                img = p_ref[:,:,tidx]; vmin, vmax = -p_max, p_max; cmap = 'seismic'
            elif row == 1:
                img = p_est_3d[:,:,tidx]; vmin, vmax = -p_max, p_max; cmap = 'seismic'
            else:
                img = p_est_3d[:,:,tidx] - p_ref[:,:,tidx]; vmin, vmax = -e_global, e_global; cmap = 'seismic'

            im = ax.imshow(img, extent=[0,L_phys,0,L_phys], origin='lower',
                          cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
            if row == 0:
                ax.set_title(f"t = {t_phys[tidx]:.2f} s", fontsize=11)
            if row == 2:
                err = p_est_3d[:,:,tidx] - p_ref[:,:,tidx]
                rl2 = np.linalg.norm(err) / (np.linalg.norm(p_ref[:,:,tidx]) + 1e-12)
                ax.set_title(f"Rel L2 = {rl2:.4f}", fontsize=10)
            ax.set_xlabel('x (km)')
            if col == 0:
                ax.set_ylabel('y (km)')
            else:
                ax.set_yticklabels([])

        first_ax = fig.axes[-n_cols]
        first_ax.text(-0.30, 0.5, row_labels[row], transform=first_ax.transAxes,
                      fontsize=13, va='center', ha='center', rotation=90, fontweight='bold')
        cax = fig.add_subplot(gs_row[0, n_cols])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label('Pressure' if row < 2 else 'Error')

    fig.suptitle(f'Wavefield Comparison — {dataset_name.capitalize()}', fontsize=16, fontweight='bold')
    path = os.path.join(FIG_DIR, f'fig2_wavefield_{dataset_name}.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")
    return p_est


# --- 图3: Ablation bar chart (Rel L2) ---
def fig3_ablation_bars(results):
    print("[图3] Ablation bar chart...")
    datasets = ['threelayer', 'marmousi', 'overthrust']
    models = ['KAN-ST-PINN', 'Pure-KAN']

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(datasets))
    width = 0.25
    colors_bar = ['#E91E63', '#4CAF50', '#2196F3']

    # KAN-ST-PINN模型
    vals_kan_st = [results.get(f'kan_st_pinn/{d}', {}).get('rel_l2', 0) for d in datasets]
    # 纯KAN模型
    vals_pure = [results.get(f'kan_pinn/{d}', {}).get('rel_l2', 0) for d in datasets]

    bars1 = ax.bar(x - width, vals_kan_st, width, label='KAN-ST-PINN', color=colors_bar[0], alpha=0.85, edgecolor='white')
    bars2 = ax.bar(x, vals_pure, width, label='Pure-KAN', color=colors_bar[1], alpha=0.85, edgecolor='white')

    # 旧ST-PINN基线（仅threelayer）
    ax.bar(x[0] + width, OLD_REL_L2, width, label='Old ST-PINN', color=colors_bar[2], alpha=0.85, edgecolor='white')
    # 其他数据集没有旧基线
    for i in [1, 2]:
        ax.bar(x[i] + width, 0, width, color='lightgray')

    # 柱子上写数值
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.003, f'{h:.3f}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    # 旧基线标签
    ax.text(x[0] + width, OLD_REL_L2 + 0.003, f'{OLD_REL_L2:.3f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold', color=colors_bar[2])

    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in datasets], fontsize=13)
    ax.set_ylabel('Relative L2 Error', fontsize=13)
    ax.set_title('Ablation Study: Model Architecture Comparison', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(max(vals_pure), OLD_REL_L2) * 1.25)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig3_ablation_rel_l2.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# --- 图4: Per-timestep Rel L2 ---
def fig4_per_timestep(per_t_results):
    print("[图4] Per-timestep Rel L2...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    datasets = ['threelayer', 'marmousi', 'overthrust']
    colors = {'KAN-ST-PINN': '#E91E63', 'Pure-KAN': '#4CAF50'}

    for i, ds in enumerate(datasets):
        ax = axes[i]
        for tag, label in [('kan_st_pinn', 'KAN-ST-PINN'), ('kan_pinn', 'Pure-KAN')]:
            key = f"{tag}/{ds}"
            if key in per_t_results:
                errs, t_phys = per_t_results[key]
                ax.plot(t_phys, errs, color=colors[label], linewidth=1.5, label=label, alpha=0.85)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Rel L2 per timestep')
        ax.set_title(ds.capitalize(), fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    plt.suptitle('Per-Timestep Relative L2 Error', fontsize=16, fontweight='bold', y=1.03)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig4_per_timestep_rel_l2.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# --- 图5: Ablation table (printed + saved as text) ---
def fig5_ablation_table(results):
    print("[图5] Ablation summary table...")

    lines = []
    lines.append("=" * 78)
    lines.append(f"{'Model':<20} {'Dataset':<14} {'Rel L2':>10} {'MAE':>12} {'vs Old ST-PINN':>16}")
    lines.append("-" * 78)

    for tag, label in [('kan_st_pinn', 'KAN-ST-PINN'), ('kan_pinn', 'Pure-KAN')]:
        for ds in ['threelayer', 'marmousi', 'overthrust']:
            key = f"{tag}/{ds}"
            if key in results:
                m = results[key]
                vs = f"{m['rel_l2']/OLD_REL_L2:.3f}x" if ds == 'threelayer' else '—'
                lines.append(f"{label:<20} {ds:<14} {m['rel_l2']:>10.4f} {m['mae']:>12.6e} {vs:>16}")
        lines.append("-" * 78)

    lines.append(f"{'Old ST-PINN':<20} {'threelayer':<14} {OLD_REL_L2:>10.4f} {'—':>12} {'1.000x':>16}")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Key findings:")
    
    kan_st = results.get('kan_st_pinn/threelayer', {}).get('rel_l2', 0)
    pure_kan = results.get('kan_pinn/threelayer', {}).get('rel_l2', 0)
    
    lines.append(f"  1. KAN-ST-PINN achieves {OLD_REL_L2/kan_st:.1f}x lower Rel L2 than Old ST-PINN ({kan_st:.4f} vs {OLD_REL_L2})")
    lines.append(f"  2. Pure-KAN without LSTM+Attention: Rel L2 = {pure_kan:.4f} ({pure_kan/OLD_REL_L2:.2f}x vs Old)")
    lines.append(f"  3. LSTM+Attention contribution: {(pure_kan-kan_st)/pure_kan*100:.1f}% Rel L2 reduction")
    lines.append(f"  4. KAN spatial encoder alone is insufficient — temporal modeling is critical")

    text = "\n".join(lines)
    print(text)

    path = os.path.join(FIG_DIR, 'ablation_summary.txt')
    with open(path, 'w') as f:
        f.write(text)
    print(f"\n  Saved: {path}")


# --- Main ---
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Output: {FIG_DIR}\n")

    # 图1：loss曲线
    fig1_loss_curves()

    # 跑所有模型的评估
    datasets = ['threelayer', 'marmousi', 'overthrust']
    experiments = [
        ('kan_st_pinn', False),
        ('kan_pinn', True),
    ]

    results = {}
    per_t_results = {}
    dataset_cache = {}

    for ds in datasets:
        print(f"\nPreparing dataset: {ds}")
        data = prepare_dataset(ds, device)
        dataset_cache[ds] = data

        for tag, pure_kan in experiments:
            model_dir = os.path.join(MODEL_DIR, ds, tag)
            if not os.path.isdir(model_dir):
                print(f"  [SKIP] {tag}/{ds}")
                continue

            if pure_kan:
                model = create_model(True, device)
            else:
                dt, sl = align_temporal(data)
                model = create_model(False, device, dt_hist=dt, seq_len=sl)
            load_model(model, model_dir)
            model.eval()

            p_est = predict(model, data, device)
            p_ref_flat = data['p_ref'].flatten()
            rel_l2 = compute_rel_l2(p_est, p_ref_flat)
            mae = np.mean(np.abs(p_est - p_ref_flat))
            per_t = per_timestep_rel_l2(p_est, data['p_ref'], data['n_L'], data['n_T'])

            key = f"{tag}/{ds}"
            results[key] = {'rel_l2': rel_l2, 'mae': mae}
            
            T_phys = data.get('T_phys', float(data['t'][-1]))
            t_phys = np.linspace(0, T_phys, data['n_T'])
            per_t_results[key] = (per_t, t_phys)

            label = 'KAN-ST-PINN' if not pure_kan else 'Pure-KAN'
            print(f"  {label}/{ds}: Rel L2 = {rel_l2:.6f}")

            # 只给KAN-ST画波场图
            if not pure_kan:
                # 暂存预测结果 for wavefield plot
                fig2_wavefield(ds, data, model, device)

    # ── Figure 3: Ablation bars ──
    fig3_ablation_bars(results)

    # ── Figure 4: Per-timestep ──
    fig4_per_timestep(per_t_results)

    # ── Figure 5: Ablation table ──
    fig5_ablation_table(results)

    print(f"\nAll figures saved to: {FIG_DIR}/")


if __name__ == '__main__':
    main()
