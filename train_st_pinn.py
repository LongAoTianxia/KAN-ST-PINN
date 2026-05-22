import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import argparse
import random
import math
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch.optim.lr_scheduler as lr_scheduler

from PINNs_util.PINNs_fdiff import solver
from PINNs_util.PINNs_aux import (
    xyt_tensor, pde_residual, rand_colloc_fixed,
    rand_boundary, bc_residual_absorbing, update_lambda
)
from PINNs_util.st_pinn import STPinn
from PINNs_util.datasets import generate_marmousi, generate_overthrust


# 训练配置
class Config:
    L = 5.0           # km
    T = 2.0           # s  
    c0 = 3.0          # km/s  
    Nx = 60
    Ny = 60

    d_model = 128       
    seq_len = 8
    dt_hist = 0.02
    n_fourier = 32
    spatial_layers = 3
    lstm_layers = 2
    num_heads = 4
    n_attn_layers = 1
    dropout = 0.0

    n_epochs = 10000
    learning_rate = 5e-4
    n_ini = 20          # initial-condition snapshots (more = stronger anchoring)
    n_data = 3000       # supervised data points per epoch (data-driven)
    n_colloc = 4000     # collocation points per epoch (increased for stronger PDE)
    n_bc = 800          # boundary points per epoch (increased for longer T)
    ratio_gaussian = 0.85
    bc_type = 'absorbing'
    grad_clip = 1.0     # gradient clipping max norm
    mini_batch_ini = 4000  # mini-batch size for initial condition

    w_ini = 10.0         # strong IC anchoring
    w_pde = 0.05         # moderate PDE — real physics enforcement
    w_bc  = 0.05         # moderate BC — match PDE scale
    w_data = 10.0        # strong data supervision
    use_adaptive_lambda = False   # disabled: causes oscillation
    lambda_alpha = 0.9
    lambda_update_freq = 200

    use_causal = True
    causal_eps = 3.0     # reduced from 5.0 — less aggressive, better late-time learning

    pde_warmup_epochs = 2000   # Phase 1: learn IC/data only (no PDE)
    pde_rampup_epochs = 2000   # Phase 2: linearly ramp PDE from 0 → w_pde (longer ramp)

    warmup_epochs = 500
    lr_min_ratio = 0.01

    FIGURES_DIR = "./figures"
    MODEL_DIR   = "./trained"
    DATASET     = "threelayer"   # threelayer | marmousi | overthrust


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =======
# 数据集
# =======

def get_wave_speed_threelayer(xx, yy, device, tensor=False):
    """三层速度模型, sigmoid平滑过渡"""
    # sigmoid平滑层界面阶跃
    steepness = 40.0
    if tensor:
        sig = torch.sigmoid
        ones = torch.ones_like(xx, device=device)
    else:
        sig = lambda x: 1.0 / (1.0 + np.exp(-x))
        ones = np.ones_like(xx)
    # 基础速度0.5, y=0.33处过渡到0.75, y=0.66处到1.0
    c_val = 0.5 * ones + 0.25 * sig(steepness * (yy - 0.33)) + 0.25 * sig(steepness * (yy - 0.66))
    return c_val


def prepare_threelayer(config, device):
    """生成三层模型数据集"""
    L = config.L / config.L  # normalized
    T = config.T * config.c0 / config.L

    def c_func(xx, yy, tensor=False):
        return get_wave_speed_threelayer(xx, yy, device, tensor)

    gpulse_std = 5e-2
    r_pulse = np.array([0.5, 0.5])
    def I_func(xx, yy):
        return np.exp(-0.5 * (((xx - r_pulse[0]) / gpulse_std) ** 2 +
                               ((yy - r_pulse[1]) / gpulse_std) ** 2))

    print("Solving FD reference (three-layer)...")
    p_ref, xx, yy, t, dt = solver(I_func, c_func, L, L, config.Nx, config.Ny, -1, T)

    # 声压归一化到[-1,1]
    p_max = np.max(np.abs(p_ref))
    if p_max > 1e-12:
        p_ref = p_ref / p_max
        print(f"  p_ref normalized by p_max = {p_max:.6e}")

    xy = np.column_stack((xx.reshape(-1, 1), yy.reshape(-1, 1)))
    r_ref = xyt_tensor(xy, t, device)
    n_T = t.shape[0]
    n_L = xx.shape[0]

    # 物理尺度（归一化前）
    L_phys = config.L       # km
    T_phys = config.T       # s

    return {
        'L': L, 'T': T,
        'L_phys': L_phys, 'T_phys': T_phys,
        'p_ref': p_ref, 'xx': xx, 'yy': yy, 't': t,
        'xy': xy, 'r_ref': r_ref, 'n_T': n_T, 'n_L': n_L,
        'c_func': c_func, 'p_max': p_max,
    }


def prepare_dataset(config, device):
    """加载或生成数据集"""
    name = config.DATASET.lower()
    if name == 'threelayer':
        return prepare_threelayer(config, device)
    elif name == 'marmousi':
        return generate_marmousi(device, Nx=config.Nx, Ny=config.Ny)
    elif name == 'overthrust':
        return generate_overthrust(device, Nx=config.Nx, Ny=config.Ny)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def align_temporal_config(config, data):
    """对齐dt_hist和seq_len到求解器时间网格"""
    if data['n_T'] > 1:
        dt_data = float(data['t'][1] - data['t'][0])
        if dt_data > 0:
            config.dt_hist = dt_data

    t_span = float(data['t'][-1] - data['t'][0]) if data['n_T'] > 1 else 0.0
    max_seq = max(1, int(np.floor((t_span + 1e-12) / config.dt_hist)) + 1)
    if config.seq_len > max_seq:
        print(f"[temporal] seq_len {config.seq_len} → {max_seq}")
        config.seq_len = max_seq


def prepare_initial_condition(data, config, device):
    t_ini = data['t'][:config.n_ini]
    r_ini = xyt_tensor(data['xy'], t_ini, device)
    p_ini = data['p_ref'][:, :, :config.n_ini].reshape(-1, 1)
    p_ini = torch.tensor(p_ini, device=device)
    return r_ini, p_ini


# =======
# 模型构建（含消融变体）
# =======

ABLATION_VARIANTS = {
    'full':          dict(use_fourier=True,  use_lstm=True,  use_attention=True,  attn_type='cross'),
    'no_fourier':    dict(use_fourier=False, use_lstm=True,  use_attention=True,  attn_type='cross'),
    'no_lstm':       dict(use_fourier=True,  use_lstm=False, use_attention=True,  attn_type='cross'),
    'no_attention':  dict(use_fourier=True,  use_lstm=True,  use_attention=False, attn_type='cross'),
    'no_lstm_attn':  dict(use_fourier=True,  use_lstm=False, use_attention=False, attn_type='cross'),
    'self_attn':     dict(use_fourier=True,  use_lstm=True,  use_attention=True,  attn_type='self'),
}


def create_model(config, device, variant='full'):
    flags = ABLATION_VARIANTS[variant]
    model = STPinn(
        d_model=config.d_model,
        seq_len=config.seq_len,
        dt_hist=config.dt_hist,
        n_fourier=config.n_fourier,
        spatial_layers=config.spatial_layers,
        lstm_layers=config.lstm_layers,
        num_heads=config.num_heads,
        n_attn_layers=config.n_attn_layers,
        dropout=config.dropout,
        **flags,
    )
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{variant}] Created STPinn v2: {n_params:,} parameters")
    return model


# =======
# 训练
# =======

def sample_data_points(data, n_data, device):
    """从参考解中随机采样监督点"""
    n_total = data['r_ref'].shape[0]
    idx = torch.randint(0, n_total, (n_data,))
    r_data = data['r_ref'][idx]  # already on device, requires_grad_=True
    p_data = data['p_ref'].flatten()[idx.cpu().numpy()]
    p_data = torch.tensor(p_data, dtype=torch.float32, device=device).unsqueeze(1)
    return r_data, p_data


def train_model(model, data, r_ini, p_ini, config, device, save_dir=None):
    n_epochs = config.n_epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    torch.backends.cudnn.benchmark = True

    # Warmup + cosine decay
    warmup = config.warmup_epochs
    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, n_epochs - warmup)
        return config.lr_min_ratio + (1 - config.lr_min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    mse = nn.MSELoss()

    t_max = float(data['t'][-1])
    use_bc = (config.bc_type == 'absorbing')
    use_data = (config.n_data > 0)

    # PDE二阶自动微分不兼容混合精度

    # 自适应损失权重
    if config.use_adaptive_lambda:
        lamb = [torch.tensor(config.w_ini, dtype=torch.float32),
                torch.tensor(config.w_pde, dtype=torch.float32),
                torch.tensor(config.w_bc, dtype=torch.float32)]
        if use_data:
            lamb.append(torch.tensor(config.w_data, dtype=torch.float32))

    history = {'loss_ini': [], 'loss_pde': [], 'loss_bc': [],
               'loss_data': [], 'loss_total': []}

    # 初始条件mini-batch（省显存）
    mini_bs = config.mini_batch_ini
    n_ini_total = r_ini.shape[0]

    print(f"Training for {n_epochs} epochs (device={device})...")
    for epoch in tqdm(range(n_epochs), disable=not sys.stdout.isatty()):
        optimizer.zero_grad()

        # 初始条件损失
        if n_ini_total <= mini_bs:
            p_pred = model(r_ini)
            loss_ini = mse(p_pred, p_ini)
        else:
            idx = torch.randperm(n_ini_total, device=device)[:mini_bs]
            p_pred = model(r_ini[idx])
            loss_ini = mse(p_pred, p_ini[idx])

        # PDE残差（因果时间加权）
        r_colloc = rand_colloc_fixed(config.n_colloc, data['L'], t_max, device,
                                     ratio_gaussian=config.ratio_gaussian)
        c_colloc = data['c_func'](r_colloc[:, 0:1], r_colloc[:, 1:2], tensor=True)
        with torch.backends.cudnn.flags(enabled=False):
            p_colloc = model(r_colloc)
        pde_res = pde_residual(p_colloc, r_colloc, c_colloc)
        if getattr(config, 'use_causal', False) and t_max > 0:
            t_frac = r_colloc[:, 2:3].detach() / t_max
            causal_w = torch.exp(-config.causal_eps * t_frac)
            loss_pde = torch.mean(causal_w * pde_res ** 2)
        else:
            loss_pde = mse(pde_res, torch.zeros_like(p_colloc))

        # 边界条件
        if use_bc:
            r_bc, side = rand_boundary(config.n_bc, data['L'], t_max, device)
            c_bc = data['c_func'](r_bc[:, 0:1], r_bc[:, 1:2], tensor=True)
            with torch.backends.cudnn.flags(enabled=False):
                p_bc = model(r_bc)
            bc_res = bc_residual_absorbing(p_bc, r_bc, side, c_bc)
            loss_bc = mse(bc_res, torch.zeros_like(p_bc))
        else:
            loss_bc = torch.tensor(0.0, device=device)

        # 数据监督
        if use_data:
            r_data, p_data_target = sample_data_points(data, config.n_data, device)
            p_data_pred = model(r_data)
            loss_data = mse(p_data_pred, p_data_target)
        else:
            loss_data = torch.tensor(0.0, device=device)

        # 自适应损失加权
        if config.use_adaptive_lambda and epoch > 0 and epoch % config.lambda_update_freq == 0:
            loss_list = [loss_ini, loss_pde, loss_bc]
            if use_data:
                loss_list.append(loss_data)
            lamb = update_lambda(model, loss_list, lamb, config.lambda_alpha)

        if config.use_adaptive_lambda:
            w_ini = lamb[0]
            w_pde = lamb[1]
            w_bc  = lamb[2]
            w_data = lamb[3] if use_data and len(lamb) > 3 else config.w_data
        else:
            w_ini = config.w_ini
            w_pde = config.w_pde
            w_bc  = config.w_bc
            w_data = config.w_data

        # PDE课程学习: 预热 → 线性增长 → 全量
        pde_warmup = getattr(config, 'pde_warmup_epochs', 0)
        pde_rampup = getattr(config, 'pde_rampup_epochs', 0)
        if epoch < pde_warmup:
            # 阶段1: 只学IC+数据, 不加PDE/BC
            w_pde = 0.0
            w_bc = 0.0
        elif epoch < pde_warmup + pde_rampup and pde_rampup > 0:
            # 阶段2: PDE/BC线性增长
            ramp_frac = (epoch - pde_warmup) / pde_rampup
            w_pde = config.w_pde * ramp_frac
            w_bc = config.w_bc * ramp_frac

        loss = w_ini * loss_ini + w_pde * loss_pde + w_bc * loss_bc
        if use_data:
            loss = loss + w_data * loss_data
        loss.backward()

        # 梯度裁剪
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        optimizer.step()
        scheduler.step()

        # log
        history['loss_ini'].append(loss_ini.item())
        history['loss_pde'].append(loss_pde.item())
        history['loss_bc'].append(loss_bc.item())
        history['loss_data'].append(loss_data.item() if use_data else 0.0)
        history['loss_total'].append(loss.item())

        if epoch % 100 == 99:
            lr_now = optimizer.param_groups[0]['lr']
            w_info = f"  w=[{float(w_pde):.1f},{float(w_ini):.1f},{float(w_bc):.1f},{float(w_data):.1f}]"
            print(f"  [{epoch+1}] total={loss.item():.3e}  ini={loss_ini.item():.3e}"
                  f"  pde={loss_pde.item():.3e}  bc={loss_bc.item():.3e}"
                  f"  data={loss_data.item():.3e}  lr={lr_now:.2e}{w_info}")

        # 每100轮存checkpoint
        if save_dir and epoch % 100 == 99:
            ckpt_dir = save_dir
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f'checkpoint_epoch{epoch+1}.pt'))
            torch.save(history, os.path.join(ckpt_dir, 'history.pt'))

    return history


# =======
# 评估
# =======

def evaluate_model(model, data, device):
    """计算Rel L2和MAE"""
    batch_size = 4096
    p_est_list = []
    with torch.no_grad():
        for i in range(0, data['r_ref'].shape[0], batch_size):
            batch = data['r_ref'][i:i+batch_size]
            p_est_list.append(model(batch).cpu())
    p_est = torch.cat(p_est_list, dim=0).numpy().flatten()
    p_ref = data['p_ref'].flatten()

    rel_l2 = np.linalg.norm(p_est - p_ref) / (np.linalg.norm(p_ref) + 1e-12)
    mae = np.mean(np.abs(p_est - p_ref))
    return {'rel_l2': rel_l2, 'mae': mae, 'p_est': p_est}


# =======
# 可视化
# =======

def save_loss_curves(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for key in ['loss_ini', 'loss_pde', 'loss_bc', 'loss_data']:
        vals = history.get(key, [])
        if vals and any(v > 0 for v in vals):
            ax.plot(vals, label=key, alpha=0.8)
    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, 'loss_curves.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def save_comparison_plot(p_est, data, save_dir):
    """论文风格波场对比图: 参考解 / PINN / 误差"""
    os.makedirs(save_dir, exist_ok=True)
    n_L = data['n_L']
    n_T = data['n_T']
    p_ref = data['p_ref']
    p_est_3d = p_est.reshape(n_L, n_L, n_T)

    # 物理尺度（画图用）
    L_phys = data.get('L_phys', data['L'])
    T_phys = data.get('T_phys', float(data['t'][-1]))
    t_norm = data['t']             # normalized time array
    t_phys_arr = t_norm / (t_norm[-1] + 1e-12) * T_phys  # map to physical seconds

    p_max = np.max(np.abs(p_ref))
    time_indices = [0, n_T // 3, 2 * n_T // 3, n_T - 1]
    n_cols = len(time_indices)

    # GridSpec布局
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(4.2 * n_cols + 1.6, 11))
    # 3行图
    gs_main = GridSpec(3, 1, figure=fig, hspace=0.38, top=0.95, bottom=0.05,
                       left=0.07, right=0.88)

    row_labels = ['Reference', 'PINN', 'Error']
    row_data = []  # (images list, vmin, vmax, cb_label)

    e_global = np.max(np.abs(p_est_3d - p_ref))

    for row_idx in range(3):
        gs_row = gs_main[row_idx].subgridspec(1, n_cols + 1,
                                               width_ratios=[1]*n_cols + [0.05],
                                               wspace=0.25)
        for col, idx in enumerate(time_indices):
            ax = fig.add_subplot(gs_row[0, col])
            if row_idx == 0:
                img_data = p_ref[:, :, idx]
                vmin, vmax = -p_max, p_max
                cmap = 'seismic'
            elif row_idx == 1:
                img_data = p_est_3d[:, :, idx]
                vmin, vmax = -p_max, p_max
                cmap = 'seismic'
            else:
                img_data = p_est_3d[:, :, idx] - p_ref[:, :, idx]
                vmin, vmax = -e_global, e_global
                cmap = 'seismic'

            im = ax.imshow(img_data, extent=[0, L_phys, 0, L_phys],
                           origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
                           aspect='equal')

            # 标题
            if row_idx == 0:
                t_sec = t_phys_arr[idx]
                ax.set_title(f"t={t_sec:.2f} s", fontsize=11)
            if row_idx == 2:
                err = p_est_3d[:, :, idx] - p_ref[:, :, idx]
                rel_l2_t = np.linalg.norm(err) / (np.linalg.norm(p_ref[:, :, idx]) + 1e-12)
                ax.set_title(f"Rel L2={rel_l2_t:.4f}", fontsize=10)

            ax.set_xlabel('x (km)')
            if col == 0:
                ax.set_ylabel('y (km)')
            else:
                ax.set_yticklabels([])

        # 行标签
        first_ax = fig.axes[-n_cols]
        first_ax.text(-0.35, 0.5, row_labels[row_idx],
                      transform=first_ax.transAxes, fontsize=12,
                      va='center', ha='center', rotation=90, fontweight='bold')

        # colorbar
        cax = fig.add_subplot(gs_row[0, n_cols])
        cb = fig.colorbar(im, cax=cax)
        if row_idx < 2:
            cb.set_label('normalized pressure')
        else:
            cb.set_label('Error')

    path = os.path.join(save_dir, 'comparison.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# 主函数

def run_single(config, device, variant='full'):
    """单次训练+评估"""
    out_dir = os.path.join(config.MODEL_DIR, config.DATASET, variant)
    fig_dir = os.path.join(config.FIGURES_DIR, config.DATASET, variant)

    set_seed(0)
    data = prepare_dataset(config, device)
    align_temporal_config(config, data)

    r_ini, p_ini = prepare_initial_condition(data, config, device)
    model = create_model(config, device, variant=variant)

    history = train_model(model, data, r_ini, p_ini, config, device, save_dir=out_dir)

    # 保存模型
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, 'model.pt'))
    torch.save(history, os.path.join(out_dir, 'history.pt'))

    # 评估
    metrics = evaluate_model(model, data, device)
    print(f"\n[{variant}] Relative L2 = {metrics['rel_l2']:.6f}  MAE = {metrics['mae']:.6f}")

    # 画图
    save_loss_curves(history, fig_dir)
    save_comparison_plot(metrics['p_est'], data, fig_dir)

    return metrics


def run_ablation(config, device):
    """跑所有消融实验"""
    results = {}
    for variant in ABLATION_VARIANTS:
        print(f"\n{'='*60}")
        print(f"  ABLATION: {variant} on {config.DATASET}")
        print(f"{'='*60}")
        metrics = run_single(config, device, variant=variant)
        results[variant] = metrics

    # 汇总 table
    print(f"\n{'='*60}")
    print(f"  ABLATION SUMMARY — {config.DATASET}")
    print(f"{'='*60}")
    print(f"{'Variant':<20} {'Rel L2':>12} {'MAE':>12}")
    print("-" * 46)
    for v, m in results.items():
        print(f"{v:<20} {m['rel_l2']:>12.6f} {m['mae']:>12.6f}")

    # 保存汇总
    summary_dir = os.path.join(config.FIGURES_DIR, config.DATASET)
    os.makedirs(summary_dir, exist_ok=True)
    with open(os.path.join(summary_dir, 'ablation_summary.txt'), 'w') as f:
        f.write(f"{'Variant':<20} {'Rel L2':>12} {'MAE':>12}\n")
        f.write("-" * 46 + "\n")
        for v, m in results.items():
            f.write(f"{v:<20} {m['rel_l2']:>12.6f} {m['mae']:>12.6f}\n")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description='ST-PINN Training (v2)')
    parser.add_argument('--dataset', type=str, default='threelayer',
                       choices=['threelayer', 'marmousi', 'overthrust'],
                       help='Dataset to train on')
    parser.add_argument('--variant', type=str, default='full',
                       choices=list(ABLATION_VARIANTS.keys()),
                       help='Model variant (for single run)')
    parser.add_argument('--ablation', action='store_true',
                       help='Run all ablation experiments')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of epochs')
    parser.add_argument('--lr', type=float, default=None,
                       help='Override learning rate')
    parser.add_argument('--nx', type=int, default=None,
                       help='Override grid Nx')
    parser.add_argument('--T', type=float, default=None,
                       help='Override simulation time T (seconds)')
    parser.add_argument('--w-pde', type=float, default=None,
                       help='Override PDE loss weight')
    parser.add_argument('--w-bc', type=float, default=None,
                       help='Override BC loss weight')
    parser.add_argument('--n-colloc', type=int, default=None,
                       help='Override collocation points')
    parser.add_argument('--d-model', type=int, default=None,
                       help='Override model dimension')
    parser.add_argument('--pde-warmup', type=int, default=None,
                       help='Override PDE warmup epochs')
    parser.add_argument('--pde-rampup', type=int, default=None,
                       help='Override PDE rampup epochs')
    parser.add_argument('--quick', action='store_true',
                       help='Fast screening mode (~10-20 min): smaller grid/model and shorter curriculum')
    parser.add_argument('--no-adaptive', action='store_true',
                       help='Disable adaptive loss weighting')
    parser.add_argument('--no-data', action='store_true',
                       help='Disable data supervision')
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()
    config.DATASET = args.dataset

    if args.quick:
        # 快速筛选模式
        config.n_epochs = 600
        config.Nx = 40
        config.Ny = 40
        config.T = 1.3
        config.d_model = 96
        config.n_colloc = 1000
        config.n_bc = 300
        config.pde_warmup_epochs = 150
        config.pde_rampup_epochs = 150

    if args.epochs is not None:
        config.n_epochs = args.epochs
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.nx is not None:
        config.Nx = args.nx
        config.Ny = args.nx
    if args.T is not None:
        config.T = args.T
    if args.w_pde is not None:
        config.w_pde = args.w_pde
    if args.w_bc is not None:
        config.w_bc = args.w_bc
    if args.n_colloc is not None:
        config.n_colloc = args.n_colloc
    if args.d_model is not None:
        config.d_model = args.d_model
    if args.pde_warmup is not None:
        config.pde_warmup_epochs = args.pde_warmup
    if args.pde_rampup is not None:
        config.pde_rampup_epochs = args.pde_rampup
    if args.no_adaptive:
        config.use_adaptive_lambda = False
    if args.no_data:
        config.n_data = 0

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Dataset: {config.DATASET}")

    if args.ablation:
        run_ablation(config, device)
    else:
        run_single(config, device, variant=args.variant)


if __name__ == "__main__":
    main()
