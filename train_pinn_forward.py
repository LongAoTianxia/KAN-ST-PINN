"""PINN正问题求解二维声波方程"""

import os
# 解决OpenMP重复加载问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import random
import numpy as np
import math
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch.optim.lr_scheduler as lr_scheduler

import optuna
_HAS_OPTUNA = True

from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
_HAS_SKOPT = True

from PINNs_util.PINNs_fdiff import solver
from PINNs_util.PINNs_aux import (
    xyt_tensor, pde_residual, update_lambda, 
    rand_colloc, rand_colloc_mixed,rand_colloc_fixed, 
    rand_boundary, bc_residual_absorbing
)
from PINNs_util.PINNs_aux import FCN, FCN_with_Attention, FCN_Attention, LSTMAttention, SpaceTimeAttentionLSTM, LongHistoryTemporalEncoderPINN

# 训练配置
class Config:
    # 路径
    FIGURES_DIR = "./figures"
    MODEL_DIR = "./trained/forward/"
    
    # 计算域参数
    L = 5           # Domain size (km)
    T = 1.3         # Time duration (s)
    c0 = 3          # Max wave speed (km/s)
    
    # 网格
    Nx = 50
    Ny = 50
    
    # 网络结构
    n_in = 3
    n_out = 1
    n_hidden = 32  # n_hidden % n_tokens == 0 且 (n_hidden / n_tokens) % num_heads == 0
    n_layers = 1

    # ------------------- SpaceTimeAttentionLSTM -------------------------------
    d_tok = 64          # token 嵌入维度  d_tok % num_heads == 0  fixed 不超过128
    n_s = 3             # 空间 token 数量
    n_t = 12             # 时间 token 数量 FIXED
    n_ffeatures = 64
    lstm_hidden = 32    # BiLSTM 隐藏维度   fixed
    lstm_layers = 1     # LSTM 层数  fixed
    num_heads = 4       # 注意力头数 fixed
    n_blocks = 1        # ST Block 数量fixed
    mlp_ratio = 4.0     # FFN 扩展比例fixed
    dt_token = 0.02     # 时间 token 间隔（默认值；可由数据时间网格动态覆盖）
    # --------------------------------------------------------------------------

    n_tokens = 4
    # num_heads = 4  # 暂未用
    fusion_mode = 'self_attn'  # 'self_attn' or 'cross_attn'
    seq_mode = 'spatial_temporal' # 'dim' or 'spatial_temporal'
    use_attention = True  

    # ---- LongHistoryTemporalEncoderPINN 专用 -------------------------
    seq_len              = 24      # 时间历史序列长度 L
    dt_hist              = 0.02   # 每步时间间隔

    temporal_embed_dim   = 32     # 时间 embedding 维度 d_t
    hidden_dim           = 48     # LSTM 隐藏维度 d_h
    spatial_hidden       = 48     # 空间 MLP 中间层维度
    spatial_layers       = 3      # 空间 MLP 隐藏层数（不含末尾输出层）

    use_fourier_spatial  = True   # 空间分支是否使用 Fourier feature
    use_fourier_temporal = True  # 时间序列是否使用 Fourier feature
    n_ffeatures_spatial   = 64
    n_ffeatures_temporal  = 64
    # ------------------------------------------------------------------

    # 训练参数
    n_ini = 10              # Number of initial condition snapshots
    n_lamb_update = 1e6     # Lambda update frequency
    n_colloc = int(6000)     # Number of collocation points
    n_causal = int(800)     # Causal training steps (set to int(2e3) for full training)
    learning_rate = 1e-4
    
    ratio_gaussian = 0.85    # for rand_colloc_fixed()

    # 边界条件 parameters
    n_bc = int(3000)         # Number of boundary condition points per epoch
    bc_type = 'absorbing'   # Boundary condition type: 'absorbing', 'none' (可扩展其他类型)
    
    # 梯度累积（省显存）
    #colloc_chunk = 20000      # 未使用
    #ini_chunk = 20000
    #bc_chunk = 10000
    
    # 阶段调度
    STAGE_A_END = 0.15
    STAGE_B_END = 0.70
 
    # 贝叶斯优化设置
    USE_STAGE_WEIGHTS = False
    RUN_BO = False              # if False: use DEFAULT_HP directly
    BO_N_TRIALS = 20
    #BO_PROXY_EPOCHS = 8000        # short-run budget per trial

    # 默认超参
    DEFAULT_HP = {
        "A_end": 0.15,
        "B_end": 0.7,
        "w_pde_A": 7.410036609388929e-05,
        "w_pde_B": 0.0001922532104554391,
        "w_pde_C": 0.00012274561300077908,
        "w_ini_B": 1.8935691608460081,
        "w_ini_C": 2.9646993302854057,
        "w_bc_C_max": 0.07941441643271903,
        "bc_ramp_start": 0.7283029715640411,
        "bc_ramp_len": 0.2981397976541317,
    }
    
    # 开关
    TRAIN_NEW_MODEL = True  # Set to True to train a new model

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =======
# 物理函数
def get_wave_speed(xx, yy, device, tensor=False):
    """三层速度模型 c(x,y)"""
    if tensor:
        c_val = 0.5 * torch.ones_like(xx, device=device)
    else:
        c_val = 0.5 * np.ones_like(xx)
    
    ind = yy >= 0.33
    c_val[ind] = 0.75
    ind = yy >= 0.66
    c_val[ind] = 1
    return c_val


def get_initial_condition(xx, yy):
    """初始条件: 中心(0.5,0.5)的高斯脉冲"""
    gpulse_std = 5e-2
    r_pulse = np.array([0.5, 0.5])
    I_val = np.exp(-0.5 * (((xx - r_pulse[0]) / gpulse_std)**2 +
                           ((yy - r_pulse[1]) / gpulse_std)**2))
    return I_val


# =======
# 绘图函数
# =======
def save_wave_speed(c_field, L, save_path):
    """保存波速图"""
    c_max = np.max(np.abs(c_field))
    c_min = np.min(np.abs(c_field))
    fig, ax = plt.subplots(figsize=(4, 3))
    img = ax.imshow(c_field, vmin=c_min, vmax=c_max, origin='lower', 
                    cmap='viridis', extent=[0, L, 0, L])
    fig.colorbar(img, ax=ax)
    ax.set_title('wave speed c(x,y)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def save_field_animation(p, L, title, save_path):
    """保存波场动画GIF"""
    p_max = np.max(np.abs(p))
    fig, [ax1, ax2] = plt.subplots(1, 2, gridspec_kw={"width_ratios": [50, 1]}, figsize=(4, 3))
    cmap = matplotlib.cm.seismic
    norm = matplotlib.colors.Normalize(vmin=-p_max, vmax=p_max)
    matplotlib.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, orientation='vertical')
    ax1.set_title(title)
    
    frames = []
    for i in range(p.shape[-1]):
        p_plot = p[:, :, i]
        img = ax1.imshow(p_plot, vmin=-p_max, vmax=p_max, origin='lower', 
                         cmap='seismic', extent=[0, L, 0, L], animated=True)
        frames.append([img])
    
    ani = animation.ArtistAnimation(fig, frames, interval=200, blit=True)
    ani.save(save_path, writer='pillow', fps=5)
    plt.close()
    print(f"Saved: {save_path}")


def save_estimation_animation(p_ref, p_est, L, save_path):
    """保存参考解vs预测对比动画"""
    p_max = np.max(np.abs(p_ref))
    n_L = p_ref.shape[0]
    n_T = p_ref.shape[-1]
    p_est = p_est.reshape(n_L, n_L, n_T)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
    ax1.set_title('Reference')
    ax2.set_title('Estimated')
    
    frames = []
    for i in range(n_T):
        img1 = ax1.imshow(p_ref[:, :, i], vmin=-p_max, vmax=p_max, origin='lower', 
                          cmap='seismic', extent=[-L/2, L/2, -L/2, L/2], animated=True)
        img2 = ax2.imshow(p_est[:, :, i], vmin=-p_max, vmax=p_max, origin='lower', 
                          cmap='seismic', extent=[-L/2, L/2, -L/2, L/2], animated=True)
        frames.append([img1, img2])
    
    ani = animation.ArtistAnimation(fig, frames, interval=200, blit=True)
    ani.save(save_path, writer='pillow', fps=5)
    plt.close()
    print(f"Saved: {save_path}")


def save_train_log(loss, lamb, label, save_path):
    """保存loss和lambda训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    ax1.set_title('Loss')
    for i in range(len(loss) - 1, -1, -1):
        ax1.plot(np.asarray(loss[i]) * np.asarray(lamb[i]) / np.asarray(lamb[0]), label=label[i])
    ax1.legend()
    ax1.set_yscale("log")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Weighted Loss")
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title('Lambda')
    for i in range(len(lamb) - 1, -1, -1):
        ax2.plot(lamb[i], label=label[i])
    ax2.legend()
    ax2.set_yscale("log")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Lambda")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def save_comparison(p_ref, p_est, L, T, t, save_path):
    """保存不同时刻的对比图"""
    n_L = p_ref.shape[0]
    n_T = p_ref.shape[-1]
    p_est = p_est.reshape(n_L, n_L, n_T)
    p_ref_max = np.max(p_ref)
    p_ref_min = -p_ref_max

    # 计算误差
    p_error = p_est - p_ref
    error_max = np.max(np.abs(p_error))

    time_indices = [0, int(n_T * 0.33), int(n_T * 0.66), n_T - 1]
    time_labels = [t[idx] * 5 / 3 for idx in time_indices]
    
    # 计算总时间场的相对L2范数
    total_l2_error = np.linalg.norm(p_est - p_ref) / (np.linalg.norm(p_ref) + 1e-12)
    print(f"Total relative L2 error over all time steps: {total_l2_error:.6f}")

    # 修改为 3 行：Reference, PINN, Error
    fig, axes = plt.subplots(3, 6, figsize=(22, 15), 
                             gridspec_kw={'width_ratios': [0.2, 1, 1, 1, 1, 0.2]})

    axes[0, 0].text(0.5, 0.5, 'Reference', fontsize=24, ha='center', va='center', rotation=0)
    axes[1, 0].text(0.5, 0.5, 'PINN', fontsize=24, ha='center', va='center', rotation=0)
    axes[2, 0].text(0.5, 0.5, 'Error', fontsize=24, ha='center', va='center', rotation=0)

    for ax in axes[:, 0]:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    for ax in axes[:, -1]:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # 第一行：Reference
    for i, idx in enumerate(time_indices):
        ax = axes[0, i + 1]
        im_ref = ax.imshow(p_ref[:, :, idx], extent=[0, 5, 0, 5], origin='lower', 
                       cmap='seismic', vmin=p_ref_min, vmax=p_ref_max)
        ax.set_title(f"t={time_labels[i]:.2f} s", fontsize=22)
        ax.set_xlabel('x (km)', fontsize=20)
        ax.set_ylabel('y (km)', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)

    # 第二行：PINN
    for i, idx in enumerate(time_indices):
        ax = axes[1, i + 1]
        im_est = ax.imshow(p_est[:, :, idx], extent=[0, 5, 0, 5], origin='lower', 
                       cmap='seismic', vmin=p_ref_min, vmax=p_ref_max)
        # ax.set_xlabel('x (km)', fontsize=20)
        ax.set_ylabel('y (km)', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)

    # 第三行：Error，并在 title 上标注相对L2范数
    for i, idx in enumerate(time_indices):
        ax = axes[2, i + 1]
        error_slice = p_error[:, :, idx]
        # 计算该时间步的相对L2范数
        l2_error = np.linalg.norm(error_slice) / (np.linalg.norm(p_ref[:, :, idx]) + 1e-12)
        im_err = ax.imshow(error_slice, extent=[0, 5, 0, 5], origin='lower', 
                       cmap='seismic', vmin=-error_max, vmax=error_max)
        ax.set_title(f"Rel L2={l2_error:.4f}", fontsize=22)
        ax.set_xlabel('x (km)', fontsize=20)
        ax.set_ylabel('y (km)', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)

    # 添加两个 colorbar：一个用于 Reference/PINN，一个用于 Error
    cbar_ax1 = fig.add_axes([0.94, 0.4, 0.02, 0.5])
    cbar1 = fig.colorbar(im_est, cax=cbar_ax1)
    cbar1.set_label('Normalized pressure', fontsize=24)
    cbar1.ax.tick_params(labelsize=20)

    cbar_ax2 = fig.add_axes([0.94, 0.08, 0.02, 0.25])
    cbar2 = fig.colorbar(im_err, cax=cbar_ax2)
    cbar2.set_label('Error', fontsize=24)
    cbar2.ax.tick_params(labelsize=20)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# =======
# 数据准备
# =======
def prepare_domain(config, device):
    """准备计算域和参考解"""
    # 归一化
    L = config.L / config.L  # Normalized to 1
    T = config.T * config.c0 / config.L
    
    # 波速函数闭包
    def c_func(xx, yy, tensor=False):
        return get_wave_speed(xx, yy, device, tensor)
    
    # 有限差分参考解
    print("正在计算有限差分参考解...")
    p_ref, xx, yy, t, dt = solver(
        get_initial_condition, c_func, L, L, config.Nx, config.Ny, -1, T
    )
    
    # 坐标张量
    xy = np.column_stack((np.reshape(xx, (-1, 1)), np.reshape(yy, (-1, 1))))
    r_ref = xyt_tensor(xy, t, device)
    # print("r_ref shape:", r_ref.shape)
    n_T = t.shape[0]
    n_L = xx.shape[0]
    
    return {
        'L': L, 'T': T,
        'p_ref': p_ref, 'xx': xx, 'yy': yy, 't': t,
        'xy': xy, 'r_ref': r_ref, 'n_T': n_T, 'n_L': n_L,
        'c_func': c_func
    }

def configure_temporal_history_from_data(config, data):
    """把dt_hist和seq_len对齐到实际的求解器时间网格"""
    # >>> FIXED_TEMPORAL_GRID_DISABLED_BEGIN (kept for possible restore)
    # fixed_dt_hist = config.dt_hist
    # fixed_seq_len = config.seq_len
    # <<< FIXED_TEMPORAL_GRID_DISABLED_END

    # 1) 从求解器时间网格获取dt_hist
    if data['n_T'] > 1:
        dt_hist_data = float(data['t'][1] - data['t'][0])
        if dt_hist_data > 0:
            config.dt_hist = dt_hist_data
        else:
            print(f"[temporal] Non-positive dt from data ({dt_hist_data}), keep config.dt_hist={config.dt_hist}")
    else:
        print("[temporal] n_T <= 1, keep config.dt_hist unchanged.")

    # 2) 约束seq_len
    t_span = float(data['t'][-1] - data['t'][0]) if data['n_T'] > 1 else 0.0
    max_seq_len = max(1, int(np.floor((t_span + 1e-12) / config.dt_hist)) + 1)
    if config.seq_len > max_seq_len:
        print(f"[temporal] Adjust seq_len: {config.seq_len} -> {max_seq_len} to satisfy (seq_len-1)*dt_hist <= T_span")
        config.seq_len = max_seq_len

    # 3) 对齐SpaceTimeAttentionLSTM窗口参数
    # >>> FIXED_ST_WINDOW_DISABLED_BEGIN (kept for possible restore)
    # fixed_dt_token = config.dt_token
    # fixed_n_t = config.n_t
    # <<< FIXED_ST_WINDOW_DISABLED_END
    if hasattr(config, "dt_token"):
        config.dt_token = config.dt_hist
    if hasattr(config, "n_t"):
        max_n_t = max_seq_len
        if config.n_t > max_n_t:
            print(f"[temporal] Adjust n_t: {config.n_t} -> {max_n_t} to satisfy (n_t-1)*dt_token <= T_span")
            config.n_t = max_n_t
        config.n_t = max(1, int(config.n_t))

    dt_token_val = getattr(config, "dt_token", config.dt_hist)
    n_t_val = getattr(config, "n_t", "N/A")
    print(f"[temporal] Using dt_hist={config.dt_hist:.6f}, seq_len={config.seq_len}, dt_token={dt_token_val:.6f}, n_t={n_t_val}, T_span={t_span:.6f}")


def prepare_initial_condition(data, config, device):
    """准备初始条件训练数据"""
    t_ini = data['t'][0:config.n_ini]
    r_ini = xyt_tensor(data['xy'], t_ini, device)
    p_ini = data['p_ref'][:, :, :config.n_ini].reshape(-1, 1)
    p_ini = torch.tensor(p_ini, device=device)
    
    return r_ini, p_ini


# =======
# 模型训练
# =======
def get_stage_weights(progress, hp):
    # progress ∈ [0,1]
    if progress < hp["A_end"]:
        return (
            1.0,                     # w_ini
            hp["w_pde_A"],            # w_pde
            0.0                       # w_bc
        )
    elif progress < hp["B_end"]:
        return (
            hp["w_ini_B"],
            hp["w_pde_B"],
            0.0
        )
    else:
        r = (progress - hp["bc_ramp_start"]) / max(hp["bc_ramp_len"], 1e-6)
        r = min(max(r, 0.0), 1.0)
        return (
            hp["w_ini_C"],
            hp["w_pde_C"],
            hp["w_bc_C_max"] * r
        )

def _random_hp(rng: np.random.RandomState, config: Config):
    # log-uniform采样
    def logu(a, b):
        return float(np.exp(rng.uniform(np.log(a), np.log(b))))
    hp = {
        "A_end": config.STAGE_A_END,
        "B_end": config.STAGE_B_END,
        "w_pde_A": logu(1e-3, 1e-1),
        "w_pde_B": logu(1e-2, 1.0),
        "w_pde_C": logu(1e-2, 1.0),
        "w_ini_B": float(rng.uniform(0.5, 2.0)),
        "w_ini_C": float(rng.uniform(0.2, 1.5)),
        "w_bc_C_max": logu(1e-4, 1e-1),
        "bc_ramp_start": float(rng.uniform(0.7, 0.9)),
        "bc_ramp_len": float(rng.uniform(0.05, 0.25)),
    }
    return hp


def create_model(config, device):
    """创建PINN模型"""
    #model = FCN(config.n_in, config.n_out, config.n_ffeatures, config.n_hidden, config.n_layers)
    #model = FCN_Attention(n_out=config.n_out, n_ffeatures=32, n_hidden=32, n_layers=2, num_heads=2, embed_dim=64, fusion_mode='cross_attn')
    #model = FCN_with_Attention(
    #    # n_hidden % n_tokens == 0 且 (n_hidden / n_tokens) % num_heads == 0
    #    n_in=config.n_in, n_out=config.n_out, n_ffeatures=config.n_ffeatures, n_hidden=config.n_hidden,   
    #    n_layers=config.n_layers, num_heads=config.num_heads, n_tokens=config.n_tokens,       # 拆分为4个token，每个16维
    #    mlp_ratio=4., use_attention=config.use_attention
    #)
    #model = LSTMAttention(input_dim=config.n_in, output_dim=config.n_out, hidden_dim=config.n_hidden, num_layers=config.n_layers, dropout=0.0, 
    #                      use_attention=config.use_attention, seq_mode=config.seq_mode, n_ffeatures=config.n_ffeatures)
    # model = SpaceTimeAttentionLSTM(d_tok = config.d_tok, n_s = config.n_s, n_t = config.n_t, n_ffeatures = config.n_ffeatures,
    #                             lstm_hidden = config.lstm_hidden, lstm_layers = config.lstm_layers, num_heads = config.num_heads,
    #                             n_blocks = config.n_blocks, mlp_ratio = config.mlp_ratio, dt_token = config.dt_token
    # )
    model = LongHistoryTemporalEncoderPINN(
        seq_len              = config.seq_len,
        dt_hist              = config.dt_hist,
        temporal_embed_dim   = config.temporal_embed_dim,
        hidden_dim           = config.hidden_dim,
        spatial_hidden       = config.spatial_hidden,
        spatial_layers       = config.spatial_layers,
        n_ffeatures_spatial  = config.n_ffeatures_spatial if config.use_fourier_spatial else 0,
        use_fourier_temporal = config.use_fourier_temporal,
        n_ffeatures_temporal = config.n_ffeatures_temporal if config.use_fourier_temporal else 0,
    )
    model = model.to(device)
    return model


def train_model(model, data, r_ini, p_ini, config, device, use_adaptive_lambda=False, use_stage_weights=False, hp=None, max_epochs=None):
    """训练PINN模型"""
    #n_epochs = int(config.n_causal * (data['n_T'] - 1))
    # n_causal 保留作为epoch预算
    n_epochs_full = int(config.n_causal * (data['n_T'] - 1))
    n_epochs = int(max_epochs) if max_epochs is not None else n_epochs_full

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    lf = lambda x: ((1 + math.cos(x * math.pi / n_epochs)) / 2) * (1 - 0.1) + 0.1  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    
    mse_loss = nn.MSELoss()
    # 使用 reduction='sum' 确保分块累积与整批计算等价
    # mse_loss_sum = nn.MSELoss(reduction='sum')
    
    # 历史记录
    loss_ini_hist, loss_pde_hist, loss_bc_hist = [], [], []
    lamb_ini_hist, lamb_pde_hist, lamb_bc_hist = [], [], []
    #lamb = [1, 1, 1]  # [lamb_ini, lamb_pde, lamb_bc]
    lamb = [5e4, 4, 1]
    #lamb = [3.5e4, 4, 3]

    # 是否启用BC
    use_bc = (config.bc_type == 'absorbing')
    
    # 提前保存标志
    early_saved = False
    loss_threshold = 0.03e-3

    if hp is None:
        hp = config.DEFAULT_HP
    print(f'Training PINN for {n_epochs} epochs... (use_adaptive_lambda={use_adaptive_lambda}, stage_weights={use_stage_weights})')
    
    # >>> CAUSAL_TRAINING_DISABLED_BEGIN (kept for possible restore)
    # i_causal = 0
    # <<< CAUSAL_TRAINING_DISABLED_END
    t_train_max = float(data['t'][-1])

    for i in tqdm(range(n_epochs),disable=not sys.stdout.isatty()):
        
        optimizer.zero_grad()
        # 初始条件损失
        p = model(r_ini)
        loss_ini = mse_loss(p, p_ini)

        # >>> CAUSAL_TRAINING_DISABLED_BEGIN (kept for possible restore)
        # if i % config.n_causal == 0:
        #     i_causal += 1
        # <<< CAUSAL_TRAINING_DISABLED_END
        
        # PDE损失
        #r_colloc = rand_colloc(config.n_colloc, data['L'], data['t'][i_causal], device)
        #r_colloc = rand_colloc_mixed(config.n_colloc, data['L'], data['t'][i_causal], device, ratio_gaussian=1)

        # >>> CAUSAL_TRAINING_DISABLED_BEGIN (kept for possible restore)
        # r_colloc = rand_colloc_fixed(config.n_colloc, data['L'], data['t'][i_causal], device, ratio_gaussian=config.ratio_gaussian)
        # <<< CAUSAL_TRAINING_DISABLED_END

        r_colloc = rand_colloc_fixed(config.n_colloc, data['L'], t_train_max, device, ratio_gaussian=config.ratio_gaussian)
        c_colloc = data['c_func'](r_colloc[:, 0:1], r_colloc[:, 1:2], tensor=True)
        with torch.backends.cudnn.flags(enabled=False):
            p = model(r_colloc)
        pde_res = pde_residual(p, r_colloc, c_colloc)
        loss_pde = mse_loss(pde_res, torch.zeros_like(p))

        # 吸收边界条件损失

        if use_bc:
            # >>> CAUSAL_TRAINING_DISABLED_BEGIN (kept for possible restore)
            # r_bc, side_bc = rand_boundary(config.n_bc, data['L'], data['t'][i_causal], device)
            # <<< CAUSAL_TRAINING_DISABLED_END
            
            r_bc, side_bc = rand_boundary(config.n_bc, data['L'], t_train_max, device)
            c_bc = data['c_func'](r_bc[:, 0:1], r_bc[:, 1:2], tensor=True)
            with torch.backends.cudnn.flags(enabled=False):
                p_bc = model(r_bc)
            bc_res = bc_residual_absorbing(p_bc, r_bc, side_bc, c_bc)
            loss_bc = mse_loss(bc_res, torch.zeros_like(p_bc))
        else:
            loss_bc = torch.tensor(0.0, device=device)

        # 更新lambda（只更新有梯度的损失项）
        if use_adaptive_lambda and i % config.n_lamb_update == 0:
            if use_bc:
                loss_lst = [loss_ini, loss_pde, loss_bc]
                #lamb = update_lambda(model, loss_lst, lamb, 0.9)
                #lamb_update = update_lambda(model, loss_lst[:2], lamb[:2], 0.9)
                #lamb[0], lamb[1] = lamb_update[0], lamb_update[1]
            else:
                # BC 关闭时，只更新 ini 和 pde 的 lambda
                loss_lst = [loss_ini, loss_pde]
                #lamb_update = update_lambda(model, loss_lst, lamb[:2], 0.9)
                #lamb[0], lamb[1] = lamb_update[0], lamb_update[1]
        
        # 总损失
        if use_stage_weights:
            progress = i / max(n_epochs - 1, 1)
            w_ini, w_pde, w_bc = get_stage_weights(progress, hp)
            lamb = [w_ini, w_pde, w_bc]
            # BC关闭时w_bc=0
            if not use_bc:
                w_bc = 0.0
            loss = w_ini * loss_ini + w_pde * loss_pde + w_bc * loss_bc
        else:
            # 自适应lambda
            #loss = lamb[0] * loss_ini + lamb[1] * loss_pde + lamb[2] * loss_bc
            loss = loss_ini + loss_pde * lamb[1] / lamb[0] + loss_bc * lamb[2] / lamb[0]
        
        # 反向传播
        loss.backward()
        # 梯度已累积完成，执行优化步骤
        optimizer.step()
        scheduler.step()
        """
        # ========== 预先准备数据 ==========
        n_ini_total = r_ini.shape[0]
        ini_chunk_size = config.ini_chunk
        
        r_colloc = rand_colloc_fixed(config.n_colloc, data['L'], data['t'][i_causal], device, ratio_gaussian=config.ratio_gaussian)
        # r_colloc = rand_colloc_mixed(config.n_colloc, data['L'], data['t'][i_causal], device, ratio_gaussian=config.ratio_gaussian)
        c_colloc = data['c_func'](r_colloc[:, 0:1], r_colloc[:, 1:2], tensor=True)
        colloc_chunk_size = config.colloc_chunk
        n_colloc_total = r_colloc.shape[0]
        
        if use_bc:
            r_bc, side_bc = rand_boundary(config.n_bc, data['L'], data['t'][i_causal], device)
            c_bc = data['c_func'](r_bc[:, 0:1], r_bc[:, 1:2], tensor=True)
            bc_chunk_size = config.bc_chunk
            n_bc_total = r_bc.shape[0]
        
        # ========== Lambda 更新（在梯度累积之前） ==========
        if use_adaptive_lambda and i % config.n_lamb_update == 0:
            # 用小批量样本计算 lambda（需要梯度信息）
            r_ini_small = r_ini[:min(config.ini_chunk, n_ini_total)]
            p_ini_small = p_ini[:min(config.ini_chunk, n_ini_total)]
            p_ini_pred = model(r_ini_small)
            loss_ini_lamb = mse_loss_sum(p_ini_pred, p_ini_small) / r_ini_small.shape[0]
            
            r_colloc_small = rand_colloc_fixed(min(config.colloc_chunk, n_colloc_total), data['L'], data['t'][i_causal], device, ratio_gaussian=config.ratio_gaussian)
            c_colloc_small = data['c_func'](r_colloc_small[:, 0:1], r_colloc_small[:, 1:2], tensor=True)
            r_colloc_small.requires_grad_(True)
            with torch.backends.cudnn.flags(enabled=False):
                p_small = model(r_colloc_small)
            pde_res_small = pde_residual(p_small, r_colloc_small, c_colloc_small)
            loss_pde_lamb = mse_loss_sum(pde_res_small, torch.zeros_like(pde_res_small)) / r_colloc_small.shape[0]
            
            if use_bc:
                r_bc_small, side_bc_small = rand_boundary(min(config.bc_chunk, n_bc_total), data['L'], data['t'][i_causal], device)
                c_bc_small = data['c_func'](r_bc_small[:, 0:1], r_bc_small[:, 1:2], tensor=True)
                r_bc_small.requires_grad_(True)
                with torch.backends.cudnn.flags(enabled=False):
                    p_bc_small = model(r_bc_small)
                bc_res_small = bc_residual_absorbing(p_bc_small, r_bc_small, side_bc_small, c_bc_small)
                loss_bc_lamb = mse_loss_sum(bc_res_small, torch.zeros_like(bc_res_small)) / r_bc_small.shape[0]
                
                loss_lst = [loss_ini_lamb, loss_pde_lamb, loss_bc_lamb]
                lamb = update_lambda(model, loss_lst, lamb, 0.9)
            else:
                loss_lst = [loss_ini_lamb, loss_pde_lamb]
                lamb_update = update_lambda(model, loss_lst, lamb[:2], 0.9)
                lamb[0], lamb[1] = lamb_update[0], lamb_update[1]
        
        # ========== 确定当前权重 ==========
        if use_stage_weights:
            progress = i / max(n_epochs - 1, 1)
            w_ini, w_pde, w_bc = get_stage_weights(progress, hp)
            if not use_bc:
                w_bc = 0.0
        else:
            # 自适应lambda模式
            w_ini = lamb[0].item() if hasattr(lamb[0], 'item') else lamb[0]
            w_pde = lamb[1].item() if hasattr(lamb[1], 'item') else lamb[1]
            w_bc = lamb[2].item() if hasattr(lamb[2], 'item') else lamb[2]
        
        # ========== 梯度累积（只计算一次，直接应用权重） ==========
        optimizer.zero_grad()
        loss_ini_sum = 0.0
        loss_pde_sum = 0.0
        loss_bc_sum = 0.0
        
        # 初始条件
        for start_idx in range(0, n_ini_total, ini_chunk_size):
            end_idx = min(start_idx + ini_chunk_size, n_ini_total)
            r_ini_chunk = r_ini[start_idx:end_idx]
            p_ini_chunk = p_ini[start_idx:end_idx]
            
            p_chunk = model(r_ini_chunk)
            loss_chunk = mse_loss_sum(p_chunk, p_ini_chunk)
            loss_ini_sum += loss_chunk.item()
            
            # 直接应用权重
            (w_ini * loss_chunk / n_ini_total).backward(retain_graph=False)
        
        # PDE残差
        for start_idx in range(0, n_colloc_total, colloc_chunk_size):
            end_idx = min(start_idx + colloc_chunk_size, n_colloc_total)
            r_chunk = r_colloc[start_idx:end_idx].detach().requires_grad_(True)
            c_chunk = c_colloc[start_idx:end_idx]
            
            with torch.backends.cudnn.flags(enabled=False):
                p_chunk = model(r_chunk)
            pde_res_chunk = pde_residual(p_chunk, r_chunk, c_chunk)
            loss_chunk = mse_loss_sum(pde_res_chunk, torch.zeros_like(pde_res_chunk))
            loss_pde_sum += loss_chunk.item()
            
            (w_pde * loss_chunk / n_colloc_total).backward(retain_graph=False)
        
        # 边界条件
        if use_bc:
            for start_idx in range(0, n_bc_total, bc_chunk_size):
                end_idx = min(start_idx + bc_chunk_size, n_bc_total)
                r_bc_chunk = r_bc[start_idx:end_idx].detach().requires_grad_(True)
                side_bc_chunk = side_bc[start_idx:end_idx]
                c_bc_chunk = c_bc[start_idx:end_idx]
                
                with torch.backends.cudnn.flags(enabled=False):
                    p_bc_chunk = model(r_bc_chunk)
                bc_res_chunk = bc_residual_absorbing(p_bc_chunk, r_bc_chunk, side_bc_chunk, c_bc_chunk)
                loss_chunk = mse_loss_sum(bc_res_chunk, torch.zeros_like(bc_res_chunk))
                loss_bc_sum += loss_chunk.item()
                
                (w_bc * loss_chunk / n_bc_total).backward(retain_graph=False)
        
        # 计算平均 loss（用于记录）
        loss_ini = loss_ini_sum / n_ini_total
        loss_pde = loss_pde_sum / n_colloc_total
        loss_bc = loss_bc_sum / n_bc_total if use_bc else 0.0
        
        # 梯度已累积完成，执行优化步骤
        optimizer.step()
        scheduler.step()

        # ========== 计算总 loss（用于日志记录）==========
        if use_stage_weights:
            total_loss = w_ini * loss_ini + w_pde * loss_pde + w_bc * loss_bc
            # 记录当前 stage weights 作为 lamb
            lamb = [w_ini, w_pde, w_bc]
        else:
            total_loss = w_ini * loss_ini + w_pde * loss_pde + w_bc * loss_bc
        """
        # 打印日志
        loss_pde_hist.append(loss_pde.item() if hasattr(loss_pde, 'item') else loss_pde)
        loss_ini_hist.append(loss_ini.item() if hasattr(loss_ini, 'item') else loss_ini)
        loss_bc_hist.append(loss_bc.item() if hasattr(loss_bc, 'item') else loss_bc)
        lamb_ini_hist.append(lamb[0].item() if hasattr(lamb[0], 'item') else lamb[0])
        lamb_pde_hist.append(lamb[1].item() if hasattr(lamb[1], 'item') else lamb[1])
        lamb_bc_hist.append(lamb[2].item() if hasattr(lamb[2], 'item') else lamb[2])

        if i % 200 == 199:
            print(f'[{i + 1:5d}] total_loss: {loss * 1e3:.3f}*1e-3 | ini: {loss_ini*1e3:.2f} | pde: {loss_pde*1e3:.2f} | bc: {loss_bc*1e3:.2f}*1e-3')

            #if not early_saved and loss.item() < loss_threshold:
            #    print(f'\n>>> Loss reached {loss.item():.6f} < {loss_threshold}, saving checkpoint...')
            #    loss_history = [loss_ini_hist, loss_pde_hist, loss_bc_hist]
            #    lamb_history = [lamb_ini_hist, lamb_pde_hist, lamb_bc_hist]
            #    save_bestmodel(model, loss_history, lamb_history, config)
            #    generate_all_bestfigures(model, data, loss_history, lamb_history, config, device)
            #    # early_saved = True
            #    print('>>> Checkpoint saved! Continuing training...\n')

        if i % 2800 == 2799:
            loss_history = [loss_ini_hist, loss_pde_hist, loss_bc_hist]
            lamb_history = [lamb_ini_hist, lamb_pde_hist, lamb_bc_hist]
            save_model(model, loss_history, lamb_history, config)
            generate_all_figures(model, data, loss_history, lamb_history, config, device)
            print('>>> Checkpoint saved! Continuing training...\n')

    loss_history = [loss_ini_hist, loss_pde_hist, loss_bc_hist]
    lamb_history = [lamb_ini_hist, lamb_pde_hist, lamb_bc_hist]
    
    return loss_history, lamb_history


def save_model(model, loss, lamb, config):
    """保存模型和训练历史"""
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), config.MODEL_DIR + "model.pt")
    torch.save(loss, config.MODEL_DIR + "loss.pt")
    torch.save(lamb, config.MODEL_DIR + "lamb.pt")
    print(f"Model saved to {config.MODEL_DIR}")

def save_bestmodel(model, loss, lamb, config):
    """保存模型和训练历史"""
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), config.MODEL_DIR + "best_model.pt")
    torch.save(loss, config.MODEL_DIR + "best_loss.pt")
    torch.save(lamb, config.MODEL_DIR + "best_lamb.pt")
    print(f"Model saved to {config.MODEL_DIR}")


def load_model(model, config, device):
    """加载预训练模型"""
    print("Loading pre-trained model...")
    model.load_state_dict(torch.load(
        config.MODEL_DIR + "model.pt", 
        weights_only=True, 
        map_location=torch.device('cpu')
    ))
    model.eval()
    model.to(device)
    
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(f"  {param_tensor}\t{model.state_dict()[param_tensor].size()}")

    loss = torch.load(config.MODEL_DIR + "loss.pt", weights_only=False, 
                      map_location=torch.device('cpu'))
    lamb = torch.load(config.MODEL_DIR + "lamb.pt", weights_only=False, 
                      map_location=torch.device('cpu'))
    print("Model loaded!")
    
    return loss, lamb


def evaluate_model(model, data):
    with torch.no_grad():
        p_est = model(data['r_ref']).cpu().numpy()
    p_ref = data['p_ref']

    # 初值误差
    n_L = data["n_L"]
    n_T = data["n_T"]
    p_est = p_est.reshape(-1)  # (n_L*n_L*n_T,)
    p_est = p_est.reshape(n_L, n_L, n_T)  # (n_L, n_L, n_T)
    # L2误差
    #l2_errors = []
    #sample_idxs = [0] + [int(i * n_T) for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]] + [n_T - 1]
    
    #for i in sample_idxs:
    #    err = np.linalg.norm(p_est[:, :, i] - p_ref[:, :, i])
    #    ref_norm = np.linalg.norm(p_ref[:, :, i]) + 1e-12
    #    l2_errors.append(err / ref_norm)
    p_est_flat = p_est.flatten()
    p_ref_flat = p_ref.flatten()
    err = np.linalg.norm(p_est_flat - p_ref_flat)
    ref_norm = np.linalg.norm(p_ref_flat) + 1e-12
    # 时间加权：后期时刻权重更高
    # weights = np.array([1.0, 1.2, 1.5, 1.8, 2.0])
    # weights = weights / weights.sum()
    L2 = err / ref_norm
    # L2_weighted = np.sum(np.array(l2_errors) * weights)
    #L2 = np.mean(np.array(l2_errors))

    # 能量守恒惩罚（双向，防止过冲和塌缩）
    #En_est = np.mean(p_est[:, :, -1]**2)
    #En_ref = np.mean(p_ref[:, :, -1]**2)
    #energy_penalty = np.abs(np.log((En_est + 1e-12) / (En_ref + 1e-12)))
    
    #return L2_weighted + 0.3 * energy_penalty
    return L2

# 可视化
def generate_all_figures(model, data, loss, lamb, config, device):
    """生成并保存所有图"""
    figures_dir = config.FIGURES_DIR
    os.makedirs(figures_dir, exist_ok=True)
    
    L, T, t = data['L'], data['T'], data['t']
    p_ref = data['p_ref']
    n_T, n_L = data['n_T'], data['n_L']
    
    # 1. Save wave speed
    save_wave_speed(
        data['c_func'](data['xx'], data['yy']), L,
        os.path.join(figures_dir, "wave_speed.png")
    )
    
    # 2. Save loss and lambda curves (with three curves: ini, pde, bc)
    label = ['ini', 'pde', 'bc']
    save_train_log(loss, lamb, label, os.path.join(figures_dir, "loss_lambda.png"))
    
    # 3. Generate estimation
    print("Generating reference vs estimation animation...")
    #with torch.backends.cudnn.flags(enabled=False):
    #    p_est = model(data['r_ref'])
    #p_est = p_est.cpu().detach().numpy()
    batch_size = 4096  # 可根据显存调整（与高分辨率推理一致）
    p_est_list = []
    with torch.no_grad():
        for i in range(0, data['r_ref'].shape[0], batch_size):
            batch = data['r_ref'][i:i+batch_size]
            with torch.backends.cudnn.flags(enabled=False):
                p_est_list.append(model(batch).cpu())
    p_est = torch.cat(p_est_list, dim=0).numpy()
    save_estimation_animation(p_ref, p_est, L, 
                              os.path.join(figures_dir, "estimation_animation.gif"))
    
    # 4. Save comparison plot
    save_comparison(p_ref, p_est, L, T, t, 
                    os.path.join(figures_dir, "comparison.png"))
    
    # 5. High resolution estimation
    print("Generating high resolution estimation...")
    t_hr = np.linspace(0, T, n_T * 2).astype(np.float32)
    r_hr = np.linspace(0, L, n_L * 2).astype(np.float32)
    
    x_hr, y_hr = np.meshgrid(r_hr, r_hr)
    r_hr_tensor = np.column_stack((np.reshape(x_hr, (-1, 1)), np.reshape(y_hr, (-1, 1))))
    r_hr_tensor = xyt_tensor(r_hr_tensor, t_hr, device)
    
    # 分批推理，防止显存溢出
    batch_size = 4096  # 可根据显存调整
    p_est_hr_list = []
    with torch.no_grad():
        for i in range(0, r_hr_tensor.shape[0], batch_size):
            batch = r_hr_tensor[i:i+batch_size]
            with torch.backends.cudnn.flags(enabled=False):
                p_est_hr_list.append(model(batch).cpu())
    p_est_hr = torch.cat(p_est_hr_list, dim=0).numpy()
    p_est_hr = np.reshape(p_est_hr, (n_L * 2, n_L * 2, n_T * 2))
    save_field_animation(p_est_hr, L, 'High Res Estimation', 
                         os.path.join(figures_dir, "high_res_estimation.gif"))
    
    print("\n" + "=" * 50)
    print(f"All figures saved to: {figures_dir}")
    print("=" * 50)


def generate_all_bestfigures(model, data, loss, lamb, config, device):
    """生成并保存所有图"""
    figures_dir = config.FIGURES_DIR
    os.makedirs(figures_dir, exist_ok=True)
    
    L, T, t = data['L'], data['T'], data['t']
    p_ref = data['p_ref']
    n_T, n_L = data['n_T'], data['n_L']
    
    # 1. Save wave speed
    save_wave_speed(
        data['c_func'](data['xx'], data['yy']), L,
        os.path.join(figures_dir, "best_wave_speed.png")
    )
    
    # 2. Save loss and lambda curves (with three curves: ini, pde, bc)
    label = ['ini', 'pde', 'bc']
    save_train_log(loss, lamb, label, os.path.join(figures_dir, "best_loss_lambda.png"))
    
    # 3. Generate estimation
    print("Generating reference vs estimation animation...")
    with torch.backends.cudnn.flags(enabled=False):
        p_est = model(data['r_ref'])
    p_est = p_est.cpu().detach().numpy()
    save_estimation_animation(p_ref, p_est, L, 
                              os.path.join(figures_dir, "best_estimation_animation.gif"))
    
    # 4. Save comparison plot
    save_comparison(p_ref, p_est, L, T, t, 
                    os.path.join(figures_dir, "best_comparison.png"))
    
    # 5. High resolution estimation
    print("Generating high resolution estimation...")
    t_hr = np.linspace(0, T, n_T * 2).astype(np.float32)
    r_hr = np.linspace(0, L, n_L * 2).astype(np.float32)
    
    x_hr, y_hr = np.meshgrid(r_hr, r_hr)
    r_hr_tensor = np.column_stack((np.reshape(x_hr, (-1, 1)), np.reshape(y_hr, (-1, 1))))
    r_hr_tensor = xyt_tensor(r_hr_tensor, t_hr, device)
    
    # 分批推理，防止显存溢出
    batch_size = 4096  # 可根据显存调整
    p_est_hr_list = []
    with torch.no_grad():
        for i in range(0, r_hr_tensor.shape[0], batch_size):
            batch = r_hr_tensor[i:i+batch_size]
            with torch.backends.cudnn.flags(enabled=False):
                p_est_hr_list.append(model(batch).cpu())
    p_est_hr = torch.cat(p_est_hr_list, dim=0).numpy()
    p_est_hr = np.reshape(p_est_hr, (n_L * 2, n_L * 2, n_T * 2))
    save_field_animation(p_est_hr, L, 'High Res Estimation', 
                         os.path.join(figures_dir, "best_high_res_estimation.gif"))
    
    print("\n" + "=" * 50)
    print(f"All figures saved to: {figures_dir}")
    print("=" * 50)

# 主函数
def main():
    """PINN训练主入口"""
    set_seed(0)
    config = Config()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    data = prepare_domain(config, device)
    configure_temporal_history_from_data(config, data)
    
    r_ini, p_ini = prepare_initial_condition(data, config, device)
    model = create_model(config, device)

    # 训练或加载模型
    if config.TRAIN_NEW_MODEL:
        #loss, lamb = train_model(model, data, r_ini, p_ini, config, device)
        best_hp = config.DEFAULT_HP
        temp = config.n_causal
        # -----------------------------
        # (Optional) BO search for stage hp
        # -----------------------------
        if config.RUN_BO and config.USE_STAGE_WEIGHTS:
            print(f"Running BO with GP + EI (trials={config.BO_N_TRIALS})...")

            def run_one(hp):
                set_seed(0)
                m = create_model(config, device)
                # BO阶段关闭自适应lambda
                config.n_causal = 100 # Reduce n_causal for BO
                train_model(
                     m, data, r_ini, p_ini, config, device,
                     use_adaptive_lambda=False,
                     use_stage_weights=True,
                     hp=hp,
                     max_epochs=None
                 )
                return evaluate_model(m, data)
            
            if _HAS_SKOPT:
                # 定义搜索空间 (log-uniform用Real的prior='log-uniform')
                search_space = [
                    Real(1e-5, 5e-3, prior='log-uniform', name='w_pde_A'),
                    Real(1e-4, 1e-3, prior='log-uniform', name='w_pde_B'),
                    Real(5e-5, 5e-4, prior='log-uniform', name='w_pde_C'),
                    Real(1.5, 2.5, prior='uniform', name='w_ini_B'),
                    Real(2.3, 3.0, prior='uniform', name='w_ini_C'),
                    Real(1e-4, 1e-1, prior='log-uniform', name='w_bc_C_max'),
                    Real(0.7, 0.8, prior='uniform', name='bc_ramp_start'),
                    Real(0.2, 0.3, prior='uniform', name='bc_ramp_len'),
                ]

                @use_named_args(search_space)
                def objective(**params):
                    """skopt目标函数，返回L2误差（最小化）"""
                    hp = {
                        "A_end": config.STAGE_A_END,
                        "B_end": config.STAGE_B_END,
                        **params
                    }
                    loss_val = run_one(hp)
                    print(f"  Trial params:")
                    print(f"    w_pde_A={params['w_pde_A']:.2e}, w_pde_B={params['w_pde_B']:.2e}, w_pde_C={params['w_pde_C']:.2e}")
                    print(f"    w_ini_B={params['w_ini_B']:.3f}, w_ini_C={params['w_ini_C']:.3f}")
                    print(f"    w_bc_C_max={params['w_bc_C_max']:.2e}, bc_ramp_start={params['bc_ramp_start']:.3f}, bc_ramp_len={params['bc_ramp_len']:.3f}")
                    print(f"    => L2={loss_val:.6f}")
                    return loss_val

                # GP代理模型 + EI采集函数
                result = gp_minimize(
                    func=objective,
                    dimensions=search_space,
                    acq_func='EI',           # Expected Improvement
                    n_calls=config.BO_N_TRIALS + 10,  # 总调用次数。先做n_init个初始化点（LHS），再做N_TRIALS次EI迭代更新
                    n_initial_points=10,      # 初始随机探索点数
                    initial_point_generator='lhs',
                    random_state=42,
                    verbose=True,
                    n_jobs=1                 # GPU训练不适合并行
                )

                # 提取最优超参
                best_hp = {
                    "A_end": config.STAGE_A_END,
                    "B_end": config.STAGE_B_END,
                    "w_pde_A": result.x[0],
                    "w_pde_B": result.x[1],
                    "w_pde_C": result.x[2],
                    "w_ini_B": result.x[3],
                    "w_ini_C": result.x[4],
                    "w_bc_C_max": result.x[5],
                    "bc_ramp_start": result.x[6],
                    "bc_ramp_len": result.x[7],
                }
                print(f"\n{'='*50}")
                print(f"BO completed! Best L2 loss: {result.fun:.6f}")
                print(f"Best HP: {best_hp}")
                print(f"{'='*50}\n")

                # 保存收敛曲线
                try:
                    from skopt.plots import plot_convergence
                    fig, ax = plt.subplots(figsize=(8, 5))
                    plot_convergence(result, ax=ax)
                    ax.set_xlabel('Number of calls', fontsize=12)
                    ax.set_ylabel('L2 relative error (min)', fontsize=12)
                    ax.set_title('BO Convergence (GP + EI)', fontsize=14)
                    fig.savefig(os.path.join(config.FIGURES_DIR, "bo_convergence.png"), dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    print(f"Saved BO convergence plot")
                except Exception as e:
                    print(f"Could not save convergence plot: {e}")


            elif _HAS_OPTUNA:
                print("skopt not available, falling back to Optuna...")
                def objective(trial):
                    hp = {
                        "A_end": config.STAGE_A_END,
                        "B_end": config.STAGE_B_END,
                        "w_pde_A": trial.suggest_float("w_pde_A", 1e-5, 1e-1, log=True),
                        "w_pde_B": trial.suggest_float("w_pde_B", 1e-4, 1e-1, log=True),
                        "w_pde_C": trial.suggest_float("w_pde_C", 1e-4, 1e-1, log=True),
                        "w_ini_B": trial.suggest_float("w_ini_B", 0.5, 3.0),
                        "w_ini_C": trial.suggest_float("w_ini_C", 0.5, 3.0),
                        "w_bc_C_max": trial.suggest_float("w_bc_C_max", 1e-4, 1e-1, log=True),
                        "bc_ramp_start": trial.suggest_float("bc_ramp_start", 0.7, 0.9),
                        "bc_ramp_len": trial.suggest_float("bc_ramp_len", 0.1, 0.3),
                    }
                    return run_one(hp)

                study = optuna.create_study(direction="minimize")
                study.optimize(objective, n_trials=config.BO_N_TRIALS, show_progress_bar=True)
                best_hp = study.best_params
                best_hp["A_end"] = config.STAGE_A_END
                best_hp["B_end"] = config.STAGE_B_END
                print("BO best hp:", best_hp)
            else:
                # 随机搜索兜底
                print("No BO library available, using random search...")
                rng = np.random.RandomState(0)
                best_val = float("inf")
                for k in range(config.BO_N_TRIALS):
                    hp = _random_hp(rng, config)
                    val = run_one(hp)
                    if val < best_val:
                        best_val = val
                        best_hp = hp
                    print(f"[RS] trial {k+1}/{config.BO_N_TRIALS}  val={val:.6f}  best={best_val:.6f}")
                print("Random-search best hp:", best_hp)

        # -----------------------------
        # 用最优超参正式训练
        # -----------------------------
        config.n_causal = temp  
        loss, lamb = train_model(
            model, data, r_ini, p_ini, config, device,
            use_adaptive_lambda=True,
            use_stage_weights=False,
            hp=best_hp,
            max_epochs=None
        )     
        save_model(model, loss, lamb, config)
        # 生成所有图
        generate_all_figures(model, data, loss, lamb, config, device)
        
    
    else:
        loss, lamb = load_model(model, config, device)
        # 修改时间参数
        config_extended = Config()
        config_extended.T = 2.0  # 延长时间至2s
        config_extended.FIGURES_DIR = "./figures/extended_T2s"  # 保存到新目录
        
        # 基于新的时间范围重新准备数据
        data_extended = prepare_domain(config_extended, device)
        
        # 生成预测结果并可视化
        os.makedirs(config_extended.FIGURES_DIR, exist_ok=True)
        generate_all_figures(model, data_extended, loss, lamb, config_extended, device)
        print(f"\n>>> Extended prediction (T=2s) figures saved to: {config_extended.FIGURES_DIR}")
    
    


if __name__ == "__main__":
    main()
