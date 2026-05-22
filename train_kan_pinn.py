import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import argparse
import random
import math
import time
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler

# 开启TF32加速
torch.set_float32_matmul_precision('high')
try:
    import torch._functorch.config as _ftc
    _ftc.donated_buffer = False
except Exception:
    try:
        torch._dynamo.config.donated_buffer = False
    except Exception:
        pass

from PINNs_util.PINNs_fdiff import solver
from PINNs_util.PINNs_aux import (
    xyt_tensor, pde_residual, rand_colloc_fixed,
    rand_boundary, bc_residual_absorbing, update_lambda
)
from PINNs_util.kan_pinn import KANPinn, KANSTPinn
from PINNs_util.datasets import (
    _marmousi_velocity,
    _overthrust_velocity,
    _threelayer_velocity,
    generate_marmousi,
    generate_overthrust,
)

KAN_TEMPORAL_VARIANTS = {
    "bilstm_attention": {
        "tag": "kan_st_pinn",
        "name": "KAN-ST-PINN (KAN+BiLSTM+Attention)",
    },
    "bilstm_no_attention": {
        "tag": "kan_bilstm_no_attention",
        "name": "KAN-BiLSTM-PINN (w/o Attention)",
    },
    "attention_no_bilstm": {
        "tag": "kan_attention_no_bilstm",
        "name": "KAN-Attention-PINN (w/o BiLSTM)",
    },
}


# =======
# 训练配置
# =======

class Config:
    L = 5.0           # km
    T = 2.0           # s
    c0 = 3.0          # km/s (normalization)
    Nx = 60
    Ny = 60

    # 网络结构
    d_model = 128        # hidden dimension (matches ST-PINN for fair comparison)
    n_fourier = 32       # Fourier feature frequencies
    spatial_kan_layers = 3  # KAN layers in spatial encoder
    grid_size = 5        # B-spline grid intervals
    spline_order = 3     # cubic B-splines
    seq_len = 8          # temporal window length
    dt_hist = 0.02       # temporal step
    lstm_layers = 2      # BiLSTM layers
    num_heads = 4        # attention heads
    n_attn_layers = 2    # cross-attention layers
    dropout = 0.0
    temporal_mode = "bilstm_attention"
    pure_kan = False     # if True, use pure KAN-PINN (no LSTM/Attention)

    # 训练参数
    n_epochs = 10000
    learning_rate = 1e-3     # KAN benefits from slightly higher LR
    n_ini = 20
    n_data = 0
    n_colloc = 4000
    n_bc = 0   #800
    ratio_gaussian = 0.85
    bc_type = 'absorbing'
    grad_clip = 1.0
    mini_batch_ini = 8000
    ini_chunk_size = 4000

    # 损失权重
    w_ini = 10.0
    w_pde = 0.01
    w_bc  = 0.01
    w_data = 0.0
    use_adaptive_lambda = True
    lambda_alpha = 0.9
    lambda_update_freq = 200

    # 因果加权
    use_causal = True
    n_causal = 100
    auto_epochs = True
    causal_eps = 3.0
    causal_eps_final = 0.5   # relax causal weighting over training

    # 晚期时刻加权
    late_data_alpha = 3.0    # time-weight: w_t = 1 + alpha*(t/T)^2 → 4x at t=T
    late_colloc_uniform = 0.5  # fraction of collocation with uniform-time (vs Gaussian)

    # PDE课程学习
    pde_warmup_epochs = 2000
    pde_rampup_epochs = 3000

    # 学习率调度
    warmup_epochs = 500
    lr_min_ratio = 0.01

    # 路径
    FIGURES_DIR = "./figures"
    MODEL_DIR   = "./trained"
    DATA_ROOT   = "./datasets"
    DATASET     = "threelayer"


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
    steepness = 40.0
    if tensor:
        sig = torch.sigmoid
        ones = torch.ones_like(xx, device=device)
    else:
        sig = lambda x: 1.0 / (1.0 + np.exp(-x))
        ones = np.ones_like(xx)
    c_val = 0.5 * ones + 0.25 * sig(steepness * (yy - 0.33)) + 0.25 * sig(steepness * (yy - 0.66))
    return c_val


def prepare_threelayer(config, device):
    L = config.L / config.L
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

    p_max = np.max(np.abs(p_ref))
    if p_max > 1e-12:
        p_ref = p_ref / p_max
        print(f"  p_ref normalized by p_max = {p_max:.6e}")

    xy = np.column_stack((xx.reshape(-1, 1), yy.reshape(-1, 1)))
    r_ref = xyt_tensor(xy, t, device)
    n_T = t.shape[0]
    n_L = xx.shape[0]
    L_phys = config.L
    T_phys = config.T

    return {
        'L': L, 'T': T,
        'L_phys': L_phys, 'T_phys': T_phys,
        'p_ref': p_ref, 'xx': xx, 'yy': yy, 't': t,
        'xy': xy, 'r_ref': r_ref, 'n_T': n_T, 'n_L': n_L,
        'c_func': c_func, 'p_max': p_max,
    }


def make_c_func(name):
    if name == 'threelayer':
        def c_func(xx, yy, tensor=False):
            return _threelayer_velocity(xx, yy, tensor=tensor)
        return c_func
    if name == 'marmousi':
        def c_func(xx, yy, tensor=False):
            return _marmousi_velocity(xx, yy, tensor=tensor) / 4.5
        return c_func
    if name == 'overthrust':
        def c_func(xx, yy, tensor=False):
            return _overthrust_velocity(xx, yy, tensor=tensor) / 6.0
        return c_func
    raise ValueError(f"Unknown dataset: {name}")


def load_npz_dataset(config, device):
    name = config.DATASET.lower()
    path = os.path.join(config.DATA_ROOT, f"{name}.npz")
    if not os.path.exists(path):
        return None

    raw = np.load(path)
    xx = raw["xx"]
    yy = raw["yy"]
    if xx.shape[0] != config.Nx or yy.shape[0] != config.Ny:
        print(
            f"Dataset file {path} has grid {xx.shape[0]}x{yy.shape[0]}, "
            f"but config requests {config.Nx}x{config.Ny}; regenerating."
        )
        return None

    t = raw["t"]
    xy = np.column_stack((xx.reshape(-1, 1), yy.reshape(-1, 1)))
    r_ref = xyt_tensor(xy, t, device)
    c_func = make_c_func(name)
    return {
        'L': float(raw["L"]),
        'T': float(raw["T"]),
        'L_phys': float(raw["L_phys"]) if "L_phys" in raw.files else float(raw["L"]),
        'T_phys': float(raw["T_phys"]) if "T_phys" in raw.files else float(raw["T"]),
        'p_ref': raw["p_ref"],
        'xx': xx,
        'yy': yy,
        't': t,
        'xy': xy,
        'r_ref': r_ref,
        'n_T': t.shape[0],
        'n_L': xx.shape[0],
        'c': raw["c"] if "c" in raw.files else c_func(xx, yy, tensor=False),
        'c_func': c_func,
        'p_max': float(raw["p_max"]) if "p_max" in raw.files else 1.0,
    }


def prepare_dataset(config, device):
    name = config.DATASET.lower()
    data = load_npz_dataset(config, device)
    if data is not None:
        print(f"Loaded dataset from {os.path.join(config.DATA_ROOT, name + '.npz')}")
        return data
    if name == 'threelayer':
        return prepare_threelayer(config, device)
    elif name == 'marmousi':
        return generate_marmousi(device, Nx=config.Nx, Ny=config.Ny)
    elif name == 'overthrust':
        return generate_overthrust(device, Nx=config.Nx, Ny=config.Ny)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def prepare_initial_condition(data, config, device):
    t_ini = data['t'][:config.n_ini]
    r_ini = xyt_tensor(data['xy'], t_ini, device)
    p_ini = data['p_ref'][:, :, :config.n_ini].reshape(-1, 1)
    p_ini = torch.tensor(p_ini, device=device)
    return r_ini, p_ini


# =======
# 模型构建
# =======

def align_temporal_config(config, data):
    """对齐dt_hist和seq_len到求解器时间网格"""
    if data['n_T'] > 1:
        dt_data = float(data['t'][1] - data['t'][0])
        if dt_data > 0:
            config.dt_hist = dt_data
    import numpy as np
    t_span = float(data['t'][-1] - data['t'][0]) if data['n_T'] > 1 else 0.0
    max_seq = max(1, int(np.floor((t_span + 1e-12) / config.dt_hist)) + 1)
    if config.seq_len > max_seq:
        print(f"[temporal] seq_len {config.seq_len} -> {max_seq}")
        config.seq_len = max_seq


def create_model(config, device):
    if config.pure_kan:
        # 纯KAN-PINN（消融对比用）
        model = KANPinn(
            d_hidden=config.d_model,
            n_layers=config.spatial_kan_layers,
            n_fourier=config.n_fourier,
            grid_size=config.grid_size,
            spline_order=config.spline_order,
        ).to(device)
        tag = "KAN-PINN (pure)"
    else:
        # KAN-ST-PINN
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
        tag = KAN_TEMPORAL_VARIANTS[config.temporal_mode]["name"]

    param_info = model.count_parameters()
    print(f"[{tag}] Created model: {param_info['total']:,} parameters")
    for k, v in param_info.items():
        if k != 'total':
            if isinstance(v, int):
                print(f"  {k}: {v:,}")
            else:
                print(f"  {k}: {v}")
    print(f"  Config: d_model={config.d_model}, kan_layers={config.spatial_kan_layers}, "
          f"grid={config.grid_size}, spline_k={config.spline_order}")
    if not config.pure_kan:
        print(f"  temporal_mode={config.temporal_mode}")
    return model


# =======
# 训练循环
# =======

def sample_data_points(data, n_data, device):
    """采样监督数据点"""
    n_total = data['r_ref'].shape[0]
    idx = torch.randint(0, n_total, (n_data,), device=device)
    r_data = data['r_ref'][idx]
    p_data = data['p_ref_gpu'][idx]  # pre-loaded GPU tensor
    return r_data, p_data


def sample_data_time_weighted(data, n_data, device, t_max):
    """时间分层采样, 加强晚期覆盖"""
    n_L = data['n_L']
    n_T = data['n_T']
    n_spatial = n_L * n_L
    n_total = data['r_ref'].shape[0]

    # 前后两半分层
    mid_t = n_T // 2
    n_early = mid_t * n_spatial
    n_late = (n_T - mid_t) * n_spatial

    # 4:6比例，侧重晚期
    n_early_samples = int(n_data * 0.4)
    n_late_samples = n_data - n_early_samples

    idx_early = torch.randint(0, n_early, (n_early_samples,), device=device)
    idx_late = torch.randint(0, n_late, (n_late_samples,), device=device) + n_early
    idx = torch.cat([idx_early, idx_late])

    # 打乱
    perm = torch.randperm(idx.shape[0], device=device)
    idx = idx[perm]

    r_data = data['r_ref'][idx]
    p_data = data['p_ref_gpu'][idx]
    return r_data, p_data


def _pregenerate_colloc_pool(pool_size, n_colloc, L, T, device, ratio_gaussian=0.85,
                              late_uniform_frac=0.0):
    """预生成配置点池"""
    pools = []
    n_base = int(n_colloc * (1.0 - late_uniform_frac))
    n_unif_t = n_colloc - n_base
    for _ in range(pool_size):
        parts = []
        if n_base > 0:
            r = rand_colloc_fixed(n_base, L, T, device, ratio_gaussian=ratio_gaussian)
            parts.append(r.detach())
        if n_unif_t > 0:
            # 均匀采样，保证晚期PDE覆盖
            xy_u = torch.rand((n_unif_t, 2), device=device) * L
            t_u = torch.rand((n_unif_t, 1), device=device) * T
            r_u = torch.cat([xy_u, t_u], dim=1)
            parts.append(r_u)
        pools.append(torch.cat(parts, dim=0).detach())
    return pools


def sample_colloc(config, data, device, t_max):
    return _pregenerate_colloc_pool(
        1,
        config.n_colloc,
        data['L'],
        t_max,
        device,
        ratio_gaussian=config.ratio_gaussian,
        late_uniform_frac=getattr(config, 'late_colloc_uniform', 0.0),
    )[0]


def causal_time_limit(epoch, data, config):
    if not getattr(config, 'use_causal', False):
        return float(data['t'][-1]), data['n_T'] - 1
    i_causal = min(epoch // max(1, int(config.n_causal)) + 1, data['n_T'] - 1)
    return float(data['t'][i_causal]), i_causal


def adapt_training_schedule(config, data):
    config.n_causal = max(1, int(getattr(config, 'n_causal', 1)))
    if getattr(config, 'use_causal', False) and getattr(config, 'auto_epochs', False):
        config.n_epochs = int(config.n_causal * (data['n_T'] - 1))
    else:
        config.n_epochs = max(1, int(config.n_epochs))


def mse_chunked(model, r, target, chunk_size):
    n_total = r.shape[0]
    if chunk_size <= 0 or n_total <= chunk_size:
        pred = model(r)
        return torch.mean((pred - target) ** 2)

    loss_sum = torch.zeros((), device=r.device)
    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        pred = model(r[start:end])
        loss_sum = loss_sum + torch.sum((pred - target[start:end]) ** 2)
    return loss_sum / n_total


def second_derivative_attention_context():
    if not torch.cuda.is_available():
        return nullcontext()

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        return sdpa_kernel(SDPBackend.MATH)
    except Exception:
        pass

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "sdp_kernel"):
        try:
            return torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_mem_efficient=False,
                enable_math=True,
            )
        except TypeError:
            return torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_math=True,
            )

    return nullcontext()


def train_model(model, data, r_ini, p_ini, config, device, save_dir=None, fig_dir=None,
                start_epoch=0, optimizer_state=None, history_prev=None):
    n_epochs = config.n_epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    torch.backends.cudnn.benchmark = True

    # torch.compile和create_graph=True不兼容，不用
    compiled_model = model
    print("  Using TF32 matmul + pre-generated collocation pools")

    # 预生成配置点和边界点
    pool_size = 100  # recycle every 100 epochs
    t_max_pre = float(data['t'][-1])
    use_bc = (config.bc_type == 'absorbing' and config.n_bc > 0)
    if getattr(config, 'use_causal', False):
        colloc_pool = None
        print(
            f"  Causal PDE time window enabled: advance one time step every "
            f"{max(1, int(config.n_causal))} epochs"
        )
    else:
        print(f"  Pre-generating {pool_size} collocation pools (n_colloc={config.n_colloc})...")
        colloc_pool = _pregenerate_colloc_pool(
            pool_size, config.n_colloc, data['L'], t_max_pre, device,
            ratio_gaussian=config.ratio_gaussian,
            late_uniform_frac=getattr(config, 'late_colloc_uniform', 0.0))
    bc_pool = []
    if use_bc:
        print(f"  Pre-generating {pool_size} boundary pools (n_bc={config.n_bc})...")
        for _ in range(pool_size):
            r_bc, side = rand_boundary(config.n_bc, data['L'], t_max_pre, device)
            bc_pool.append((r_bc.detach(), side))
    else:
        print("  Boundary loss disabled (n_bc <= 0 or bc_type != 'absorbing')")

    # 预加载监督数据到GPU
    use_data = (config.n_data > 0)
    if use_data and 'p_ref_gpu' not in data:
        p_flat = data['p_ref'].flatten()
        data['p_ref_gpu'] = torch.tensor(p_flat, dtype=torch.float32, device=device).unsqueeze(1)
        print(f"  Pre-loaded p_ref on GPU ({data['p_ref_gpu'].shape[0]} points)")

    warmup = config.warmup_epochs
    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, n_epochs - warmup)
        return config.lr_min_ratio + (1 - config.lr_min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # 恢复优化器状态（断点续训）
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    if start_epoch > 0:
        scheduler.last_epoch = start_epoch

    mse = nn.MSELoss()

    t_max = float(data['t'][-1])

    if history_prev is not None:
        history = history_prev
        history.setdefault('train_time_sec', 0.0)
    else:
        history = {'loss_ini': [], 'loss_pde': [], 'loss_bc': [],
                   'loss_data': [], 'loss_total': [], 'train_time_sec': 0.0}
    history.setdefault('lamb_ini', [])
    history.setdefault('lamb_pde', [])
    history.setdefault('lamb_bc', [])
    history.setdefault('lambda_ratio_pde', [])
    history.setdefault('lambda_ratio_bc', [])

    mini_bs = config.mini_batch_ini
    n_ini_total = r_ini.shape[0]
    lamb = [
        torch.tensor(1.0, device=device),
        torch.tensor(1.0, device=device),
        torch.tensor(1.0, device=device),
    ]
    if history['lamb_ini'] and history['lamb_pde'] and history['lamb_bc']:
        lamb = [
            torch.tensor(float(history['lamb_ini'][-1]), device=device),
            torch.tensor(float(history['lamb_pde'][-1]), device=device),
            torch.tensor(float(history['lamb_bc'][-1]), device=device),
        ]

    status_file = os.path.join(save_dir, 'status.txt') if save_dir else None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    best_sup_loss = float('inf')  # Track best supervised loss for fixed PDE cap

    print(f"Training KAN-PINN for {n_epochs} epochs (device={device}), starting from epoch {start_epoch}...")
    prior_train_time = float(history.get('train_time_sec', 0.0) or 0.0)
    train_start_time = time.time()
    for epoch in tqdm(range(start_epoch, n_epochs), desc='Training', ncols=80,
                      mininterval=30, file=sys.stderr):
        optimizer.zero_grad()

        # 初始条件损失
        if mini_bs <= 0 or n_ini_total <= mini_bs:
            loss_ini = mse_chunked(compiled_model, r_ini, p_ini, config.ini_chunk_size)
        else:
            idx = torch.randperm(n_ini_total, device=device)[:mini_bs]
            p_pred = compiled_model(r_ini[idx])
            loss_ini = mse(p_pred, p_ini[idx])

        # PDE残差
        t_colloc_max, i_causal = causal_time_limit(epoch, data, config)
        if getattr(config, 'use_causal', False):
            r_colloc = sample_colloc(config, data, device, t_colloc_max).requires_grad_(True)
        else:
            r_colloc = colloc_pool[epoch % pool_size].clone().requires_grad_(True)
        c_colloc = data['c_func'](r_colloc[:, 0:1], r_colloc[:, 1:2], tensor=True)
        #with torch.backends.cudnn.flags(enabled=False):
        with second_derivative_attention_context(), torch.backends.cudnn.flags(enabled=False):
            p_colloc = model(r_colloc)
        pde_res = pde_residual(p_colloc, r_colloc, c_colloc)
        if False:
            # 因果eps松弛：初期严格，逐渐放松
            progress = min(1.0, epoch / max(1, n_epochs * 0.8))
            eps_now = config.causal_eps + (config.causal_eps_final - config.causal_eps) * progress
            t_frac = r_colloc[:, 2:3].detach() / t_max
            causal_w = torch.exp(-eps_now * t_frac)
            loss_pde = torch.mean(causal_w * pde_res ** 2)
        else:
            loss_pde = mse(pde_res, torch.zeros_like(p_colloc))

        # 边界条件
        if use_bc:
            r_bc_det, side = bc_pool[epoch % pool_size]
            r_bc = r_bc_det.clone().requires_grad_(True)
            c_bc = data['c_func'](r_bc[:, 0:1], r_bc[:, 1:2], tensor=True)
            #with torch.backends.cudnn.flags(enabled=False):
            with second_derivative_attention_context(), torch.backends.cudnn.flags(enabled=False):
                p_bc = model(r_bc)
            bc_res = bc_residual_absorbing(p_bc, r_bc, side, c_bc)
            loss_bc = mse(bc_res, torch.zeros_like(p_bc))
        else:
            loss_bc = torch.tensor(0.0, device=device)

        # 数据监督（时间加权）
        if use_data:
            r_data, p_data_target = sample_data_time_weighted(data, config.n_data, device, t_max)
            p_data_pred = compiled_model(r_data)
            # 时间加权MSE，强化晚期
            t_frac = r_data[:, 2:3].detach() / (t_max + 1e-12)
            time_weight = 1.0 + config.late_data_alpha * (t_frac ** 2)
            loss_data = torch.mean(time_weight * (p_data_pred - p_data_target) ** 2)
        else:
            loss_data = torch.tensor(0.0, device=device)

        # PDE课程学习权重
        pde_warmup = config.pde_warmup_epochs
        pde_rampup = config.pde_rampup_epochs
        pde_ramp_factor = 1.0
        bc_ramp_factor = 1.0 if use_bc else 0.0
        if False and epoch < pde_warmup:
            pde_ramp_factor = 0.0
            bc_ramp_factor = 0.0
        elif False and epoch < pde_warmup + pde_rampup and pde_rampup > 0:
            ramp_frac = (epoch - pde_warmup) / max(1, pde_rampup)
            pde_ramp_factor = ramp_frac
            bc_ramp_factor = ramp_frac if use_bc else 0.0

        # PDE/BC自归一化，防止B样条梯度爆炸
        if config.use_adaptive_lambda:
            if epoch % config.lambda_update_freq == 0:
                if use_bc:
                    lamb = update_lambda(model, [loss_ini, loss_pde, loss_bc], lamb, config.lambda_alpha)
                else:
                    lamb_2 = update_lambda(model, [loss_ini, loss_pde], lamb[:2], config.lambda_alpha)
                    lamb = [lamb_2[0], lamb_2[1], torch.tensor(1.0, device=device)]
            ratio_pde = lamb[1] / (lamb[0] + 1e-12)
            ratio_bc = lamb[2] / (lamb[0] + 1e-12) if use_bc else torch.tensor(0.0, device=device)
        else:
            ratio_pde = torch.tensor(config.w_pde, device=device)
            ratio_bc = torch.tensor(config.w_bc if use_bc else 0.0, device=device)


        # 监督损失
        if config.use_adaptive_lambda:
            loss_phys = ratio_pde * loss_pde
            loss = loss_ini + loss_phys
            if use_bc:
                loss = loss + ratio_bc * loss_bc
            if use_data:
                loss = loss + config.w_data * loss_data
            max_phys = float('nan')
        else:
            loss_sup = config.w_ini * loss_ini
            if use_data:
                loss_sup = loss_sup + config.w_data * loss_data

        # 记录监督损失最优值
            sup_val = loss_sup.item()
            if sup_val < best_sup_loss:
                best_sup_loss = sup_val

        # 物理损失cap: PDE+BC不超过最优监督损失的50%
            loss_phys = pde_ramp_factor * ratio_pde * loss_pde + bc_ramp_factor * ratio_bc * loss_bc
            max_phys = best_sup_loss * 0.5
            if loss_phys.item() > max_phys and max_phys > 1e-10:
                loss_phys = loss_phys * (max_phys / (loss_phys.detach().item() + 1e-10))
            loss = loss_sup + loss_phys
        loss.backward()

        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        optimizer.step()
        scheduler.step()

        # 记录
        history['loss_ini'].append(loss_ini.item())
        history['loss_pde'].append(loss_pde.item())
        history['loss_bc'].append(loss_bc.item())
        history['loss_data'].append(loss_data.item() if use_data else 0.0)
        history['loss_total'].append(loss.item())
        history['lamb_ini'].append(float(lamb[0].detach().cpu()))
        history['lamb_pde'].append(float(lamb[1].detach().cpu()))
        history['lamb_bc'].append(float(lamb[2].detach().cpu()))
        history['lambda_ratio_pde'].append(float(ratio_pde.detach().cpu()))
        history['lambda_ratio_bc'].append(float(ratio_bc.detach().cpu()))
        history['train_time_sec'] = prior_train_time + (time.time() - train_start_time)

        if epoch % 100 == 99:
            lr_now = optimizer.param_groups[0]['lr']
            if config.use_adaptive_lambda:
                print(f"  [{epoch+1}] total={loss.item():.3e}  ini={loss_ini.item():.3e}"
                      f"  pde={loss_pde.item():.3e}  bc={loss_bc.item():.3e}"
                      f"  data={loss_data.item():.3e}  lr={lr_now:.2e}"
                      f"  lambda_ratio=[{ratio_pde.item():.3e},{ratio_bc.item():.3e}]"
                      f"  t_pde={t_colloc_max:.3f} i_causal={i_causal}")
            else:
                cap_info = f"  cap={max_phys:.2e}" if pde_ramp_factor > 0 else ""
                print(f"  [{epoch+1}] total={loss.item():.3e}  ini={loss_ini.item():.3e}"
                      f"  pde={loss_pde.item():.3e}  bc={loss_bc.item():.3e}"
                      f"  data={loss_data.item():.3e}  lr={lr_now:.2e}"
                      f"  w=[{float(config.w_pde * pde_ramp_factor):.3e},{float(config.w_ini):.1f},"
                      f"{float(config.w_bc * bc_ramp_factor):.3e},{float(config.w_data):.1f}]"
                      f"{cap_info}")

        # 每10轮写状态文件
        if status_file and epoch % 10 == 0:
            with open(status_file, 'w') as _sf:
                _sf.write(f"{epoch+1}/{n_epochs} loss={loss.item():.4e} lr={optimizer.param_groups[0]['lr']:.2e}\n")

        if save_dir and epoch % 100 == 99:
            os.makedirs(save_dir, exist_ok=True)
            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'history': history,
                'epoch': epoch + 1,
                'train_time_sec': history['train_time_sec'],
            }, os.path.join(save_dir, f'checkpoint_epoch{epoch+1}.pt'))
            torch.save(history, os.path.join(save_dir, 'history.pt'))

        if fig_dir and (epoch + 1) % 1000 == 0:
            os.makedirs(fig_dir, exist_ok=True)
            if save_dir:
                torch.save(history, os.path.join(save_dir, 'history.pt'))
            was_training = model.training
            model.eval()
            metrics = evaluate_model(model, data, device)
            save_loss_curves(history, fig_dir)
            save_comparison_plot(metrics['p_est'], data, fig_dir)
            if was_training:
                model.train()

    history['train_time_sec'] = prior_train_time + (time.time() - train_start_time)
    return history


# =======
# 评估与可视化
# =======

def evaluate_model(model, data, device):
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
    os.makedirs(save_dir, exist_ok=True)
    n_L = data['n_L']
    n_T = data['n_T']
    p_ref = data['p_ref']
    p_est_3d = p_est.reshape(n_L, n_L, n_T)

    L_phys = data.get('L_phys', data['L'])
    T_phys = data.get('T_phys', float(data['t'][-1]))
    t_norm = data['t']
    t_phys_arr = t_norm / (t_norm[-1] + 1e-12) * T_phys

    p_max = np.max(np.abs(p_ref))
    time_indices = [0, n_T // 3, 2 * n_T // 3, n_T - 1]
    n_cols = len(time_indices)

    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(4.2 * n_cols + 1.6, 11))
    gs_main = GridSpec(3, 1, figure=fig, hspace=0.38, top=0.95, bottom=0.05,
                       left=0.07, right=0.88)

    row_labels = ['Reference', 'KAN-ST-PINN', 'Error']
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

        first_ax = fig.axes[-n_cols]
        first_ax.text(-0.35, 0.5, row_labels[row_idx],
                      transform=first_ax.transAxes, fontsize=12,
                      va='center', ha='center', rotation=90, fontweight='bold')

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

def run(config, device, resume=False):
    tag = "kan_pinn" if config.pure_kan else KAN_TEMPORAL_VARIANTS[config.temporal_mode]["tag"]
    out_dir = os.path.join(config.MODEL_DIR, config.DATASET, tag)
    fig_dir = os.path.join(config.FIGURES_DIR, config.DATASET, tag)

    set_seed(0)
    data = prepare_dataset(config, device)
    if not config.pure_kan:
        align_temporal_config(config, data)
    adapt_training_schedule(config, data)

    r_ini, p_ini = prepare_initial_condition(data, config, device)
    model = create_model(config, device)

    start_epoch = 0
    optimizer_state = None
    history_prev = None

    if resume:
        # 找到最新checkpoint
        ckpt_files = sorted(
            [f for f in os.listdir(out_dir) if f.startswith('checkpoint_epoch') and f.endswith('.pt')],
            key=lambda f: int(f.replace('checkpoint_epoch', '').replace('.pt', ''))
        ) if os.path.isdir(out_dir) else []
        if ckpt_files:
            latest = ckpt_files[-1]
            start_epoch = int(latest.replace('checkpoint_epoch', '').replace('.pt', ''))
            ckpt_path = os.path.join(out_dir, latest)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            if isinstance(ckpt, dict) and 'model_state' in ckpt:
                model.load_state_dict(ckpt['model_state'])
                optimizer_state = ckpt.get('optimizer_state')
                history_prev = ckpt.get('history')
            else:
                model.load_state_dict(ckpt)
                hist_path = os.path.join(out_dir, 'history.pt')
                if os.path.exists(hist_path):
                    history_prev = torch.load(hist_path, weights_only=False)
            print(f"Resumed from {latest} (epoch {start_epoch})")
        else:
            print("No checkpoint found, starting from scratch.")

    history = train_model(model, data, r_ini, p_ini, config, device, save_dir=out_dir, fig_dir=fig_dir,
                          start_epoch=start_epoch, optimizer_state=optimizer_state,
                          history_prev=history_prev)

    os.makedirs(out_dir, exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'history': history,
        'epoch': config.n_epochs,
        'train_time_sec': history.get('train_time_sec', 0.0),
    }, os.path.join(out_dir, 'model.pt'))
    torch.save(history, os.path.join(out_dir, 'history.pt'))

    metrics = evaluate_model(model, data, device)
    model_name = 'KAN-PINN' if config.pure_kan else KAN_TEMPORAL_VARIANTS[config.temporal_mode]["name"]
    print(f"\n[{model_name}] Relative L2 = {metrics['rel_l2']:.6f}  MAE = {metrics['mae']:.6f}")
    print(f"[{model_name}] Train time = {history.get('train_time_sec', 0.0):.1f}s ({history.get('train_time_sec', 0.0)/3600:.3f}h)")

    save_loss_curves(history, fig_dir)
    save_comparison_plot(metrics['p_est'], data, fig_dir)

    return metrics


def run_ablation(config, device, resume=False):
    """Run KAN-ST-PINN temporal ablations under the same training protocol."""
    if config.pure_kan:
        raise ValueError("--ablation is only for KAN-ST temporal variants; do not combine it with --pure-kan.")

    results = {}
    ablation_modes = ("bilstm_no_attention", "attention_no_bilstm")
    # 消融实验包含KANSTPINN
    # for temporal_mode, meta in KAN_TEMPORAL_VARIANTS.items():
    for temporal_mode in ablation_modes:
        meta = KAN_TEMPORAL_VARIANTS[temporal_mode]
        print(f"\n{'='*60}")
        print(f"  KAN-ST ABLATION: {meta['name']} on {config.DATASET}")
        print(f"  temporal_mode={temporal_mode}")
        print(f"{'='*60}")

        config.temporal_mode = temporal_mode
        metrics = run(config, device, resume=resume)
        results[temporal_mode] = metrics

    print(f"\n{'='*60}")
    print(f"  KAN-ST ABLATION SUMMARY - {config.DATASET}")
    print(f"{'='*60}")
    print(f"{'Variant':<26} {'Rel L2':>12} {'MAE':>12}")
    print("-" * 52)
    for temporal_mode, metrics in results.items():
        print(f"{temporal_mode:<26} {metrics['rel_l2']:>12.6f} {metrics['mae']:>12.6f}")

    summary_dir = os.path.join(config.FIGURES_DIR, config.DATASET)
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, "kan_st_ablation_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"{'Variant':<26} {'Rel L2':>12} {'MAE':>12}\n")
        f.write("-" * 52 + "\n")
        for temporal_mode, metrics in results.items():
            f.write(f"{temporal_mode:<26} {metrics['rel_l2']:>12.6f} {metrics['mae']:>12.6f}\n")
    print(f"Saved: {summary_path}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description='KAN-PINN Training for 2D Acoustic Wave')
    parser.add_argument('--dataset', type=str, default='threelayer',
                       choices=['threelayer', 'marmousi', 'overthrust', 'all'])
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--nx', type=int, default=None)
    parser.add_argument('--T', type=float, default=None)
    parser.add_argument('--w-pde', type=float, default=None)
    parser.add_argument('--w-bc', type=float, default=None)
    parser.add_argument('--n-colloc', type=int, default=None)
    parser.add_argument('--n-bc', type=int, default=None)
    parser.add_argument('--mini-batch-ini', type=int, default=None,
                       help='Initial-condition samples per epoch; <= 0 uses all IC points')
    parser.add_argument('--ini-chunk-size', type=int, default=None,
                       help='Chunk size used when mini_batch_ini <= 0')
    parser.add_argument('--data-root', type=str, default=None)
    parser.add_argument('--d-model', type=int, default=None,
                       help='Model hidden dimension (default: 128)')
    parser.add_argument('--kan-layers', type=int, default=None,
                       help='Number of spatial KAN layers (default: 3)')
    parser.add_argument('--grid-size', type=int, default=None,
                       help='B-spline grid intervals (default: 5)')
    parser.add_argument('--spline-order', type=int, default=None,
                       help='B-spline order (default: 3, cubic)')
    parser.add_argument('--pure-kan', action='store_true',
                       help='Use pure KAN-PINN (no LSTM/Attention) for ablation')
    parser.add_argument('--ablation', action='store_true',
                       help='Run KAN-ST temporal ablations')
    parser.add_argument('--pde-warmup', type=int, default=None)
    parser.add_argument('--pde-rampup', type=int, default=None)
    parser.add_argument('--adaptive-lambda', dest='adaptive_lambda', action='store_true', default=None,
                       help='Use gradient-norm adaptive weights for IC/PDE/BC losses')
    parser.add_argument('--fixed-weights', dest='adaptive_lambda', action='store_false',
                       help='Use fixed w_ini/w_pde/w_bc weights with the legacy physics cap')
    parser.add_argument('--lambda-update-freq', type=int, default=None,
                       help='Adaptive loss-weight update interval in epochs')
    parser.add_argument('--lambda-alpha', type=float, default=None,
                       help='EMA smoothing factor for adaptive loss weights')
    parser.add_argument('--causal', dest='use_causal', action='store_true', default=None,
                       help='Use Raissi-style causal PDE time-window training')
    parser.add_argument('--no-causal', dest='use_causal', action='store_false',
                       help='Disable causal PDE time-window training')
    parser.add_argument('--n-causal', type=int, default=None,
                       help='Epochs per causal time-step; if --epochs is omitted, total epochs = n_causal * (n_T - 1)')
    parser.add_argument('--quick', action='store_true',
                       help='Fast screening mode')
    parser.add_argument('--with-data', type=int, default=None,
                       help='Enable reference wavefield data supervision with this many samples per epoch')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from latest checkpoint')
    # 消融设置
    parser.add_argument(
        "--temporal-mode",
        choices=list(KAN_TEMPORAL_VARIANTS.keys()),
        default="bilstm_attention",
        help='KAN-ST temporal mode for a single run',
    )
    return parser.parse_args()


def config_from_args(args, dataset):
    config = Config()
    config.DATASET = dataset

    if args.quick:
        config.n_epochs = 600
        config.auto_epochs = False
        config.Nx = 40
        config.Ny = 40
        config.T = 1.3
        config.d_model = 96
        config.n_colloc = 1000
        config.n_bc = 300
        config.pde_warmup_epochs = 150
        config.pde_rampup_epochs = 150

    # 命令行覆盖
    if args.epochs is not None:
        config.n_epochs = args.epochs
        config.auto_epochs = False
    if args.lr is not None:        config.learning_rate = args.lr
    if args.nx is not None:        config.Nx = args.nx; config.Ny = args.nx
    if args.T is not None:         config.T = args.T
    if args.w_pde is not None:     config.w_pde = args.w_pde
    if args.w_bc is not None:      config.w_bc = args.w_bc
    if args.n_colloc is not None:  config.n_colloc = args.n_colloc
    if args.n_bc is not None:      config.n_bc = args.n_bc
    if args.mini_batch_ini is not None: config.mini_batch_ini = args.mini_batch_ini
    if args.ini_chunk_size is not None: config.ini_chunk_size = args.ini_chunk_size
    if args.data_root is not None: config.DATA_ROOT = args.data_root
    if args.d_model is not None:   config.d_model = args.d_model
    if args.kan_layers is not None: config.spatial_kan_layers = args.kan_layers
    if args.grid_size is not None: config.grid_size = args.grid_size
    if args.spline_order is not None: config.spline_order = args.spline_order
    if args.pure_kan:              config.pure_kan = True
    config.temporal_mode = args.temporal_mode
    if args.pde_warmup is not None:   config.pde_warmup_epochs = args.pde_warmup
    if args.pde_rampup is not None:   config.pde_rampup_epochs = args.pde_rampup
    if args.adaptive_lambda is not None: config.use_adaptive_lambda = args.adaptive_lambda
    if args.lambda_update_freq is not None: config.lambda_update_freq = args.lambda_update_freq
    if args.lambda_alpha is not None: config.lambda_alpha = args.lambda_alpha
    if args.use_causal is not None: config.use_causal = args.use_causal
    if args.n_causal is not None: config.n_causal = args.n_causal
    if args.with_data is not None:
        config.n_data = args.with_data
        config.w_data = 10.0

    return config


def print_run_config(config, mode):
    print(f"Dataset: {config.DATASET}")
    print(f"Mode: {mode}")
    print(f"Config: d_model={config.d_model}, kan_layers={config.spatial_kan_layers}, "
          f"grid_size={config.grid_size}, spline_order={config.spline_order}")
    print(f"Training points: n_ini={config.n_ini}, mini_batch_ini={config.mini_batch_ini}, "
          f"ini_chunk_size={config.ini_chunk_size}, n_colloc={config.n_colloc}, "
          f"n_bc={config.n_bc}, n_data={config.n_data}")
    print(f"Loss weighting: {'adaptive' if config.use_adaptive_lambda else 'fixed'}")
    if config.use_adaptive_lambda:
        print(f"Adaptive lambda: update_freq={config.lambda_update_freq}, alpha={config.lambda_alpha}")
    if config.use_causal:
        print(f"Causal PDE: n_causal={config.n_causal}, auto_epochs={config.auto_epochs}")


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if args.ablation and args.pure_kan:
        raise SystemExit("--ablation cannot be combined with --pure-kan")

    datasets = ['threelayer', 'marmousi', 'overthrust'] if args.dataset == 'all' else [args.dataset]
    for dataset in datasets:
        print("\n" + "=" * 80)
        print(f"Training dataset: {dataset}")
        print("=" * 80)
        config = config_from_args(args, dataset)
        mode = 'KAN-PINN (pure)' if config.pure_kan else KAN_TEMPORAL_VARIANTS[config.temporal_mode]["name"]
        print_run_config(config, mode)

        if args.ablation:
            run_ablation(config, device, resume=args.resume)
        else:
            run(config, device, resume=args.resume)


if __name__ == '__main__':
    main()
