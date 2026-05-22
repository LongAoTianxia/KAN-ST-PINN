"""Marmousi和Overthrust速度模型数据集生成"""
import os
import numpy as np
import torch
from PINNs_util.PINNs_fdiff import solver
from PINNs_util.PINNs_aux import xyt_tensor

DEFAULT_NX = 60
DEFAULT_NY = 60
# Default grid resolution is Nx=Ny=60 to match train_kan_pinn.py.


# ===
# 速度模型生成
# ===

def _marmousi_velocity(xx, yy, tensor=False):
    """合成Marmousi速度场: 层状结构+横向变化+低速反转区, 1.5-4.5 km/s"""
    if tensor:
        sin = torch.sin
        cos = torch.cos
        exp = torch.exp
        ones = torch.ones_like(xx)
    else:
        sin = np.sin
        cos = np.cos
        exp = np.exp
        ones = np.ones_like(xx)

    pi = np.pi

    # 深度梯度
    c = 1.5 + 2.5 * yy

    # 横向扰动
    c = c + 0.3 * sin(3 * pi * xx) * cos(2 * pi * yy)
    c = c + 0.2 * sin(5 * pi * xx + 0.5) * sin(4 * pi * yy)

    # 低速透镜体（速度反转区）
    lens_center_y = 0.45
    lens_sigma = 0.06
    lens_amp = -0.8
    if tensor:
        lens = lens_amp * exp(-((yy - lens_center_y) ** 2) / (2 * lens_sigma ** 2))
    else:
        lens = lens_amp * exp(-((yy - lens_center_y) ** 2) / (2 * lens_sigma ** 2))
    c = c + lens * (0.5 + 0.5 * cos(2 * pi * xx))

    # 截断到物理范围
    if tensor:
        c = torch.clamp(c, 1.5, 4.5)
    else:
        c = np.clip(c, 1.5, 4.5)

    return c


def _overthrust_velocity(xx, yy, tensor=False):
    """合成Overthrust速度场: 强层状+逆冲断层, 2.0-6.0 km/s"""
    if tensor:
        sin = torch.sin
        cos = torch.cos
        where = torch.where
        ones = torch.ones_like(xx)
        zeros = torch.zeros_like(xx)
    else:
        sin = np.sin
        cos = np.cos
        where = np.where
        ones = np.ones_like(xx)
        zeros = np.zeros_like(xx)

    pi = np.pi

    # 层状模型+深度梯度
    c = 2.0 + 3.5 * yy

    # 锐利层界面
    c = c + 0.4 * where(yy > 0.25, ones, zeros)
    c = c + 0.5 * where(yy > 0.55, ones, zeros)
    c = c + 0.3 * where(yy > 0.75, ones, zeros)

    # 逆冲断层倾斜界面
    fault_y = 0.3 + 0.35 * xx
    c = c + 0.6 * where(yy > fault_y, ones, zeros)

    # 地质横向非均质性
    c = c + 0.25 * sin(4 * pi * xx) * cos(3 * pi * yy)

    # 截断
    if tensor:
        c = torch.clamp(c, 2.0, 6.0)
    else:
        c = np.clip(c, 2.0, 6.0)

    return c


def _threelayer_velocity(xx, yy, tensor=False):
    """三层速度模型: 0.5-1.0 km/s (归一化), 用于验证"""
    if tensor:
        sig = torch.sigmoid
        ones = torch.ones_like(xx)
    else:
        sig = lambda x: 1.0 / (1.0 + np.exp(-x))
        ones = np.ones_like(xx)

    steepness = 40.0
    c_val = 0.5 * ones + 0.25 * sig(steepness * (yy - 0.33)) + 0.25 * sig(steepness * (yy - 0.66))
    
    if tensor:
        c_val = torch.clamp(c_val, 0.5, 1.0)
    else:
        c_val = np.clip(c_val, 0.5, 1.0)
    
    return c_val


# ===
# 数据集生成函数
# ===

def _build_dataset(c_raw_func, c_max, L_km, T_s, Nx, Ny, gpulse_center, gpulse_std, device,
                   normalize_velocity=True):
    """通用数据集构建"""
    # 归一化计算域
    L = L_km / L_km   # = 1.0
    T = T_s * c_max / L_km

    # 速度归一化 c/c_max
    def c_func(xx, yy, tensor=False):
        c_raw = c_raw_func(xx * L_km, yy * L_km, tensor=tensor)
        return c_raw / c_max if normalize_velocity else c_raw

    # 初始条件: 高斯脉冲
    cx, cy = gpulse_center
    def I_func(xx, yy):
        return np.exp(-0.5 * (((xx - cx) / gpulse_std) ** 2 +
                               ((yy - cy) / gpulse_std) ** 2))

    # 有限差分求解
    print(f"正在算FD参考解 (Nx={Nx}, Ny={Ny})...")
    p_ref, xx, yy, t, dt = solver(I_func, c_func, L, L, Nx, Ny, -1, T)

    # 声压归一化到[-1,1]
    p_max = np.max(np.abs(p_ref))
    if p_max > 1e-12:
        p_ref = p_ref / p_max
        print(f"  p_ref normalized by p_max = {p_max:.6e}")

    xy = np.column_stack((xx.reshape(-1, 1), yy.reshape(-1, 1)))
    r_ref = xyt_tensor(xy, t, device)
    n_T = t.shape[0]
    n_L = xx.shape[0]
    c = c_func(xx, yy, tensor=False)

    return {
        'L': L, 'T': T,
        'L_phys': L_km, 'T_phys': T_s,
        'p_ref': p_ref, 'xx': xx, 'yy': yy, 't': t,
        'xy': xy, 'r_ref': r_ref, 'n_T': n_T, 'n_L': n_L,
        'c': c, 'c_func': c_func, 'p_max': p_max,
    }


# Marmousi defaults to Nx=Ny=60, matching train_kan_pinn.py.
def generate_marmousi(device, Nx=DEFAULT_NX, Ny=DEFAULT_NY):
    """Marmousi数据集: 3km×3km, T=1s, 1.5-4.5 km/s"""
    L_km = 3.0
    T_s = 1.0
    c_max = 4.5

    # c_raw_func在物理坐标上计算, _build_dataset内部归一化
    def c_raw_func(xx_km, yy_km, tensor=False):
        # km → [0,1]
        return _marmousi_velocity(xx_km / L_km, yy_km / L_km, tensor=tensor)

    return _build_dataset(
        c_raw_func=c_raw_func,
        c_max=c_max,
        L_km=L_km,
        T_s=T_s,
        Nx=Nx, Ny=Ny,
        gpulse_center=(0.5, 0.15),
        gpulse_std=0.04,
        device=device,
    )


# Overthrust defaults to Nx=Ny=60, matching train_kan_pinn.py.
def generate_overthrust(device, Nx=DEFAULT_NX, Ny=DEFAULT_NY):
    """Overthrust数据集: 4km×4km, T=0.8s, 2.0-6.0 km/s"""
    L_km = 4.0
    T_s = 0.8
    c_max = 6.0

    def c_raw_func(xx_km, yy_km, tensor=False):
        return _overthrust_velocity(xx_km / L_km, yy_km / L_km, tensor=tensor)

    return _build_dataset(
        c_raw_func=c_raw_func,
        c_max=c_max,
        L_km=L_km,
        T_s=T_s,
        Nx=Nx, Ny=Ny,
        gpulse_center=(0.5, 0.12),
        gpulse_std=0.035,
        device=device,
    )


# Three-layer defaults to Nx=Ny=60 and keeps the same normalized c_func
# as train_kan_pinn.py; c_max is only used to compute normalized T.
def generate_threelayer(device, Nx=DEFAULT_NX, Ny=DEFAULT_NY):
    """Three-layer数据集: 5km×5km, T=2.0s, 0.5-1.0 km/s (实际), c0=3.0 km/s (参考)
    
    注意: c_max=3.0 为参考波速，与 train_kan_pinn.py 中 Config.c0=3.0 保持一致
         实际物理波速范围为 0.5-1.0 km/s
    """
    L_km = 5.0
    T_s = 2.0
    c_max = 3.0  # 参考波速（与train_kan_pinn中的c0一致，而非实际最大波速1.0）

    def c_raw_func(xx_km, yy_km, tensor=False):
        # 三层速度模型在归一化坐标上计算
        return _threelayer_velocity(xx_km / L_km, yy_km / L_km, tensor=tensor)

    return _build_dataset(
        c_raw_func=c_raw_func,
        c_max=c_max,
        L_km=L_km,
        T_s=T_s,
        Nx=Nx, Ny=Ny,
        gpulse_center=(0.5, 0.5),
        gpulse_std=0.05,
        device=device,
        normalize_velocity=False,
    )


# ===
# 单独运行: 生成数据并保存
# ===
if __name__ == "__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'datasets')
    os.makedirs(save_dir, exist_ok=True)

    for name, gen_fn in [('marmousi', generate_marmousi),
                         ('overthrust', generate_overthrust),
                         ('threelayer', generate_threelayer)]:
        print(f"\n{'='*50}")
        print(f"Generating {name} dataset...")
        data = gen_fn(device, Nx=DEFAULT_NX, Ny=DEFAULT_NY)
        path = os.path.join(save_dir, f'{name}.npz')
        np.savez_compressed(
            path,
            p_ref=data['p_ref'], xx=data['xx'], yy=data['yy'], t=data['t'],
            c=data['c'], L=data['L'], T=data['T'],
            L_phys=data['L_phys'], T_phys=data['T_phys'], p_max=data['p_max'],
        )
        print(f"Saved to {path}  (p_ref shape: {data['p_ref'].shape})")
