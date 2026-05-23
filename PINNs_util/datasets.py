"""Marmousi和Overthrust速度模型数据集生成"""
import os
import numpy as np
try:
    import torch
    from PINNs_util.PINNs_fdiff import solver
    from PINNs_util.PINNs_aux import xyt_tensor
except ModuleNotFoundError:
    torch = None
    solver = None
    xyt_tensor = None

DEFAULT_NX = 60
DEFAULT_NY = 60
# Default grid resolution is Nx=Ny=60 to match train_kan_pinn.py.

MARMOUSI_RAW_SHAPE = (751, 2301)  # z, x
MARMOUSI_RAW_DX_M = 4.0
OVERTHRUST_DX_M = 25.0


def _repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _default_data_dir():
    return os.path.join(_repo_root(), "datasets")


def _center_start(length, crop_size):
    if crop_size > length:
        raise ValueError(f"crop_size={crop_size} exceeds length={length}")
    return (length - crop_size) // 2


def _crop_square_2d(arr, crop_size=None, x_start=None, z_start=None):
    """Crop a square from a 2D velocity model arranged as [z, x]."""
    nz, nx = arr.shape
    if crop_size is None:
        crop_size = min(nz, nx)
    if z_start is None:
        z_start = _center_start(nz, crop_size)
    if x_start is None:
        x_start = _center_start(nx, crop_size)
    z_end = z_start + crop_size
    x_end = x_start + crop_size
    if z_start < 0 or x_start < 0 or z_end > nz or x_end > nx:
        raise ValueError(
            f"Invalid crop: shape={arr.shape}, crop_size={crop_size}, "
            f"z_start={z_start}, x_start={x_start}"
        )
    return arr[z_start:z_end, x_start:x_end], {
        "z_start": z_start,
        "z_end": z_end,
        "x_start": x_start,
        "x_end": x_end,
        "crop_size": crop_size,
    }


def _block_downsample_mean(arr, factor):
    if factor <= 1:
        return arr.copy()
    nz, nx = arr.shape
    nz2 = (nz // factor) * factor
    nx2 = (nx // factor) * factor
    trimmed = arr[:nz2, :nx2]
    return trimmed.reshape(nz2 // factor, factor, nx2 // factor, factor).mean(axis=(1, 3))


def load_raw_marmousi_crop(data_dir=None, crop_size=704, downsample=4):
    """Load marmousi_vp.bin and return a square Marmousi crop in km/s.

    The downloaded file has the common Marmousi layout 751 x 2301 [z, x]
    with 4 m spacing and velocity values in m/s.  A 704 x 704 crop followed
    by 4x block averaging gives a 176 x 176 model with 16 m spacing, close
    to the FF-PINN paper's Marmousi sampling.
    """
    data_dir = data_dir or _default_data_dir()
    path = os.path.join(data_dir, "marmousi_vp.bin")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    vp = np.fromfile(path, dtype=np.float32)
    expected = int(np.prod(MARMOUSI_RAW_SHAPE))
    if vp.size != expected:
        raise ValueError(f"Unexpected marmousi_vp.bin size: {vp.size}, expected {expected}")
    vp = vp.reshape(MARMOUSI_RAW_SHAPE, order="F")
    vp_km_s = vp / 1000.0 if float(np.nanmax(vp)) > 20.0 else vp
    # Use the full shallow-to-deep range and a centered x window.
    crop, meta = _crop_square_2d(vp_km_s, crop_size=crop_size, z_start=0)
    crop = _block_downsample_mean(crop, downsample)
    meta.update({
        "source_path": path,
        "raw_shape": MARMOUSI_RAW_SHAPE,
        "raw_dx_m": MARMOUSI_RAW_DX_M,
        "downsample": downsample,
        "dx_m": MARMOUSI_RAW_DX_M * downsample,
        "model": "marmousi",
    })
    return crop.astype(np.float32), meta


def load_raw_marmousi_extracted(data_dir=None, x_start=812, z_start=0, size=600, downsample=4):
    """Load a square Fig. 10-like Marmousi extracted region in km/s.

    The default 600 x 600 raw crop followed by 4x block averaging gives a
    150 x 150 square model with 16 m spacing.
    """
    data_dir = data_dir or _default_data_dir()
    path = os.path.join(data_dir, "marmousi_vp.bin")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    vp = np.fromfile(path, dtype=np.float32)
    expected = int(np.prod(MARMOUSI_RAW_SHAPE))
    if vp.size != expected:
        raise ValueError(f"Unexpected marmousi_vp.bin size: {vp.size}, expected {expected}")
    vp = vp.reshape(MARMOUSI_RAW_SHAPE, order="F")
    vp_km_s = vp / 1000.0 if float(np.nanmax(vp)) > 20.0 else vp
    z_end = z_start + size
    x_end = x_start + size
    if z_start < 0 or x_start < 0 or z_end > vp_km_s.shape[0] or x_end > vp_km_s.shape[1]:
        raise ValueError(f"Invalid Marmousi extracted crop: shape={vp_km_s.shape}, x={x_start}:{x_end}, z={z_start}:{z_end}")
    crop = vp_km_s[z_start:z_end, x_start:x_end]
    crop = _block_downsample_mean(crop, downsample)
    meta = {
        "source_path": path,
        "raw_shape": MARMOUSI_RAW_SHAPE,
        "raw_dx_m": MARMOUSI_RAW_DX_M,
        "downsample": downsample,
        "dx_m": MARMOUSI_RAW_DX_M * downsample,
        "x_start": x_start,
        "x_end": x_end,
        "z_start": z_start,
        "z_end": z_end,
        "model": "marmousi_extracted",
        "source_x_km": 0.5 * crop.shape[1] * MARMOUSI_RAW_DX_M * downsample / 1000.0,
        "source_z_km": 0.5 * crop.shape[0] * MARMOUSI_RAW_DX_M * downsample / 1000.0,
    }
    return crop.astype(np.float32), meta


def _load_overthrust_h5_velocity(path, key):
    import h5py
    with h5py.File(path, "r") as f:
        m = f[key][()]
    # SEG/EAGE HDF5 stores squared slowness in s^2/km^2; convert to km/s.
    return (1.0 / np.sqrt(np.maximum(m, 1e-12))).astype(np.float32)


def load_raw_overthrust_crop(data_dir=None, crop_size=201, use_initial=False, y_index=None):
    """Load an Overthrust 3D model and return a square 2D slice in km/s.

    For forward wave simulation use the true model.  The initial model is
    loaded only when explicitly requested, mainly for comparison/FWI starts.
    """
    data_dir = data_dir or _default_data_dir()
    filename = "overthrust_3D_initial_model.h5" if use_initial else "overthrust_3D_true_model.h5"
    key = "m0" if use_initial else "m"
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    vel = _load_overthrust_h5_velocity(path, key)
    # File layout is [z, y, x].  Take a central y crossline and crop x/z.
    if y_index is None:
        y_index = vel.shape[1] // 2
    section = vel[:, y_index, :]
    crop, meta = _crop_square_2d(section, crop_size=crop_size, z_start=0)
    meta.update({
        "source_path": path,
        "raw_shape": vel.shape,
        "raw_dx_m": OVERTHRUST_DX_M,
        "dx_m": OVERTHRUST_DX_M,
        "y_index": y_index,
        "model": "overthrust_initial" if use_initial else "overthrust_true",
    })
    return crop.astype(np.float32), meta


def load_raw_overthrust_extracted(data_dir=None, use_initial=False, y_index=300, x_start=280, z_start=0, size=97):
    """Load a square Fig. 10-like 2D Overthrust extracted region in km/s."""
    data_dir = data_dir or _default_data_dir()
    filename = "overthrust_3D_initial_model.h5" if use_initial else "overthrust_3D_true_model.h5"
    key = "m0" if use_initial else "m"
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    vel = _load_overthrust_h5_velocity(path, key)
    if y_index is None:
        y_index = vel.shape[1] // 2
    section = vel[:, y_index, :]
    z_end = z_start + size
    x_end = x_start + size
    if z_start < 0 or x_start < 0 or z_end > section.shape[0] or x_end > section.shape[1]:
        raise ValueError(f"Invalid Overthrust extracted crop: shape={section.shape}, x={x_start}:{x_end}, z={z_start}:{z_end}")
    crop = section[z_start:z_end, x_start:x_end]
    meta = {
        "source_path": path,
        "raw_shape": vel.shape,
        "raw_dx_m": OVERTHRUST_DX_M,
        "dx_m": OVERTHRUST_DX_M,
        "x_start": x_start,
        "x_end": x_end,
        "z_start": z_start,
        "z_end": z_end,
        "y_index": y_index,
        "model": "overthrust_initial_extracted" if use_initial else "overthrust_extracted",
        "source_x_km": 0.5 * crop.shape[1] * OVERTHRUST_DX_M / 1000.0,
        "source_z_km": 0.8 * crop.shape[0] * OVERTHRUST_DX_M / 1000.0,
    }
    return crop.astype(np.float32), meta


def save_raw_velocity_crops(data_dir=None, out_dir=None):
    """Save square crops and PNG previews for the downloaded raw models."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data_dir = data_dir or _default_data_dir()
    out_dir = out_dir or os.path.join(_repo_root(), "figures", "raw_velocity_crops")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    specs = [
        ("marmousi",) + load_raw_marmousi_crop(data_dir=data_dir),
        ("overthrust_true",) + load_raw_overthrust_crop(data_dir=data_dir, use_initial=False),
        ("overthrust_initial",) + load_raw_overthrust_crop(data_dir=data_dir, use_initial=True),
    ]
    saved = []
    for name, vel, meta in specs:
        npz_path = os.path.join(data_dir, f"{name}_square_crop.npz")
        np.savez_compressed(npz_path, velocity=vel, **meta)

        extent_km = vel.shape[1] * float(meta["dx_m"]) / 1000.0
        depth_km = vel.shape[0] * float(meta["dx_m"]) / 1000.0
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        im = ax.imshow(
            vel,
            cmap="turbo",
            origin="upper",
            extent=[0.0, extent_km, depth_km, 0.0],
            aspect="equal",
        )
        ax.set_title(f"{name} square crop ({vel.shape[1]} x {vel.shape[0]})")
        ax.set_xlabel("x (km)")
        ax.set_ylabel("z (km)")
        cb = fig.colorbar(im, ax=ax)
        cb.set_label("Velocity (km/s)")
        fig.tight_layout()
        png_path = os.path.join(out_dir, f"{name}_square_crop.png")
        fig.savefig(png_path, dpi=180)
        plt.close(fig)
        saved.append((npz_path, png_path, vel.shape, float(vel.min()), float(vel.max())))
    return saved


def _save_velocity_preview(velocity, meta, path, title, vmin=None, vmax=None, add_source=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dx_m = float(meta["dx_m"])
    extent_x = velocity.shape[1] * dx_m / 1000.0
    extent_z = velocity.shape[0] * dx_m / 1000.0
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    im = ax.imshow(
        velocity,
        cmap="viridis",
        origin="upper",
        extent=[0.0, extent_x, extent_z, 0.0],
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
    )
    if add_source:
        sx = float(meta["source_x_km"])
        sz = float(meta["source_z_km"])
        ax.plot(sx, sz, marker="o", markersize=10, markerfacecolor="white", markeredgecolor="black", markeredgewidth=2.0)
        ax.plot(sx, sz, marker=".", markersize=5, color="black")
    ax.set_title(title)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("z (km)")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Velocity (km/s)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_extracted_velocity_models(data_dir=None):
    """Save FF-PINN Fig. 10-like extracted models and previews under datasets/."""
    data_dir = data_dir or _default_data_dir()
    os.makedirs(data_dir, exist_ok=True)
    specs = [
        ("marmousi_extracted",) + load_raw_marmousi_extracted(data_dir=data_dir),
        ("overthrust_extracted",) + load_raw_overthrust_extracted(data_dir=data_dir, use_initial=False),
    ]
    saved = []
    for name, velocity, meta in specs:
        npz_path = os.path.join(data_dir, f"{name}.npz")
        png_path = os.path.join(data_dir, f"{name}.png")
        np.savez_compressed(npz_path, velocity=velocity, **meta)
        vmax = 4.2 if name.startswith("marmousi") else 6.0
        _save_velocity_preview(velocity, meta, png_path, name.replace("_", " "), vmin=1.5, vmax=vmax)
        saved.append((npz_path, png_path, velocity.shape, float(velocity.min()), float(velocity.max())))
    return saved


def _make_velocity_interpolator(velocity_km_s):
    """Create a bilinear velocity function on normalized [0, 1] x [0, 1]."""
    vel_np = np.asarray(velocity_km_s, dtype=np.float32)
    nz, nx = vel_np.shape

    def interp(xx, yy, tensor=False):
        if tensor:
            vel_t = torch.as_tensor(vel_np, dtype=xx.dtype, device=xx.device)
            x = torch.clamp(xx, 0.0, 1.0) * (nx - 1)
            y = torch.clamp(yy, 0.0, 1.0) * (nz - 1)
            x0 = torch.floor(x).long()
            y0 = torch.floor(y).long()
            x1 = torch.clamp(x0 + 1, max=nx - 1)
            y1 = torch.clamp(y0 + 1, max=nz - 1)
            wx = x - x0.to(dtype=xx.dtype)
            wy = y - y0.to(dtype=yy.dtype)
            v00 = vel_t[y0, x0]
            v10 = vel_t[y0, x1]
            v01 = vel_t[y1, x0]
            v11 = vel_t[y1, x1]
            return (1 - wx) * (1 - wy) * v00 + wx * (1 - wy) * v10 + (1 - wx) * wy * v01 + wx * wy * v11

        x = np.clip(xx, 0.0, 1.0) * (nx - 1)
        y = np.clip(yy, 0.0, 1.0) * (nz - 1)
        x0 = np.floor(x).astype(np.int64)
        y0 = np.floor(y).astype(np.int64)
        x1 = np.clip(x0 + 1, 0, nx - 1)
        y1 = np.clip(y0 + 1, 0, nz - 1)
        wx = x - x0
        wy = y - y0
        v00 = vel_np[y0, x0]
        v10 = vel_np[y0, x1]
        v01 = vel_np[y1, x0]
        v11 = vel_np[y1, x1]
        return (1 - wx) * (1 - wy) * v00 + wx * (1 - wy) * v10 + (1 - wx) * wy * v01 + wx * wy * v11

    return interp


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
    if torch is None or solver is None or xyt_tensor is None:
        raise RuntimeError("Generating wavefield datasets requires torch and PINNs training dependencies.")
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

    raw_path = os.path.join(_default_data_dir(), "marmousi_vp.bin")
    if os.path.exists(raw_path):
        velocity, meta = load_raw_marmousi_extracted()
        L_km = velocity.shape[1] * float(meta["dx_m"]) / 1000.0
        c_max = float(np.max(velocity))
        interp = _make_velocity_interpolator(velocity)

        def c_raw_func(xx_km, yy_km, tensor=False):
            return interp(xx_km / L_km, yy_km / L_km, tensor=tensor)

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

    raw_path = os.path.join(_default_data_dir(), "overthrust_3D_true_model.h5")
    if os.path.exists(raw_path):
        velocity, meta = load_raw_overthrust_extracted(use_initial=False)
        L_km = velocity.shape[1] * float(meta["dx_m"]) / 1000.0
        c_max = float(np.max(velocity))
        interp = _make_velocity_interpolator(velocity)

        def c_raw_func(xx_km, yy_km, tensor=False):
            return interp(xx_km / L_km, yy_km / L_km, tensor=tensor)

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
    import argparse
    parser = argparse.ArgumentParser(description="Generate or preview PINN velocity datasets.")
    parser.add_argument("--raw-crops", action="store_true", help="Save square crops and preview PNGs from downloaded raw models.")
    parser.add_argument("--extracted", action="store_true", help="Save Fig. 10-like extracted models and previews under datasets/.")
    args = parser.parse_args()

    if args.raw_crops:
        for npz_path, png_path, shape, vmin, vmax in save_raw_velocity_crops():
            print(f"Saved crop: {npz_path}")
            print(f"Saved preview: {png_path}  shape={shape} velocity=[{vmin:.3f}, {vmax:.3f}] km/s")
        raise SystemExit(0)

    if args.extracted:
        for npz_path, png_path, shape, vmin, vmax in save_extracted_velocity_models():
            print(f"Saved extracted: {npz_path}")
            print(f"Saved preview: {png_path}  shape={shape} velocity=[{vmin:.3f}, {vmax:.3f}] km/s")
        raise SystemExit(0)

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
