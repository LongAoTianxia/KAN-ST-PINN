"""
Evaluate Rel L2 and PDE residual metrics for trained PINN experiments.
Usage: python compute_l2.py
"""
import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch

from PINNs_util.PINNs_fdiff import solver
from PINNs_util.PINNs_aux import FCN, pde_residual, xyt_tensor
from PINNs_util.kan_pinn import KANPinn, KANSTPinn
from PINNs_util.datasets import generate_marmousi, generate_overthrust
from PINNs_util.datasets import _marmousi_velocity, _overthrust_velocity, _threelayer_velocity
from PINNs_util.gabor_wave_pinn import GaborEnhancedWavePINN, GaborWavePINN
from PINNs_util.ff_pinn import HardInitialFFPINN
from PINNs_util.td_pinn import TDSinePINN
from PINNs_util.pinnsformer_wave import PINNsFormerWave2D


# ---------- Config defaults (must match training) ----------
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

L_KM = 5.0
T_SEC = 2.0
C0 = 3.0
NX = 60
NY = 60

MODEL_DIR = "./trained"
DATA_ROOT = "./datasets"
PDE_BATCH_SIZE = 512
PDE_MAX_POINTS = 20000  # deterministic full-domain sample for second-derivative metrics


def get_wave_speed_threelayer(xx, yy, device, tensor=False):
    steepness = 40.0
    if tensor:
        sig = torch.sigmoid
        ones = torch.ones_like(xx, device=device)
    else:
        sig = lambda x: 1.0 / (1.0 + np.exp(-x))
        ones = np.ones_like(xx)
    return 0.5 * ones + 0.25 * sig(steepness * (yy - 0.33)) + 0.25 * sig(steepness * (yy - 0.66))


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


def load_npz_dataset(name, device, data_root=DATA_ROOT):
    path = os.path.join(data_root, f"{name}.npz")
    if not os.path.exists(path):
        return None
    raw = np.load(path)
    xx = raw["xx"]
    yy = raw["yy"]
    t = raw["t"]
    xy = np.column_stack((xx.reshape(-1, 1), yy.reshape(-1, 1)))
    r_ref = xyt_tensor(xy, t, device)
    c_func = make_c_func(name)
    return {
        'p_ref': raw["p_ref"],
        'r_ref': r_ref,
        'n_T': t.shape[0],
        'n_L': xx.shape[0],
        't': t,
        'L': float(raw["L"]),
        'T': float(raw["T"]),
        'L_phys': float(raw["L_phys"]) if "L_phys" in raw.files else float(raw["L"]),
        'T_phys': float(raw["T_phys"]) if "T_phys" in raw.files else float(raw["T"]),
        'c': raw["c"] if "c" in raw.files else c_func(xx, yy, tensor=False),
        'c_func': c_func,
        'p_max': float(raw["p_max"]) if "p_max" in raw.files else 1.0,
    }


def prepare_threelayer(device):
    L = L_KM / L_KM
    T = T_SEC * C0 / L_KM

    def c_func(xx, yy, tensor=False):
        return get_wave_speed_threelayer(xx, yy, device, tensor)

    gpulse_std = 5e-2
    r_pulse = np.array([0.5, 0.5])
    def I_func(xx, yy):
        return np.exp(-0.5 * (((xx - r_pulse[0]) / gpulse_std) ** 2 +
                               ((yy - r_pulse[1]) / gpulse_std) ** 2))

    print("Solving FD reference (threelayer)...")
    p_ref, xx, yy, t, dt = solver(I_func, c_func, L, L, NX, NY, -1, T)
    p_max = np.max(np.abs(p_ref))
    if p_max > 1e-12:
        p_ref = p_ref / p_max
    xy = np.column_stack((xx.reshape(-1, 1), yy.reshape(-1, 1)))
    r_ref = xyt_tensor(xy, t, device)
    c = c_func(xx, yy, tensor=False)
    return {'p_ref': p_ref, 'r_ref': r_ref, 'n_T': t.shape[0], 'n_L': xx.shape[0],
            't': t, 'L': L, 'T': T, 'c': c, 'c_func': c_func}


def prepare_dataset(name, device):
    data = load_npz_dataset(name, device)
    if data is not None:
        print(f"Loaded dataset from {os.path.join(DATA_ROOT, name + '.npz')}")
        return data
    if name == 'threelayer':
        return prepare_threelayer(device)
    elif name == 'marmousi':
        return generate_marmousi(device, Nx=NX, Ny=NY)
    elif name == 'overthrust':
        return generate_overthrust(device, Nx=NX, Ny=NY)


def align_temporal(data):
    """根据数据的dt调整seq_len"""
    dt = DT_HIST
    sl = SEQ_LEN
    if data['n_T'] > 1:
        dt_data = float(data['t'][1] - data['t'][0])
        if dt_data > 0:
            dt = dt_data
    t_span = float(data['t'][-1] - data['t'][0]) if data['n_T'] > 1 else 0.0
    max_seq = max(1, int(np.floor((t_span + 1e-12) / dt)) + 1)
    if sl > max_seq:
        sl = max_seq
    return dt, sl


def create_model(pure_kan, device, dt_hist=DT_HIST, seq_len=SEQ_LEN,
                 temporal_mode="bilstm_attention"):
    if pure_kan:
        model = KANPinn(
            d_hidden=D_MODEL, n_layers=SPATIAL_KAN_LAYERS,
            n_fourier=N_FOURIER, grid_size=GRID_SIZE, spline_order=SPLINE_ORDER,
        ).to(device)
    else:
        model = KANSTPinn(
            d_model=D_MODEL, seq_len=seq_len, dt_hist=dt_hist,
            n_fourier=N_FOURIER, spatial_kan_layers=SPATIAL_KAN_LAYERS,
            grid_size=GRID_SIZE, spline_order=SPLINE_ORDER,
            lstm_layers=LSTM_LAYERS, num_heads=NUM_HEADS,
            n_attn_layers=N_ATTN_LAYERS, dropout=DROPOUT,
            temporal_mode=temporal_mode,
        ).to(device)
    return model


def create_raissi_model(device, model_dir):
    model_path = os.path.join(model_dir, 'model.pt')
    config = {'n_hidden': 128, 'n_layers': 6, 'n_ffeatures': 0}
    state = None
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location='cpu', weights_only=False)
        if isinstance(state, dict):
            config.update(state.get('model_config', {}))
    else:
        ckpts = sorted(
            [f for f in os.listdir(model_dir) if f.startswith('checkpoint_epoch') and f.endswith('.pt')],
            key=lambda f: int(f.replace('checkpoint_epoch', '').replace('.pt', ''))
        )
        if not ckpts:
            raise FileNotFoundError(f"No Raissi-PINN checkpoint found in {model_dir}")
        state = torch.load(os.path.join(model_dir, ckpts[-1]), map_location='cpu', weights_only=False)
        if isinstance(state, dict):
            config.update(state.get('model_config', {}))

    model = FCN(
        n_in=3,
        n_out=1,
        n_ffeatures=int(config.get('n_ffeatures', 0)),
        n_hidden=int(config.get('n_hidden', 128)),
        n_layers=int(config.get('n_layers', 6)),
    ).to(device)

    if isinstance(state, dict) and 'model_state' in state:
        model.load_state_dict(state['model_state'])
        ckpt_name = 'model.pt' if os.path.exists(model_path) else f"checkpoint_epoch{state.get('epoch', 'latest')}.pt"
    elif state is not None:
        model.load_state_dict(state)
        ckpt_name = 'model.pt'
    else:
        raise FileNotFoundError(f"No Raissi-PINN checkpoint found in {model_dir}")
    return model, ckpt_name


def create_ff_pinn_model(device, model_dir):
    model_path = os.path.join(model_dir, 'model.pt')
    state = None
    ckpt_name = 'model.pt'
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location='cpu', weights_only=False)
    else:
        ckpts = sorted(
            [f for f in os.listdir(model_dir) if f.startswith('checkpoint_epoch') and f.endswith('.pt')],
            key=lambda f: int(f.replace('checkpoint_epoch', '').replace('.pt', ''))
        )
        if not ckpts:
            raise FileNotFoundError(f"No FF-PINN checkpoint found in {model_dir}")
        ckpt_name = ckpts[-1]
        state = torch.load(os.path.join(model_dir, ckpt_name), map_location='cpu', weights_only=False)

    if not isinstance(state, dict) or 'model_config' not in state:
        raise ValueError(f"FF-PINN checkpoint must include model_config: {model_dir}")
    config = state['model_config']
    model = HardInitialFFPINN(**config).to(device)
    if 'model_state' in state:
        model.load_state_dict(state['model_state'])
    else:
        model.load_state_dict(state)
    model._training_config = state.get('training_config', {}) if isinstance(state, dict) else {}
    return model, ckpt_name


def create_td_pinn_model(device, model_dir):
    model_path = os.path.join(model_dir, 'model.pt')
    state = None
    ckpt_name = 'model.pt'
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location='cpu', weights_only=False)
    else:
        ckpts = sorted(
            [f for f in os.listdir(model_dir) if f.startswith('checkpoint_epoch') and f.endswith('.pt')],
            key=lambda f: int(f.replace('checkpoint_epoch', '').replace('.pt', ''))
        )
        if not ckpts:
            raise FileNotFoundError(f"No TD-PINN checkpoint found in {model_dir}")
        ckpt_name = ckpts[-1]
        state = torch.load(os.path.join(model_dir, ckpt_name), map_location='cpu', weights_only=False)

    if not isinstance(state, dict) or 'model_config' not in state:
        raise ValueError(f"TD-PINN checkpoint must include model_config: {model_dir}")
    model = TDSinePINN(**state['model_config']).to(device)
    if 'model_state' in state:
        model.load_state_dict(state['model_state'])
    else:
        model.load_state_dict(state)
    return model, ckpt_name


def create_pinnsformer_model(device, model_dir):
    model_path = os.path.join(model_dir, 'model.pt')
    state = None
    ckpt_name = 'model.pt'
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location='cpu', weights_only=False)
    else:
        ckpts = sorted(
            [f for f in os.listdir(model_dir) if f.startswith('checkpoint_epoch') and f.endswith('.pt')],
            key=lambda f: int(f.replace('checkpoint_epoch', '').replace('.pt', ''))
        )
        if not ckpts:
            raise FileNotFoundError(f"No PINNsFormer checkpoint found in {model_dir}")
        ckpt_name = ckpts[-1]
        state = torch.load(os.path.join(model_dir, ckpt_name), map_location='cpu', weights_only=False)

    if not isinstance(state, dict) or 'model_config' not in state:
        raise ValueError(f"PINNsFormer checkpoint must include model_config: {model_dir}")
    model = PINNsFormerWave2D(**state['model_config']).to(device)
    if 'model_state' in state:
        model.load_state_dict(state['model_state'])
    else:
        model.load_state_dict(state)
    return model, ckpt_name


def _build_gabor_like_model(config, model_kind, device):
    if model_kind == 'gepinn':
        return GaborEnhancedWavePINN(**config).to(device)
    return GaborWavePINN(**config).to(device)


def create_gabor_model(device, model_dir, model_kind='gabor'):
    model_path = os.path.join(model_dir, 'model.pt')
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location='cpu', weights_only=False)
        config = state.get('model_config', {}) if isinstance(state, dict) else {}
        model = _build_gabor_like_model(config, model_kind, device)
        if isinstance(state, dict) and 'model_state' in state:
            model.load_state_dict(state['model_state'])
        elif isinstance(state, dict) and 'net' in state:
            model.load_state_dict(state['net'])
        else:
            model.load_state_dict(state)
        return model, 'model.pt'

    ckpts = sorted(
        [f for f in os.listdir(model_dir) if f.startswith('checkpoint_epoch') and f.endswith('.pt')],
        key=lambda f: int(f.replace('checkpoint_epoch', '').replace('.pt', ''))
    )
    if ckpts:
        ckpt_name = ckpts[-1]
        state = torch.load(os.path.join(model_dir, ckpt_name), map_location='cpu', weights_only=False)
        config = state.get('model_config', {}) if isinstance(state, dict) else {}
        model = _build_gabor_like_model(config, model_kind, device)
        model.load_state_dict(state['model_state'])
        return model, ckpt_name
    raise FileNotFoundError(f"No Gabor checkpoint found in {model_dir}")


def _extract_train_time_from_state(state):
    if isinstance(state, dict):
        if 'train_time_sec' in state:
            return float(state['train_time_sec'])
        history = state.get('history')
        if isinstance(history, dict) and 'train_time_sec' in history:
            return float(history['train_time_sec'])
    return None


def read_training_time(model_dir):
    candidates = [os.path.join(model_dir, 'model.pt'), os.path.join(model_dir, 'history.pt')]
    ckpts = []
    if os.path.isdir(model_dir):
        ckpts = sorted(
            [f for f in os.listdir(model_dir) if f.startswith('checkpoint_epoch') and f.endswith('.pt')],
            key=lambda f: int(f.replace('checkpoint_epoch', '').replace('.pt', ''))
        )
        if ckpts:
            candidates.append(os.path.join(model_dir, ckpts[-1]))

    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            state = torch.load(path, map_location='cpu', weights_only=False)
        except Exception:
            continue
        seconds = _extract_train_time_from_state(state)
        if seconds is not None:
            return seconds
    return float('nan')


def format_train_time(seconds):
    if seconds is None or not np.isfinite(seconds):
        return 'N/A'
    return f"{seconds / 3600:.3f}h"


def get_parameter_info(model):
    """Return total parameter count and optional model-specific breakdown."""
    if hasattr(model, "count_parameters"):
        info = model.count_parameters()
        if isinstance(info, dict):
            total = int(info.get("total", sum(p.numel() for p in model.parameters())))
            return {
                "total_params": total,
                "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "param_breakdown": info,
            }

    return {
        "total_params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "param_breakdown": None,
    }


def format_params(n_params):
    if n_params is None:
        return 'N/A'
    n_params = int(n_params)
    if n_params >= 1_000_000:
        return f"{n_params / 1_000_000:.2f}M"
    if n_params >= 1_000:
        return f"{n_params / 1_000:.1f}K"
    return str(n_params)


def evaluate(model, data, device):
    """算全局和逐时刻的Rel L2"""
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

    # 逐时刻
    n_L = data['n_L']
    n_T = data['n_T']
    p_est_3d = p_est.reshape(n_L, n_L, n_T)
    p_ref_3d = data['p_ref']
    per_t = []
    for k in range(n_T):
        ref_k = p_ref_3d[:, :, k].flatten()
        est_k = p_est_3d[:, :, k].flatten()
        norm_ref = np.linalg.norm(ref_k)
        if norm_ref > 1e-12:
            per_t.append(np.linalg.norm(est_k - ref_k) / norm_ref)
        else:
            per_t.append(np.linalg.norm(est_k - ref_k))

    return {
        'rel_l2': rel_l2,
        'mae': mae,
        'per_timestep': per_t,
        'mean_per_t': np.mean(per_t),
        'max_per_t': np.max(per_t),
    }


def _pde_eval_indices(n_total, max_points, device):
    if max_points is None or max_points <= 0 or max_points >= n_total:
        return torch.arange(n_total, device=device)
    return torch.linspace(0, n_total - 1, steps=max_points, device=device).long()


def _ff_pinn_source_correction(model, r, c):
    training_config = getattr(model, '_training_config', None)
    if not isinstance(training_config, dict):
        return None
    if training_config.get('problem_form') != 'source':
        return None
    params = training_config.get('source_params') or {}
    if not params:
        return None

    f0 = float(params.get('f0', 10.0))
    alpha = float(params.get('alpha', 0.01))
    xs, ys = params.get('center', (0.5, 0.5))
    amp = float(training_config.get('source_amplitude', 1.0))
    x = r[:, 0:1]
    y = r[:, 1:2]
    t = r[:, 2:3]
    tau = np.pi * f0 * (t - 1.0 / f0)
    wavelet = amp * (1.0 - 2.0 * tau ** 2) * torch.exp(-(tau ** 2))
    spatial = torch.exp(-0.5 * (((x - float(xs)) / alpha) ** 2 + ((y - float(ys)) / alpha) ** 2))
    return wavelet * spatial / (c ** 2)


def _ff_pinn_flux_source(model, r):
    training_config = getattr(model, '_training_config', None)
    if not isinstance(training_config, dict):
        return None
    if training_config.get('problem_form') != 'source':
        return None
    params = training_config.get('source_params') or {}
    if not params:
        return None
    f0 = float(params.get('f0', 10.0))
    alpha = float(params.get('alpha', 0.01))
    xs, ys = params.get('center', (0.5, 0.5))
    amp = float(training_config.get('source_amplitude', 1.0))
    x = r[:, 0:1]
    y = r[:, 1:2]
    t = r[:, 2:3]
    tau = np.pi * f0 * (t - 1.0 / f0)
    wavelet = amp * (1.0 - 2.0 * tau ** 2) * torch.exp(-(tau ** 2))
    spatial = torch.exp(-0.5 * (((x - float(xs)) / alpha) ** 2 + ((y - float(ys)) / alpha) ** 2))
    return wavelet * spatial


def _ff_pinn_flux_residual(p, r, c, model):
    p_r = torch.autograd.grad(p, r, torch.ones_like(p), create_graph=True)[0]
    p_x = p_r[:, 0:1]
    p_y = p_r[:, 1:2]
    p_xx = torch.autograd.grad(p_r[:, 0], r, torch.ones_like(p_r[:, 0]), create_graph=True)[0][:, 0:1]
    p_yy = torch.autograd.grad(p_r[:, 1], r, torch.ones_like(p_r[:, 1]), create_graph=True)[0][:, 1:2]
    p_tt = torch.autograd.grad(p_r[:, 2], r, torch.ones_like(p_r[:, 2]), create_graph=True)[0][:, 2:3]
    q = c ** 2
    if q.requires_grad:
        q_r = torch.autograd.grad(q, r, torch.ones_like(q), create_graph=True, allow_unused=True)[0]
    else:
        q_r = None
    if q_r is None:
        q_x = torch.zeros_like(p_x)
        q_y = torch.zeros_like(p_y)
    else:
        q_x = q_r[:, 0:1]
        q_y = q_r[:, 1:2]
    res = q * (p_xx + p_yy) + q_x * p_x + q_y * p_y - p_tt
    source = _ff_pinn_flux_source(model, r)
    if source is not None:
        res = res + source
    return res


def evaluate_pde_residual(model, data, device, max_points=PDE_MAX_POINTS,
                          batch_size=PDE_BATCH_SIZE):
    """Evaluate p_xx + p_yy - p_tt / c^2 on deterministic domain samples."""
    if 'c_func' not in data:
        raise KeyError("Dataset must provide c_func to evaluate PDE residual.")

    n_total = data['r_ref'].shape[0]
    indices = _pde_eval_indices(n_total, max_points, device)
    n_eval = int(indices.numel())

    sum_sq = 0.0
    sum_abs = 0.0
    max_abs = 0.0

    was_training = model.training
    model.eval()

    for start in range(0, n_eval, batch_size):
        idx = indices[start:start + batch_size]
        r = data['r_ref'][idx].detach().clone().requires_grad_(True)
        c = data['c_func'](r[:, 0:1], r[:, 1:2], tensor=True)

        # cuDNN RNN kernels do not support the double backward needed for p_tt.
        with torch.backends.cudnn.flags(enabled=False):
            p = model(r)
            training_config = getattr(model, '_training_config', None)
            if isinstance(training_config, dict) and training_config.get('pde_form') == 'flux':
                res = _ff_pinn_flux_residual(p, r, c, model)
            else:
                res = pde_residual(p, r, c)
                source_correction = _ff_pinn_source_correction(model, r, c)
                if source_correction is not None:
                    res = res + source_correction

        res_det = res.detach()
        sum_sq += float(torch.sum(res_det ** 2).cpu())
        sum_abs += float(torch.sum(torch.abs(res_det)).cpu())
        max_abs = max(max_abs, float(torch.max(torch.abs(res_det)).cpu()))

        del r, c, p, res, res_det
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    if was_training:
        model.train()

    mse = sum_sq / max(1, n_eval)
    mae = sum_abs / max(1, n_eval)
    return {
        'pde_mse': mse,
        'pde_rmse': float(np.sqrt(mse)),
        'pde_mae': mae,
        'pde_max_abs': max_abs,
        'pde_n_eval': n_eval,
    }


def load_best_checkpoint(model, model_dir):
    """先找model.pt，没有就用最新的checkpoint"""
    model_path = os.path.join(model_dir, 'model.pt')
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location='cpu', weights_only=False)
        if isinstance(state, dict) and 'model_state' in state:
            model.load_state_dict(state['model_state'])
        else:
            model.load_state_dict(state)
        return 'model.pt'

    # 退而求其次：用最新checkpoint
    ckpts = sorted(
        [f for f in os.listdir(model_dir) if f.startswith('checkpoint_epoch') and f.endswith('.pt')],
        key=lambda f: int(f.replace('checkpoint_epoch', '').replace('.pt', ''))
    )
    if ckpts:
        ckpt = torch.load(os.path.join(model_dir, ckpts[-1]), map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        return ckpts[-1]
    raise FileNotFoundError(f"No checkpoint found in {model_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained PINN wavefield models.")
    parser.add_argument(
        "--dataset",
        choices=["all", "threelayer", "marmousi", "overthrust"],
        default="all",
        help="Dataset to evaluate.",
    )
    parser.add_argument(
        "--model",
        choices=[
            "all",
            "kan_st_pinn",
            "kan_bilstm_no_attention",
            "kan_attention_no_bilstm",
            "kan_pinn",
            "gabor_pinn",
            "gabor_enhanced_pinn",
            "ff_pinn",
            "raissi_pinn",
            "td_pinn_paper",
            "pinnsformer_paper",
        ],
        default="all",
        help="Model tag to evaluate.",
    )
    parser.add_argument(
        "--skip-pde",
        action="store_true",
        help="Skip second-derivative PDE residual metrics for quick Rel L2 checks.",
    )
    parser.add_argument(
        "--pde-max-points",
        type=int,
        default=PDE_MAX_POINTS,
        help="Deterministic sample size for PDE residual metrics.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    experiments = [
        ('kan_st_pinn', 'threelayer', 'kan_st'),
        ('kan_st_pinn', 'marmousi',   'kan_st'),
        ('kan_st_pinn', 'overthrust', 'kan_st'),
        ('kan_bilstm_no_attention', 'threelayer', 'kan_bilstm_no_attention'),
        ('kan_bilstm_no_attention', 'marmousi',   'kan_bilstm_no_attention'),
        ('kan_bilstm_no_attention', 'overthrust', 'kan_bilstm_no_attention'),
        ('kan_attention_no_bilstm', 'threelayer', 'kan_attention_no_bilstm'),
        ('kan_attention_no_bilstm', 'marmousi',   'kan_attention_no_bilstm'),
        ('kan_attention_no_bilstm', 'overthrust', 'kan_attention_no_bilstm'),
        ('kan_pinn',    'threelayer', 'kan'),
        ('kan_pinn',    'marmousi',   'kan'),
        ('kan_pinn',    'overthrust', 'kan'),
        ('gabor_pinn',  'threelayer', 'gabor'),
        ('gabor_pinn',  'marmousi',   'gabor'),
        ('gabor_pinn',  'overthrust', 'gabor'),
        ('gabor_enhanced_pinn', 'threelayer', 'gepinn'),
        ('gabor_enhanced_pinn', 'marmousi',   'gepinn'),
        ('gabor_enhanced_pinn', 'overthrust', 'gepinn'),
        ('ff_pinn', 'threelayer', 'ff_pinn'),
        ('ff_pinn', 'marmousi',   'ff_pinn'),
        ('ff_pinn', 'overthrust', 'ff_pinn'),
        ('ff_pinn_paper', 'threelayer', 'ff_pinn'),
        ('ff_pinn_paper', 'marmousi',   'ff_pinn'),
        ('ff_pinn_paper', 'overthrust', 'ff_pinn'),
        ('ff_pinn_initial_ablation', 'threelayer', 'ff_pinn'),
        ('ff_pinn_initial_ablation', 'marmousi',   'ff_pinn'),
        ('ff_pinn_initial_ablation', 'overthrust', 'ff_pinn'),
        ('ff_pinn_initial_tuned', 'threelayer', 'ff_pinn'),
        ('ff_pinn_initial_tuned', 'marmousi',   'ff_pinn'),
        ('ff_pinn_initial_tuned', 'overthrust', 'ff_pinn'),
        ('td_pinn_paper', 'threelayer', 'td_pinn'),
        ('td_pinn_paper', 'marmousi',   'td_pinn'),
        ('td_pinn_paper', 'overthrust', 'td_pinn'),
        ('pinnsformer_paper', 'threelayer', 'pinnsformer'),
        ('pinnsformer_paper', 'marmousi',   'pinnsformer'),
        ('pinnsformer_paper', 'overthrust', 'pinnsformer'),
        ('raissi_pinn_adaptive_nobc', 'threelayer', 'raissi'),
        ('raissi_pinn_adaptive_nobc', 'marmousi',   'raissi'),
        ('raissi_pinn_adaptive_nobc', 'overthrust', 'raissi'),
    ]
    if args.dataset != "all":
        experiments = [exp for exp in experiments if exp[1] == args.dataset]
    if args.model != "all":
        requested_model = "raissi_pinn_adaptive_nobc" if args.model == "raissi_pinn" else args.model
        experiments = [exp for exp in experiments if exp[0] == requested_model]

    # 旧ST-PINN基线 (threelayer only)
    old_baseline = {'threelayer': 0.113}

    results = {}
    # 缓存数据集，避免重复算FD
    dataset_cache = {}

    for tag, dataset, model_kind in experiments:
        model_dir = os.path.join(MODEL_DIR, dataset, tag)
        if not os.path.isdir(model_dir):
            print(f"[SKIP] {tag}/{dataset}: directory not found")
            continue

        print(f"{'='*60}")
        print(f"Evaluating: {tag} / {dataset}")
        print(f"{'='*60}")

        # 取数据集（有缓存就用缓存）
        if dataset not in dataset_cache:
            dataset_cache[dataset] = prepare_dataset(dataset, device)
        data = dataset_cache[dataset]

        # 建模型，对齐时间步
        if model_kind in ('gabor', 'gepinn'):
            model, ckpt_name = create_gabor_model(device, model_dir, model_kind=model_kind)
        elif model_kind == 'ff_pinn':
            model, ckpt_name = create_ff_pinn_model(device, model_dir)
        elif model_kind == 'td_pinn':
            model, ckpt_name = create_td_pinn_model(device, model_dir)
        elif model_kind == 'pinnsformer':
            model, ckpt_name = create_pinnsformer_model(device, model_dir)
        elif model_kind == 'raissi':
            model, ckpt_name = create_raissi_model(device, model_dir)
        elif model_kind == 'kan':
            model = create_model(True, device)
            ckpt_name = load_best_checkpoint(model, model_dir)
        elif model_kind in ('kan_st', 'kan_bilstm_no_attention', 'kan_attention_no_bilstm'):
            temporal_modes = {
                'kan_st': 'bilstm_attention',
                'kan_bilstm_no_attention': 'bilstm_no_attention',
                'kan_attention_no_bilstm': 'attention_no_bilstm',
            }
            dt, sl = align_temporal(data)
            model = create_model(
                False,
                device,
                dt_hist=dt,
                seq_len=sl,
                temporal_mode=temporal_modes[model_kind],
            )
            ckpt_name = load_best_checkpoint(model, model_dir)
        else:
            raise ValueError(f"Unsupported model kind: {model_kind}")

        # 加载权重
        print(f"  Loaded: {ckpt_name}")

        model.eval()
        param_info = get_parameter_info(model)
        metrics = evaluate(model, data, device)
        if args.skip_pde:
            pde_metrics = {
                'pde_mse': float('nan'),
                'pde_rmse': float('nan'),
                'pde_mae': float('nan'),
                'pde_max_abs': float('nan'),
                'pde_n_eval': 0,
            }
        else:
            pde_metrics = evaluate_pde_residual(
                model, data, device, max_points=args.pde_max_points
            )
        metrics.update(pde_metrics)
        metrics['train_time_sec'] = read_training_time(model_dir)
        metrics['params'] = param_info['total_params']
        metrics['trainable_params'] = param_info['trainable_params']
        metrics['param_breakdown'] = param_info['param_breakdown']

        key = f"{tag}/{dataset}"
        results[key] = metrics
        train_time_label = format_train_time(metrics['train_time_sec'])
        param_label = format_params(metrics['params'])
        print(f"  Global Rel L2  = {metrics['rel_l2']:.6f}")
        print(f"  MAE            = {metrics['mae']:.6e}")
        print(f"  Per-step mean  = {metrics['mean_per_t']:.6f}")
        print(f"  Per-step max   = {metrics['max_per_t']:.6f}")
        print(f"  PDE RMSE       = {metrics['pde_rmse']:.6e}  (n={metrics['pde_n_eval']})")
        print(f"  PDE MAE        = {metrics['pde_mae']:.6e}")
        print(f"  PDE max abs    = {metrics['pde_max_abs']:.6e}")
        print(f"  Parameters     = {param_label} ({metrics['params']:,})")
        print(f"  Train time     = {train_time_label}")

        if dataset in old_baseline:
            ratio = metrics['rel_l2'] / old_baseline[dataset]
            print(f"  vs Old ST-PINN = {ratio:.3f}x  ({'BETTER' if ratio < 1 else 'WORSE'})")
        print()

    # 汇总 table
    print(f"\n{'='*136}")
    print(f"{'Model':<25} {'Dataset':<12} {'Params':>10} {'Rel L2':>10} {'MAE':>12} "
          f"{'PDE RMSE':>12} {'PDE MAE':>12} {'PDE max':>12} {'Train':>10} {'vs Old':>10}")
    print(f"{'-'*136}")
    for (tag, dataset, model_kind) in experiments:
        key = f"{tag}/{dataset}"
        if key not in results:
            continue
        m = results[key]
        vs = ''
        if dataset in old_baseline:
            ratio = m['rel_l2'] / old_baseline[dataset]
            vs = f"{ratio:.3f}x"
        if model_kind == 'gepinn':
            name = 'Gabor-Enhanced-PINN'
        elif model_kind == 'gabor':
            name = 'Gabor-PINN'
        elif model_kind == 'ff_pinn':
            if tag == 'ff_pinn_paper':
                name = 'FF-PINN (paper)'
            elif tag in ('ff_pinn_initial_ablation', 'ff_pinn_initial_tuned'):
                name = 'FF-PINN (initial tuned)'
            else:
                name = 'FF-PINN'
        elif model_kind == 'raissi':
            name = 'raissi'
        elif model_kind == 'td_pinn':
            name = 'TD-PINN (paper)'
        elif model_kind == 'pinnsformer':
            name = 'PINNsFormer (paper)'
        elif model_kind == 'kan':
            name = 'Pure-KAN'
        elif model_kind == 'kan_bilstm_no_attention':
            name = 'KAN+BiLSTM w/o Attention'
        elif model_kind == 'kan_attention_no_bilstm':
            name = 'KAN+Attention w/o BiLSTM'
        else:
            name = 'KAN-ST-PINN'
        print(f"{name:<25} {dataset:<12} {format_params(m.get('params')):>10} "
              f"{m['rel_l2']:>10.6f} {m['mae']:>12.6e} "
              f"{m['pde_rmse']:>12.6e} {m['pde_mae']:>12.6e} "
              f"{m['pde_max_abs']:>12.6e} {format_train_time(m.get('train_time_sec')):>10} {vs:>10}")
    print(f"{'='*136}")


if __name__ == '__main__':
    main()
