"""诊断晚期时刻误差偏大的问题"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from compute_l2 import prepare_threelayer, align_temporal, create_model, load_best_checkpoint

def predict(model, r_ref, batch_size=4096):
    out = []
    with torch.no_grad():
        for i in range(0, r_ref.shape[0], batch_size):
            out.append(model(r_ref[i:i+batch_size]).cpu())
    return torch.cat(out, 0).numpy().flatten()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = prepare_threelayer(device)
    dt, sl = align_temporal(data)
    model = create_model(False, device, dt_hist=dt, seq_len=sl)
    load_best_checkpoint(model, './trained/threelayer/kan_st_pinn')
    model.eval()

    nL, nT = data['n_L'], data['n_T']
    p_ref = data['p_ref']
    p_est = predict(model, data['r_ref']).reshape(nL, nL, nT)

    T_phys = 2.0
    t_phys = data['t'] / (data['t'][-1] + 1e-12) * T_phys

    # 每个时刻的指标
    rel_l2 = []
    abs_err_mean = []
    abs_err_max = []
    ref_energy = []  # ||p_ref(:,:,k)||_2
    ref_max_amp = []

    for k in range(nT):
        ref_k = p_ref[:, :, k].flatten()
        est_k = p_est[:, :, k].flatten()
        err_k = est_k - ref_k

        norm_ref = np.linalg.norm(ref_k)
        ref_energy.append(norm_ref)
        ref_max_amp.append(float(np.max(np.abs(ref_k))))

        if norm_ref > 1e-12:
            rel_l2.append(np.linalg.norm(err_k) / norm_ref)
        else:
            rel_l2.append(np.linalg.norm(err_k))

        abs_err_mean.append(float(np.mean(np.abs(err_k))))
        abs_err_max.append(float(np.max(np.abs(err_k))))

    rel_l2 = np.array(rel_l2)
    abs_err_mean = np.array(abs_err_mean)
    abs_err_max = np.array(abs_err_max)
    ref_energy = np.array(ref_energy)
    ref_max_amp = np.array(ref_max_amp)

    # 打印关键指标
    print("=== Per-Timestep Diagnostic ===")
    print(f"{'idx':>4} {'t(s)':>7} {'Rel_L2':>10} {'AbsErrMean':>12} {'AbsErrMax':>12} {'RefEnergy':>12} {'RefMaxAmp':>12}")
    for k in range(0, nT, max(1, nT//20)):
        print(f"{k:4d} {t_phys[k]:7.3f} {rel_l2[k]:10.6f} {abs_err_mean[k]:12.6e} {abs_err_max[k]:12.6e} {ref_energy[k]:12.6e} {ref_max_amp[k]:12.6e}")
    # 打印最后一个时刻
    k = nT - 1
    print(f"{k:4d} {t_phys[k]:7.3f} {rel_l2[k]:10.6f} {abs_err_mean[k]:12.6e} {abs_err_max[k]:12.6e} {ref_energy[k]:12.6e} {ref_max_amp[k]:12.6e}")

    # 能量衰减比
    print(f"\nRef energy at t=0: {ref_energy[0]:.6e}")
    print(f"Ref energy at t=2: {ref_energy[-1]:.6e}")
    print(f"Energy drop ratio: {ref_energy[-1]/ref_energy[0]:.4f}x")
    print(f"Ref max amp at t=0: {ref_max_amp[0]:.6e}")
    print(f"Ref max amp at t=2: {ref_max_amp[-1]:.6e}")

    # t=2和t=1的绝对误差对比
    k_mid = nT // 2
    print(f"\nAbs err mean at t={t_phys[k_mid]:.2f}s: {abs_err_mean[k_mid]:.6e}")
    print(f"Abs err mean at t={t_phys[-1]:.2f}s: {abs_err_mean[-1]:.6e}")
    print(f"Abs err ratio (t2/t1): {abs_err_mean[-1]/abs_err_mean[k_mid]:.2f}x")

    # 画诊断图
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 0.9,
    })

    # Keep four panels because they jointly explain the numerator and
    # denominator effects in the per-timestep relative L2 metric.
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.2))

    ax = axes[0, 0]
    ax.plot(t_phys, rel_l2, color='#D55E00', linewidth=2.0)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Rel $L_2$')
    ax.set_title('Per-timestep relative error'); ax.grid(True, linestyle='--', alpha=0.35)

    ax = axes[0, 1]
    ax.plot(t_phys, ref_energy, color='#0072B2', linewidth=2.0, label=r'$\|p_{\mathrm{ref}}\|_2$')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Reference $L_2$ norm')
    ax.set_title('Reference wavefield energy'); ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(t_phys, abs_err_mean, color='#009E73', linewidth=2.0, label='Mean |error|')
    ax.plot(t_phys, abs_err_max, color='#CC79A7', linestyle='--', linewidth=1.6, label='Max |error|')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Absolute error')
    ax.set_title('Absolute error magnitude'); ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(t_phys, ref_max_amp, color='#0072B2', linewidth=2.0, label='Max $|p_{ref}|$')
    ax2 = ax.twinx()
    ax2.plot(t_phys, abs_err_max / (ref_max_amp + 1e-12), color='#D55E00', linewidth=1.8, label='Max error / Max ref')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Max $|p_{ref}|$')
    ax2.set_ylabel('Max error / Max ref')
    ax.set_title('Signal amplitude vs. relative error scale'); ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(loc='upper left'); ax2.legend(loc='upper right')

    for ax in axes.ravel():
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    plt.tight_layout(w_pad=2.0, h_pad=1.8)

    out_dir = './figures/paper'
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'diagnostic_r5_latetime.png')
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"\nDiagnostic figure: {path}")


if __name__ == '__main__':
    main()
