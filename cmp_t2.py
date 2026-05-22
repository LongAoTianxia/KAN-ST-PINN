import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from compute_l2 import (
    prepare_threelayer,
    align_temporal,
    create_model,
    load_best_checkpoint,
    L_KM,
    C0,
)


def predict_full(model, r_ref, batch_size=4096):
    out = []
    with torch.no_grad():
        for i in range(0, r_ref.shape[0], batch_size):
            out.append(model(r_ref[i:i + batch_size]).cpu())
    return torch.cat(out, dim=0).numpy().flatten()


def rel_l2_at_t(p_est_3d, p_ref_3d, k):
    est = p_est_3d[:, :, k].reshape(-1)
    ref = p_ref_3d[:, :, k].reshape(-1)
    return float(np.linalg.norm(est - ref) / (np.linalg.norm(ref) + 1e-12))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = prepare_threelayer(device)

    t_dimless = data['t']
    t_phys = t_dimless * (L_KM / C0)
    k = int(np.argmin(np.abs(t_phys - 2.0)))

    dt, sl = align_temporal(data)

    model_st = create_model(False, device, dt_hist=dt, seq_len=sl)
    model_k = create_model(True, device)

    dir_st = './trained/threelayer/kan_st_pinn'
    dir_k = './trained/threelayer/kan_pinn'

    ckpt_st = load_best_checkpoint(model_st, dir_st)
    ckpt_k = load_best_checkpoint(model_k, dir_k)

    model_st.eval()
    model_k.eval()

    p_ref = data['p_ref']
    nL = data['n_L']
    nT = data['n_T']

    pred_st = predict_full(model_st, data['r_ref']).reshape(nL, nL, nT)
    pred_k = predict_full(model_k, data['r_ref']).reshape(nL, nL, nT)

    rel_st = rel_l2_at_t(pred_st, p_ref, k)
    rel_k = rel_l2_at_t(pred_k, p_ref, k)

    ref_k = p_ref[:, :, k]
    st_k = pred_st[:, :, k]
    k_k = pred_k[:, :, k]
    err_st = np.abs(st_k - ref_k)
    err_k = np.abs(k_k - ref_k)

    vmax_ref = float(np.max(np.abs(ref_k)))
    vmax_err = float(max(np.max(err_st), np.max(err_k)))

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)

    im0 = axes[0, 0].imshow(ref_k, cmap='seismic', vmin=-vmax_ref, vmax=vmax_ref, origin='lower')
    axes[0, 0].set_title('Reference @ t=2.0s')
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(st_k, cmap='seismic', vmin=-vmax_ref, vmax=vmax_ref, origin='lower')
    axes[0, 1].set_title(f'KAN-ST Prediction\nRel L2={rel_st:.4f}')
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[0, 2].imshow(err_st, cmap='magma', vmin=0, vmax=vmax_err, origin='lower')
    axes[0, 2].set_title('KAN-ST |Error|')
    fig.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    im3 = axes[1, 0].imshow(ref_k, cmap='seismic', vmin=-vmax_ref, vmax=vmax_ref, origin='lower')
    axes[1, 0].set_title('Reference @ t=2.0s')
    fig.colorbar(im3, ax=axes[1, 0], fraction=0.046)

    im4 = axes[1, 1].imshow(k_k, cmap='seismic', vmin=-vmax_ref, vmax=vmax_ref, origin='lower')
    axes[1, 1].set_title(f'Pure-KAN Prediction\nRel L2={rel_k:.4f}')
    fig.colorbar(im4, ax=axes[1, 1], fraction=0.046)

    im5 = axes[1, 2].imshow(err_k, cmap='magma', vmin=0, vmax=vmax_err, origin='lower')
    axes[1, 2].set_title('Pure-KAN |Error|')
    fig.colorbar(im5, ax=axes[1, 2], fraction=0.046)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    out_dir = './figures/paper'
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, 'fig_t2_compare_threelayer_r5.png')
    fig.savefig(out_png, dpi=220)

    out_txt = os.path.join(out_dir, 't2_rel_l2_threelayer_r5.txt')
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write(f't_phys={t_phys[k]:.6f}s\n')
        f.write(f't_index={k}\n')
        f.write(f'kan_st_checkpoint={ckpt_st}\n')
        f.write(f'pure_kan_checkpoint={ckpt_k}\n')
        f.write(f'kan_st_rel_l2_t2={rel_st:.8f}\n')
        f.write(f'pure_kan_rel_l2_t2={rel_k:.8f}\n')

    print(f't_nearest_phys={t_phys[k]:.6f}s (index={k})')
    print(f'KAN-ST  t=2.0s Rel L2 = {rel_st:.8f}')
    print(f'PureKAN t=2.0s Rel L2 = {rel_k:.8f}')
    print(f'figure: {out_png}')
    print(f'metrics: {out_txt}')


if __name__ == '__main__':
    main()
