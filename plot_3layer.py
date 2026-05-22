import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from compute_l2 import prepare_threelayer, align_temporal, create_model, load_best_checkpoint

def predict(model, r_ref, batch_size=4096):
    out = []
    with torch.no_grad():
        for i in range(0, r_ref.shape[0], batch_size):
            out.append(model(r_ref[i:i + batch_size]).cpu())
    return torch.cat(out, 0).numpy().flatten()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = prepare_threelayer(device)

    dt, sl = align_temporal(data)
    model = create_model(False, device, dt_hist=dt, seq_len=sl)
    ckpt = load_best_checkpoint(model, './trained/threelayer/kan_st_pinn')
    model.eval()

    nL, nT = data['n_L'], data['n_T']
    p_ref = data['p_ref']
    p_est = predict(model, data['r_ref']).reshape(nL, nL, nT)

    T_phys = 2.0
    t_phys = data['t'] / (data['t'][-1] + 1e-12) * T_phys

    time_indices = [0, nT // 4, nT // 2, 3 * nT // 4, nT - 1]
    p_max = float(np.max(np.abs(p_ref)))
    e_global = float(np.max(np.abs(p_est - p_ref)))

    fig = plt.figure(figsize=(4.0 * len(time_indices) + 1.2, 11))
    gs_main = GridSpec(3, 1, figure=fig, hspace=0.35, top=0.93, bottom=0.04, left=0.07, right=0.88)
    row_labels = ['Reference', 'KAN-ST-PINN', 'Error']

    rel_rows = []

    for row in range(3):
        gs_row = gs_main[row].subgridspec(1, len(time_indices) + 1, width_ratios=[1] * len(time_indices) + [0.05], wspace=0.22)
        for col, tidx in enumerate(time_indices):
            ax = fig.add_subplot(gs_row[0, col])
            if row == 0:
                img = p_ref[:, :, tidx]
                vmin, vmax = -p_max, p_max
                cmap = 'seismic'
            elif row == 1:
                img = p_est[:, :, tidx]
                vmin, vmax = -p_max, p_max
                cmap = 'seismic'
            else:
                img = p_est[:, :, tidx] - p_ref[:, :, tidx]
                vmin, vmax = -e_global, e_global
                cmap = 'seismic'

            im = ax.imshow(img, extent=[0, 5.0, 0, 5.0], origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')

            if row == 0:
                ax.set_title(f't = {t_phys[tidx]:.2f} s', fontsize=11)
            if row == 2:
                err = p_est[:, :, tidx] - p_ref[:, :, tidx]
                rl2 = np.linalg.norm(err) / (np.linalg.norm(p_ref[:, :, tidx]) + 1e-12)
                rel_rows.append((int(tidx), float(t_phys[tidx]), float(rl2)))
                ax.set_title(f'Rel L2 = {rl2:.4f}', fontsize=10)

            ax.set_xlabel('x (km)')
            if col == 0:
                ax.set_ylabel('y (km)')
            else:
                ax.set_yticklabels([])

        first_ax = fig.axes[-len(time_indices)]
        first_ax.text(-0.30, 0.5, row_labels[row], transform=first_ax.transAxes,
                      fontsize=13, va='center', ha='center', rotation=90, fontweight='bold')

        cax = fig.add_subplot(gs_row[0, len(time_indices)])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label('Pressure' if row < 2 else 'Error')

    fig.suptitle('Wavefield Comparison - Threelayer (R5, old style)', fontsize=16, fontweight='bold')

    out_dir = './figures/paper'
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, 'fig2_wavefield_threelayer_r5_oldstyle.png')
    out_txt = os.path.join(out_dir, 'fig2_wavefield_threelayer_r5_oldstyle_rel.txt')

    plt.savefig(out_png, dpi=220)
    plt.close()

    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write(f'checkpoint={ckpt}\n')
        for tidx, tval, rl2 in rel_rows:
            f.write(f'idx={tidx}, t={tval:.6f}s, rel_l2={rl2:.8f}\n')

    print(f'checkpoint={ckpt}')
    print(f'figure={out_png}')
    print(f'metrics={out_txt}')
    if rel_rows:
        tidx, tval, rl2 = rel_rows[-1]
        print(f't_last idx={tidx}, t={tval:.6f}s, rel_l2={rl2:.8f}')


if __name__ == '__main__':
    main()
