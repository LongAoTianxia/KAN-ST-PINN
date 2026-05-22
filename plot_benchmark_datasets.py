import os
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors

from PINNs_util.PINNs_fdiff import solver

plt.rcParams.update({
    "font.size": 17,
    "axes.labelsize": 18,
    "axes.titlesize": 21,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
    "figure.dpi": 160,
    "savefig.dpi": 260,
})


DATASET_SPECS = {
    "threelayer": {
        "title": "Three-layer",
        "L_km": 5.0,
        "T_s": 2.0,
        "c_max": 3.0,
        "source": (0.5, 0.5),
        "std": 0.05,
    },
    "marmousi": {
        "title": "Marmousi",
        "L_km": 3.0,
        "T_s": 1.0,
        "c_max": 4.5,
        "source": (0.5, 0.15),
        "std": 0.04,
    },
    "overthrust": {
        "title": "Overthrust",
        "L_km": 4.0,
        "T_s": 0.8,
        "c_max": 6.0,
        "source": (0.5, 0.12),
        "std": 0.035,
    },
}


def threelayer_velocity_unit(xx, yy):
    steepness = 40.0
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))
    return 0.5 + 0.25 * sig(steepness * (yy - 0.33)) + 0.25 * sig(steepness * (yy - 0.66))


def marmousi_velocity_unit(xx, yy):
    pi = np.pi
    c = 1.5 + 2.5 * yy
    c = c + 0.3 * np.sin(3 * pi * xx) * np.cos(2 * pi * yy)
    c = c + 0.2 * np.sin(5 * pi * xx + 0.5) * np.sin(4 * pi * yy)
    lens = -0.8 * np.exp(-((yy - 0.45) ** 2) / (2 * 0.06 ** 2))
    c = c + lens * (0.5 + 0.5 * np.cos(2 * pi * xx))
    return np.clip(c, 1.5, 4.5)


def overthrust_velocity_unit(xx, yy):
    pi = np.pi
    ones = np.ones_like(xx)
    zeros = np.zeros_like(xx)
    c = 2.0 + 3.5 * yy
    c = c + 0.4 * np.where(yy > 0.25, ones, zeros)
    c = c + 0.5 * np.where(yy > 0.55, ones, zeros)
    c = c + 0.3 * np.where(yy > 0.75, ones, zeros)
    fault_y = 0.3 + 0.35 * xx
    c = c + 0.6 * np.where(yy > fault_y, ones, zeros)
    c = c + 0.25 * np.sin(4 * pi * xx) * np.cos(3 * pi * yy)
    return np.clip(c, 2.0, 6.0)


def physical_velocity(name, xx_unit, yy_unit):
    if name == "threelayer":
        return threelayer_velocity_unit(xx_unit, yy_unit) * DATASET_SPECS[name]["c_max"]
    if name == "marmousi":
        return marmousi_velocity_unit(xx_unit, yy_unit)
    if name == "overthrust":
        return overthrust_velocity_unit(xx_unit, yy_unit)
    raise ValueError(f"Unknown dataset: {name}")


def normalized_velocity_func(name):
    spec = DATASET_SPECS[name]

    def c_func(xx, yy):
        return physical_velocity(name, xx, yy) / spec["c_max"]

    return c_func


def initial_pulse_func(name):
    source = DATASET_SPECS[name]["source"]
    std = DATASET_SPECS[name]["std"]

    def I_func(xx, yy):
        return np.exp(-0.5 * (((xx - source[0]) / std) ** 2 + ((yy - source[1]) / std) ** 2))

    return I_func


def solve_dataset(name, nx):
    spec = DATASET_SPECS[name]
    L = 1.0
    T = spec["T_s"] * spec["c_max"] / spec["L_km"]
    p_ref, xx, yy, t, dt = solver(
        initial_pulse_func(name),
        normalized_velocity_func(name),
        L, L, nx, nx, -1, T,
    )
    p_max = np.max(np.abs(p_ref))
    if p_max > 1e-12:
        p_ref = p_ref / p_max
    return p_ref, xx, yy, t


def gradient_magnitude(c_phys, L_km):
    n_x, n_y = c_phys.shape
    dx = L_km / max(1, n_x - 1)
    dy = L_km / max(1, n_y - 1)
    dc_dx, dc_dy = np.gradient(c_phys, dx, dy, edge_order=1)
    return np.sqrt(dc_dx ** 2 + dc_dy ** 2)


def extent_km(name):
    L = DATASET_SPECS[name]["L_km"]
    return [0.0, L, 0.0, L]


def plot_velocity_overview(datasets, nx, out_dir):
    fig, axes = plt.subplots(2, len(datasets), figsize=(6.4 * len(datasets), 10.4))
    if len(datasets) == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for col, name in enumerate(datasets):
        spec = DATASET_SPECS[name]
        grid = np.linspace(0.0, 1.0, nx)
        xx_u, yy_u = np.meshgrid(grid, grid, indexing="ij")
        c_phys = physical_velocity(name, xx_u, yy_u)
        grad = gradient_magnitude(c_phys, spec["L_km"])

        im0 = axes[0, col].imshow(c_phys.T, origin="lower", extent=extent_km(name),
                                  cmap="turbo", aspect="equal")
        axes[0, col].scatter([spec["source"][0] * spec["L_km"]],
                             [spec["source"][1] * spec["L_km"]],
                             s=60, c="white", edgecolors="black", linewidths=1.2)
        axes[0, col].set_title(f"{spec['title']} velocity")
        axes[0, col].set_xlabel("x (km)")
        axes[0, col].set_ylabel("y (km)")
        cbar0 = fig.colorbar(im0, ax=axes[0, col], label="km/s", fraction=0.046, pad=0.02)
        cbar0.ax.tick_params(labelsize=14)

        im1 = axes[1, col].imshow(grad.T, origin="lower", extent=extent_km(name),
                                  cmap="viridis", aspect="equal")
        axes[1, col].set_title(f"{spec['title']} velocity gradient")
        axes[1, col].set_xlabel("x (km)")
        axes[1, col].set_ylabel("y (km)")
        cbar1 = fig.colorbar(im1, ax=axes[1, col], label="1/s", fraction=0.046, pad=0.02)
        cbar1.ax.tick_params(labelsize=14)

    fig.tight_layout(w_pad=1.2, h_pad=1.0)
    path = os.path.join(out_dir, "benchmark_velocity_overview.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_pressure_volume(name, p_ref, t_norm, out_dir, stride=2):
    spec = DATASET_SPECS[name]
    n_x, n_y, n_t = p_ref.shape
    x = np.linspace(0.0, spec["L_km"], n_x)
    y = np.linspace(0.0, spec["L_km"], n_y)
    t_phys = t_norm / (t_norm[-1] + 1e-12) * spec["T_s"]

    x_ds = x[::stride]
    y_ds = y[::stride]
    t_ds = t_phys[::stride]
    p_ds = p_ref[::stride, ::stride, ::stride]

    sx = int(round(spec["source"][0] * (p_ds.shape[0] - 1)))
    sy = int(round(spec["source"][1] * (p_ds.shape[1] - 1)))

    energy = np.percentile(np.abs(p_ref), 99.0, axis=(0, 1))
    start_idx = max(1, int(0.12 * len(energy)))
    xy_idx_full = start_idx + int(np.argmax(energy[start_idx:]))
    xy_idx = int(np.argmin(np.abs(t_ds - t_phys[xy_idx_full])))

    top_vals = p_ds[:, :, xy_idx]
    xt_vals = p_ds[:, sy, :].T
    yt_vals = p_ds[sx, :, :].T

    plotted_values = np.concatenate([
        top_vals.ravel(),
        xt_vals.ravel(),
        yt_vals.ravel(),
    ])
    vmax = np.percentile(np.abs(plotted_values), 98.5)
    vmax = max(float(vmax), 1e-6)

    cmap = plt.get_cmap("seismic")
    norm = colors.SymLogNorm(
        linthresh=0.06 * vmax,
        linscale=0.75,
        vmin=-vmax,
        vmax=vmax,
        base=10,
    )

    fig = plt.figure(figsize=(9.4, 6.9))
    ax = fig.add_subplot(111, projection="3d")

    x_grid, y_grid = np.meshgrid(x_ds, y_ds, indexing="ij")
    top_z = np.full_like(x_grid, t_ds[xy_idx])
    ax.plot_surface(
        x_grid, y_grid, top_z,
        facecolors=cmap(norm(top_vals)),
        rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False, alpha=1.0,
    )

    x_grid_xt, t_grid_xt = np.meshgrid(x_ds, t_ds, indexing="xy")
    y_plane = np.full_like(x_grid_xt, y_ds[sy])
    ax.plot_surface(
        x_grid_xt, y_plane, t_grid_xt,
        facecolors=cmap(norm(xt_vals)),
        rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False, alpha=0.96,
    )

    y_grid_yt, t_grid_yt = np.meshgrid(y_ds, t_ds, indexing="xy")
    x_plane = np.full_like(y_grid_yt, x_ds[sx])
    ax.plot_surface(
        x_plane, y_grid_yt, t_grid_yt,
        facecolors=cmap(norm(yt_vals)),
        rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False, alpha=0.96,
    )

    ax.set_title(
        f"{spec['title']} pressure field $p(x,y,t)$\n"
        f"$x$-$y$ slice at $t={t_ds[xy_idx]:.2f}$ s",
        fontsize=22,
        pad=14,
    )
    ax.set_xlabel("x (km)", fontsize=20, labelpad=10)
    ax.set_ylabel("y (km)", fontsize=20, labelpad=10)
    ax.set_zlabel("time (s)", fontsize=20, labelpad=9)
    ax.set_xlim(0.0, spec["L_km"])
    ax.set_ylim(0.0, spec["L_km"])
    ax.set_zlim(0.0, spec["T_s"])
    ax.view_init(elev=24, azim=-55)
    ax.set_box_aspect((1.1, 1.1, 0.72))
    ax.grid(False)

    scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_map.set_array([])
    cbar_ticks = [-vmax, -0.1 * vmax, 0.0, 0.1 * vmax, vmax]
    cbar = fig.colorbar(scalar_map, ax=ax, shrink=0.72, pad=0.055, ticks=cbar_ticks)
    cbar.set_label("normalized pressure", fontsize=16)
    cbar.ax.tick_params(labelsize=13)

    path = os.path.join(out_dir, f"{name}_pressure_volume.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_wave_snapshots(name, p_ref, t_norm, out_dir):
    spec = DATASET_SPECS[name]
    n_t = p_ref.shape[2]
    indices = sorted(set([0, n_t // 4, n_t // 2, 3 * n_t // 4, n_t - 1]))
    t_phys = t_norm / (t_norm[-1] + 1e-12) * spec["T_s"]

    fig, axes = plt.subplots(1, len(indices), figsize=(3.2 * len(indices), 3.2))
    vmax = np.max(np.abs(p_ref))
    for ax, idx in zip(axes, indices):
        im = ax.imshow(p_ref[:, :, idx].T, origin="lower", extent=extent_km(name),
                       cmap="seismic", vmin=-vmax, vmax=vmax, aspect="equal")
        ax.set_title(f"t={t_phys[idx]:.2f}s")
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
    fig.suptitle(f"{spec['title']} normalized pressure snapshots", y=1.04)
    fig.colorbar(im, ax=axes.ravel().tolist(), label="normalized pressure", fraction=0.025)
    path = os.path.join(out_dir, f"{name}_pressure_snapshots.png")
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_space_time_slices(name, p_ref, t_norm, out_dir):
    spec = DATASET_SPECS[name]
    n_x, n_y, _ = p_ref.shape
    sx = int(round(spec["source"][0] * (n_x - 1)))
    sy = int(round(spec["source"][1] * (n_y - 1)))
    t_phys = t_norm / (t_norm[-1] + 1e-12) * spec["T_s"]
    L = spec["L_km"]
    x = np.linspace(0.0, L, n_x)
    y = np.linspace(0.0, L, n_y)

    xt = p_ref[:, sy, :].T
    yt = p_ref[sx, :, :].T
    vmax = max(np.max(np.abs(xt)), np.max(np.abs(yt)))

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    im0 = axes[0].imshow(xt, origin="lower", aspect="auto",
                         extent=[x[0], x[-1], t_phys[0], t_phys[-1]],
                         cmap="seismic", vmin=-vmax, vmax=vmax)
    axes[0].set_title(f"x-t slice at y={y[sy]:.2f} km")
    axes[0].set_xlabel("x (km)")
    axes[0].set_ylabel("t (s)")

    im1 = axes[1].imshow(yt, origin="lower", aspect="auto",
                         extent=[y[0], y[-1], t_phys[0], t_phys[-1]],
                         cmap="seismic", vmin=-vmax, vmax=vmax)
    axes[1].set_title(f"y-t slice at x={x[sx]:.2f} km")
    axes[1].set_xlabel("y (km)")
    axes[1].set_ylabel("t (s)")

    fig.suptitle(f"{spec['title']} pressure space-time slices", y=1.03)
    fig.colorbar(im1, ax=axes.ravel().tolist(), label="normalized pressure", fraction=0.035)
    path = os.path.join(out_dir, f"{name}_pressure_xt_yt_slices.png")
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_dataset_panel(name, p_ref, t_norm, out_dir):
    spec = DATASET_SPECS[name]
    n = p_ref.shape[0]
    grid = np.linspace(0.0, 1.0, n)
    xx_u, yy_u = np.meshgrid(grid, grid, indexing="ij")
    c_phys = physical_velocity(name, xx_u, yy_u)

    n_t = p_ref.shape[2]
    indices = [0, n_t // 3, 2 * n_t // 3, n_t - 1]
    t_phys = t_norm / (t_norm[-1] + 1e-12) * spec["T_s"]

    fig, axes = plt.subplots(1, 5, figsize=(16.0, 3.2))
    im0 = axes[0].imshow(c_phys.T, origin="lower", extent=extent_km(name),
                         cmap="turbo", aspect="equal")
    axes[0].scatter([spec["source"][0] * spec["L_km"]],
                    [spec["source"][1] * spec["L_km"]],
                    s=32, c="white", edgecolors="black", linewidths=0.8)
    axes[0].set_title("velocity")
    axes[0].set_xlabel("x (km)")
    axes[0].set_ylabel("y (km)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    vmax = np.max(np.abs(p_ref))
    for ax, idx in zip(axes[1:], indices):
        im = ax.imshow(p_ref[:, :, idx].T, origin="lower", extent=extent_km(name),
                       cmap="seismic", vmin=-vmax, vmax=vmax, aspect="equal")
        ax.set_title(f"p, t={t_phys[idx]:.2f}s")
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
    fig.suptitle(f"{spec['title']} benchmark", y=1.05)
    fig.colorbar(im, ax=axes[1:].ravel().tolist(), label="normalized pressure", fraction=0.025)
    path = os.path.join(out_dir, f"{name}_benchmark_panel.png")
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot benchmark dataset figures for the Numerical Experiments section.")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASET_SPECS.keys()),
                        default=list(DATASET_SPECS.keys()))
    parser.add_argument("--nx", type=int, default=60)
    parser.add_argument("--out-dir", default="./figures/benchmark_datasets")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    plot_velocity_overview(args.datasets, args.nx, args.out_dir)
    for name in args.datasets:
        print(f"Generating reference wavefield for {name}...")
        p_ref, xx, yy, t = solve_dataset(name, args.nx)
        plot_pressure_volume(name, p_ref, t, args.out_dir)


if __name__ == "__main__":
    main()
