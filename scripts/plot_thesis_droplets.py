#!/usr/bin/env python3
"""Generate droplet morphology snapshots for thesis figures.

Reads .npz snapshots from results/thesis_snapshots/ which now include
xz_slice, xy_slice, xz_solid, and grid dimensions.

Outputs to results/thesis_figures/:
  fig7_substrate_evolution.png   — 4×10 panel: 4 substrates × 10 moments
  fig8_ridge_zoom.png            — Ridge R*=1.0 key moments with Dx/Dy annotation
  fig9_peak_comparison.png       — 4 substrates at peak spreading + interface contour
  fig10_ridge_contour.png        — Ridge contour evolution with Dx annotation
  fig11_we_comparison.png        — We effect on ridge R*=1.0 droplet shape
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import sys, os

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 14,
    'legend.fontsize': 10, 'figure.dpi': 300,
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'savefig.bbox': 'tight',
})

ROOT = Path(__file__).resolve().parent.parent
snap_dir = ROOT / 'results' / 'thesis_snapshots'
fig_dir = ROOT / 'results' / 'thesis_figures'
fig_dir.mkdir(exist_ok=True, parents=True)

# Custom colormap: white (gas) -> blue (liquid)
colors_phase = ['#FFFFFF', '#D6EAF8', '#85C1E9', '#3498DB', '#2471A3', '#1A5276']
cmap_phase = LinearSegmentedColormap.from_list('phase', colors_phase, N=256)

solid_color = '#7F8C8D'

# Global physical parameters (must match simulation)
D0 = 45.0
R_drop = D0 / 2.0


def load_snapshot(label, step):
    """Load a snapshot .npz file, returns dict with full 3D + 2D slice data."""
    snap_file = snap_dir / f'{label}_step{step:05d}.npz'
    if not snap_file.exists():
        return None
    data = np.load(snap_file, allow_pickle=True)
    result = {
        'xz': data['xz_slice'],
        'yz': data['yz_slice'] if 'yz_slice' in data else None,
        'solid': data['xz_solid'] if 'xz_solid' in data else None,
        'nx': int(data['nx']) if 'nx' in data else data['xz_slice'].shape[0],
        'nz': int(data['nz']) if 'nz' in data else data['xz_slice'].shape[1],
    }
    # Full 3D fields (if available)
    if 'phi_3d' in data:
        result['phi_3d'] = data['phi_3d']
        result['u_3d'] = data['u_3d'] if 'u_3d' in data else None
        result['solid_3d'] = data['solid_3d'] if 'solid_3d' in data else None
        result['ny'] = int(data['ny']) if 'ny' in data else result['phi_3d'].shape[1]
    return result


def plot_droplet_panel(ax, xz_slice, solid_xz, title='', show_cbar=False,
                       annotate_k=None):
    """Plot a single droplet xz cross-section with solid overlay."""
    nx, nz = xz_slice.shape

    # Plot phase field
    im = ax.imshow(xz_slice.T, origin='lower', cmap=cmap_phase,
                   vmin=0, vmax=1, aspect='equal',
                   extent=[0, nx, 0, nz])

    # Overlay solid
    if solid_xz is not None:
        masked = np.ma.masked_where(~solid_xz.T,
                                    np.ones_like(solid_xz.T, dtype=float))
        ax.imshow(masked, origin='lower',
                  cmap=LinearSegmentedColormap.from_list(
                      'solid', [solid_color, solid_color], N=2),
                  vmin=0, vmax=1, aspect='equal', alpha=0.9,
                  extent=[0, nx, 0, nz])

    if title:
        ax.set_title(title, fontsize=11)

    if annotate_k is not None:
        ax.annotate(f'k={annotate_k:.3f}', xy=(0.05, 0.95),
                    xycoords='axes fraction', fontsize=10,
                    color='#E74C3C', fontweight='bold', va='top',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white',
                              ec='none', alpha=0.7))

    ax.set_xticks([])
    ax.set_yticks([])
    return im


def compute_k(xz_slice, solid_xz):
    """Compute asymmetry factor k = Dx/Dy from xz slice."""
    interface = (xz_slice > 0.5)
    if solid_xz is not None:
        interface = interface & ~solid_xz
    if not interface.any():
        return 0.0, 0, 0
    coords = np.argwhere(interface)
    Dx = coords[:, 0].max() - coords[:, 0].min() + 1
    # Note: from xz slice, Dy is not directly available;
    # we compute Dx from the slice. For full k we need xy too.
    return float(Dx), Dx, 0


# =====================================================================
# Figure 7: Substrate comparison — droplet evolution (4 × 10 panel)
# =====================================================================
print("Generating fig7: substrate evolution comparison...")

substrates = [
    ('s2_flat_flat',      'Flat'),
    ('s2_ridge_R1.0',     'Ridge ($R^*$=1.0)'),
    ('s2_convex_R1.0',    'Convex ($R^*$=1.0)'),
    ('s2_concave_R1.0',   'Concave ($R^*$=1.0)'),
]

# 10 snapshot moments (matching simulation snap_steps fractions)
N = 3000
snap_fracs = [0.03, 0.07, 0.13, 0.20, 0.33, 0.47, 0.60, 0.73, 0.87, 1.0]
snap_steps = [int(f * N) for f in snap_fracs]
# Time labels (normalized t* = step * |U0| / D0)
time_labels = [f't*={s * 0.05 / 45:.2f}' for s in snap_steps]

fig, axes = plt.subplots(len(substrates), len(snap_steps),
                         figsize=(3 * len(snap_steps), 3.5 * len(substrates)),
                         gridspec_kw={'hspace': 0.15, 'wspace': 0.05})

for row, (prefix, label) in enumerate(substrates):
    for col, step in enumerate(snap_steps):
        ax = axes[row, col]
        snap = load_snapshot(prefix, step)

        if snap is not None:
            im7 = plot_droplet_panel(ax, snap['xz'], snap['solid'])
        else:
            ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                    ha='center', va='center', color='gray')
            ax.set_xticks([])
            ax.set_yticks([])

        if row == 0:
            ax.set_title(f't*={step * 0.05 / 45:.2f}', fontsize=10)
        if col == 0:
            ax.set_ylabel(label, fontsize=11, fontweight='bold',
                         rotation=0, labelpad=90, ha='right', va='center')

# Colorbar
fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.93, 0.15, 0.012, 0.7])
cbar = fig.colorbar(im7, cax=cbar_ax)
cbar.set_label('Phase field $\\phi$', fontsize=12)
cbar.set_ticks([0, 0.5, 1.0])
cbar.set_ticklabels(['Gas', 'Interface', 'Liquid'])

fig.suptitle('Droplet Impact Evolution on Different Substrates\n'
             '(We=7.9, $\\theta$=162°, $D_0$=45, VP)',
             fontsize=14, fontweight='bold', y=1.01)
fig.savefig(fig_dir / 'fig7_substrate_evolution.png', dpi=300)
plt.close()
print('  Saved fig7_substrate_evolution.png')


# =====================================================================
# Figure 8: Ridge R*=1.0 — 6 key moments with Dx/Dy annotation
# =====================================================================
print("Generating fig8: ridge zoomed evolution...")

key_steps = [int(0.07 * N), int(0.20 * N), int(0.33 * N),
             int(0.47 * N), int(0.73 * N), int(1.0 * N)]
fig, axes = plt.subplots(1, 6, figsize=(24, 5))

for col, step in enumerate(key_steps):
    ax = axes[col]
    snap = load_snapshot('s1_ridge_R1.0', step)

    if snap is not None:
        xz = snap['xz']
        # Compute Dx from interface
        interface = xz > 0.5
        if snap['solid'] is not None:
            interface = interface & ~snap['solid']
        if interface.any():
            coords = np.argwhere(interface)
            Dx = coords[:, 0].max() - coords[:, 0].min() + 1
            k_text = f'$D_x$={Dx}'
        else:
            k_text = ''

        im8 = plot_droplet_panel(ax, xz, snap['solid'] if snap else None,
                                 f't*={step * 0.05 / 45:.2f}')
        if k_text:
            ax.annotate(k_text, xy=(0.05, 0.95), xycoords='axes fraction',
                       fontsize=11, color='#E74C3C', fontweight='bold', va='top',
                       bbox=dict(boxstyle='round,pad=0.2', fc='white',
                                 ec='none', alpha=0.7))
        # Add interface contour
        ax.contour(xz.T, levels=[0.5], colors='#E74C3C', linewidths=1.2,
                   origin='lower', extent=[0, xz.shape[0], 0, xz.shape[1]])
    else:
        ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes, ha='center', va='center')

fig.suptitle('Ridge Substrate ($R^*$=1.0) — Droplet Evolution with Interface Contour\n'
             'We=7.9, $\\theta$=162°',
             fontsize=13, fontweight='bold')
fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.93, 0.15, 0.012, 0.7])
fig.colorbar(im8, cax=cbar_ax, label='$\\phi$')
fig.savefig(fig_dir / 'fig8_ridge_zoom.png', dpi=300)
plt.close()
print('  Saved fig8_ridge_zoom.png')


# =====================================================================
# Figure 9: 4 substrates at peak spreading — comparison with contour
# =====================================================================
print("Generating fig9: peak spreading comparison...")

peak_cases = [
    ('s2_flat_flat',    'Flat'),
    ('s2_ridge_R1.0',   'Ridge ($R^*$=1.0)'),
    ('s2_convex_R1.0',  'Convex ($R^*$=1.0)'),
    ('s2_concave_R1.0', 'Concave ($R^*$=1.0)'),
]

# Try step 1000 (t*≈1.1, near peak spreading) and nearby
peak_steps = [int(0.33 * N)]  # step 990

fig, axes = plt.subplots(1, 4, figsize=(16, 5))
for col, (prefix, label) in enumerate(peak_cases):
    ax = axes[col]
    snap = load_snapshot(prefix, peak_steps[0])

    if snap is not None:
        xz = snap['xz']
        im9 = plot_droplet_panel(ax, xz, snap['solid'], label)
        # Interface contour
        ax.contour(xz.T, levels=[0.5], colors='#E74C3C', linewidths=1.5,
                   origin='lower', extent=[0, xz.shape[0], 0, xz.shape[1]])
    else:
        ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes, ha='center', va='center')

fig.suptitle('Droplet Shape at Peak Spreading (t*$\\approx$1.1)\n'
             'We=7.9, $\\theta$=162°, Red: $\\phi$=0.5 interface',
             fontsize=13, fontweight='bold')
fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.93, 0.15, 0.012, 0.7])
fig.colorbar(im9, cax=cbar_ax, label='$\\phi$')
fig.savefig(fig_dir / 'fig9_peak_comparison.png', dpi=300)
plt.close()
print('  Saved fig9_peak_comparison.png')


# =====================================================================
# Figure 10: Ridge contour evolution (5 moments + Dx annotation)
# =====================================================================
print("Generating fig10: ridge contour evolution...")

contour_steps = [int(f * N) for f in [0.07, 0.20, 0.33, 0.60, 1.0]]

fig, axes = plt.subplots(1, len(contour_steps), figsize=(4 * len(contour_steps), 5))
if len(contour_steps) == 1:
    axes = [axes]

for ax, step in zip(axes, contour_steps):
    snap = load_snapshot('s1_ridge_R1.0', step)
    if snap is not None:
        xz = snap['xz']
        interface = xz > 0.5
        if snap['solid'] is not None:
            interface = interface & ~snap['solid']
        if interface.any():
            coords = np.argwhere(interface)
            Dx = coords[:, 0].max() - coords[:, 0].min() + 1
            ax.annotate(f'$D_x$={Dx}', xy=(0.05, 0.95), xycoords='axes fraction',
                       fontsize=11, color='#E74C3C', fontweight='bold', va='top',
                       bbox=dict(boxstyle='round,pad=0.2', fc='white',
                                 ec='none', alpha=0.7))

        im10 = ax.imshow(xz.T, origin='lower', cmap=cmap_phase, vmin=0, vmax=1,
                         aspect='equal', extent=[0, xz.shape[0], 0, xz.shape[1]])

        if snap['solid'] is not None:
            masked = np.ma.masked_where(~snap['solid'].T,
                                        np.ones_like(snap['solid'].T, dtype=float))
            ax.imshow(masked, origin='lower',
                      cmap=LinearSegmentedColormap.from_list(
                          's', [solid_color, solid_color], N=2),
                      vmin=0, vmax=1, aspect='equal', alpha=0.9,
                      extent=[0, xz.shape[0], 0, xz.shape[1]])

        try:
            ax.contour(xz.T, levels=[0.5], colors='#E74C3C', linewidths=1.5,
                       origin='lower', extent=[0, xz.shape[0], 0, xz.shape[1]])
        except ValueError:
            pass
    else:
        ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes, ha='center', va='center')

    ax.set_title(f't*={step * 0.05 / 45:.2f}', fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle('Ridge Substrate ($R^*$=1.0) — Contour Evolution\n'
             'We=7.9, $\\theta$=162°',
             fontsize=13, fontweight='bold')
fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.93, 0.15, 0.012, 0.7])
fig.colorbar(im10, cax=cbar_ax, label='$\\phi$')
fig.savefig(fig_dir / 'fig10_ridge_contour.png', dpi=300)
plt.close()
print('  Saved fig10_ridge_contour.png')


# =====================================================================
# Figure 11: R* effect — ridge at R*=0.5, 1.0, 2.0, 2.76 at peak
# =====================================================================
print("Generating fig11: R* effect comparison...")

rstar_cases = [
    ('s1_ridge_R0.5',  '$R^*$=0.5'),
    ('s1_ridge_R1.0',  '$R^*$=1.0'),
    ('s1_ridge_R2.0',  '$R^*$=2.0'),
    ('s1_ridge_R2.76', '$R^*$=2.76'),
    ('s1_flat',        'Flat ($R^* \\rightarrow \\infty$)'),
]

peak_step = int(0.33 * N)

fig, axes = plt.subplots(1, len(rstar_cases), figsize=(4 * len(rstar_cases), 5))
if len(rstar_cases) == 1:
    axes = [axes]

for ax, (prefix, label) in zip(axes, rstar_cases):
    snap = load_snapshot(prefix, peak_step)
    if snap is not None:
        xz = snap['xz']
        im11 = plot_droplet_panel(ax, xz, snap['solid'], label)
        ax.contour(xz.T, levels=[0.5], colors='#E74C3C', linewidths=1.5,
                   origin='lower', extent=[0, xz.shape[0], 0, xz.shape[1]])
    else:
        ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes, ha='center', va='center')

fig.suptitle('Curvature Ratio Effect at Peak Spreading (t*$\\approx$1.1)\n'
             'We=7.9, $\\theta$=162°',
             fontsize=13, fontweight='bold')
fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.93, 0.15, 0.012, 0.7])
fig.colorbar(im11, cax=cbar_ax, label='$\\phi$')
fig.savefig(fig_dir / 'fig11_rstar_comparison.png', dpi=300)
plt.close()
print('  Saved fig11_rstar_comparison.png')


print(f'\nAll droplet morphology figures saved to {fig_dir}/')


# =====================================================================
# Figure 12: Top-view (xy) spreading asymmetry — 4 substrates × 4 moments
# Shows the elliptical footprint from above, directly revealing Dx vs Dy
# =====================================================================
print("Generating fig12: top-view xy spreading asymmetry...")

xy_cases = [
    ('s2_flat_flat',    'Flat'),
    ('s2_ridge_R1.0',   'Ridge ($R^*$=1.0)'),
    ('s2_convex_R1.0',  'Convex ($R^*$=1.0)'),
    ('s2_concave_R1.0', 'Concave ($R^*$=1.0)'),
]

# 4 moments: impact, peak, retract, final
xy_steps = [int(f * N) for f in [0.13, 0.33, 0.60, 1.0]]

fig, axes = plt.subplots(len(xy_cases), len(xy_steps),
                         figsize=(4 * len(xy_steps), 4 * len(xy_cases)),
                         gridspec_kw={'hspace': 0.15, 'wspace': 0.05})

for row, (prefix, label) in enumerate(xy_cases):
    for col, step in enumerate(xy_steps):
        ax = axes[row, col]
        snap = load_snapshot(prefix, step)

        if snap is not None:
            # Prefer full 3D yz slice at mid-x, fallback to saved yz_slice
            yz = None
            if snap.get('phi_3d') is not None:
                yz = snap['phi_3d'][snap['nx'] // 2, :, :]
            elif snap.get('yz') is not None:
                yz = snap['yz']

            if yz is not None:
                im12 = ax.imshow(yz.T, origin='lower', cmap=cmap_phase,
                                vmin=0, vmax=1, aspect='equal',
                                extent=[0, yz.shape[0], 0, yz.shape[1]])
            # Interface contour
                try:
                    ax.contour(yz.T, levels=[0.5], colors='#E74C3C', linewidths=1.2,
                              origin='lower', extent=[0, yz.shape[0], 0, yz.shape[1]])
            except ValueError:
                pass
        else:
            ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                    ha='center', va='center', color='gray')

        if row == 0:
            ax.set_title(f't*={step * 0.05 / 45:.2f}', fontsize=10)
        if col == 0:
            ax.set_ylabel(label, fontsize=11, fontweight='bold',
                         rotation=0, labelpad=90, ha='right', va='center')
        ax.set_xticks([])
        ax.set_yticks([])

fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.93, 0.15, 0.012, 0.7])
fig.colorbar(im12, cax=cbar_ax, label='$\\phi$')
fig.suptitle('Front View (yz Plane) — Droplet Shape Evolution\n'
             'We=7.9, $\\theta$=162°, Red: $\\phi$=0.5 interface',
             fontsize=13, fontweight='bold', y=1.01)
fig.savefig(fig_dir / 'fig12_frontview_yz.png', dpi=300)
plt.close()
print('  Saved fig12_frontview_yz.png')


# =====================================================================
# Figure 13: Triple-view panels (xz + xy + combined) for 4 substrates
# Side view + Front view side by side for each substrate at peak spreading
# =====================================================================
print("Generating fig13: triple-view comparison...")

peak_step = int(0.33 * N)

fig, axes = plt.subplots(2, 4, figsize=(20, 10),
                         gridspec_kw={'hspace': 0.25, 'wspace': 0.15})

for col, (prefix, label) in enumerate(xy_cases):
    snap = load_snapshot(prefix, peak_step)

    # Top row: xz side view
    ax_xz = axes[0, col]
    if snap is not None:
        xz = snap['xz']
        im13xz = plot_droplet_panel(ax_xz, xz, snap['solid'], '')
        ax_xz.contour(xz.T, levels=[0.5], colors='#E74C3C', linewidths=1.5,
                      origin='lower', extent=[0, xz.shape[0], 0, xz.shape[1]])
        # Compute Dx from xz
        interface = xz > 0.5
        if snap['solid'] is not None:
            interface &= ~snap['solid']
        if interface.any():
            c = np.argwhere(interface)
            Dx = c[:, 0].max() - c[:, 0].min() + 1
            ax_xz.annotate(f'$D_x$={Dx}', xy=(0.05, 0.95),
                          xycoords='axes fraction', fontsize=10,
                          color='#E74C3C', fontweight='bold', va='top',
                          bbox=dict(boxstyle='round,pad=0.2', fc='white',
                                    ec='none', alpha=0.7))
    else:
        ax_xz.text(0.5, 0.5, 'N/A', transform=ax_xz.transAxes,
                  ha='center', va='center')

    ax_xz.set_title(label, fontsize=11, fontweight='bold')
    ax_xz.set_xlabel('x', fontsize=10)
    ax_xz.set_ylabel('z', fontsize=10)

    # Bottom row: yz front view (prefer full 3D mid-i slice, fallback to saved yz)
    ax_yz = axes[1, col]
    yz = None
    if snap is not None:
        if snap.get('phi_3d') is not None:
            # Full 3D available: extract yz at x = nx//2
            yz = snap['phi_3d'][snap['nx'] // 2, :, :]
        elif snap.get('yz') is not None:
            yz = snap['yz']

    if yz is not None:
        im13yz = ax_yz.imshow(yz.T, origin='lower', cmap=cmap_phase,
                              vmin=0, vmax=1, aspect='equal',
                              extent=[0, yz.shape[0], 0, yz.shape[1]])
        ax_yz.contour(yz.T, levels=[0.5], colors='#E74C3C', linewidths=1.5,
                     origin='lower', extent=[0, yz.shape[0], 0, yz.shape[1]])
        interface_yz = yz > 0.5
        if interface_yz.any():
            c = np.argwhere(interface_yz)
            Dy = c[:, 0].max() - c[:, 0].min() + 1
            ax_yz.annotate(f'$D_y$={Dy}', xy=(0.05, 0.95),
                          xycoords='axes fraction', fontsize=10,
                          color='#2980B9', fontweight='bold', va='top',
                          bbox=dict(boxstyle='round,pad=0.2', fc='white',
                                    ec='none', alpha=0.7))
    else:
        ax_yz.text(0.5, 0.5, 'N/A', transform=ax_yz.transAxes,
                  ha='center', va='center')

    ax_yz.set_xlabel('y', fontsize=10)
    ax_yz.set_ylabel('z', fontsize=10)

# Row labels
axes[0, 0].set_ylabel('Side view (xz)', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Front view (yz)', fontsize=12, fontweight='bold')

fig.suptitle('Droplet Shape at Peak Spreading — Side + Front Views\n'
             'We=7.9, $\\theta$=162°, t*$\\approx$1.1',
             fontsize=14, fontweight='bold', y=1.02)
fig.savefig(fig_dir / 'fig13_dualview_comparison.png', dpi=300)
plt.close()
print('  Saved fig13_dualview_comparison.png')


# =====================================================================
# Figure 14: 3D oblique view using isometric projection of interface
# Uses xz + yz slices to reconstruct approximate 3D droplet shape
# =====================================================================
print("Generating fig14: 3D oblique views...")

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

fig = plt.figure(figsize=(20, 5))

for col, (prefix, label) in enumerate(xy_cases):
    snap = load_snapshot(prefix, peak_step)
    ax = fig.add_subplot(1, 4, col + 1, projection='3d')

    if snap is not None:
        nx_s = snap['nx']
        nz_s = snap['nz']

        if snap.get('phi_3d') is not None:
            # Full 3D data available — use directly with downsampling
            phi_3d = snap['phi_3d']
            solid_3d = snap.get('solid_3d')
            ny_s = phi_3d.shape[1]

            # Downsample for 3D rendering (every 3rd point)
            ds = 3
            phi_ds = phi_3d[::ds, ::ds, ::ds]
            solid_ds = solid_3d[::ds, ::ds, ::ds] if solid_3d is not None else None

            voxels = phi_ds > 0.5
            if voxels.any():
                colors = np.empty(voxels.shape + (4,))
                colors[voxels] = [0.2, 0.5, 0.85, 0.4]

                if solid_ds is not None:
                    sub_vox = solid_ds & ~voxels
                    colors[sub_vox] = [0.5, 0.5, 0.5, 0.6]

                ax.voxels(voxels | (solid_ds if solid_ds is not None else np.zeros_like(voxels)),
                          facecolors=colors, edgecolor='none', shade=True)
        else:
            # Fallback: reconstruct from xz + yz slices
            xz = snap['xz']
            yz_data = snap.get('yz')

            if yz_data is not None:
                ny_s = yz_data.shape[0]
                xz_3d = np.broadcast_to(xz[:, np.newaxis, :], (nx_s, ny_s, nz_s))
                yz_3d = np.broadcast_to(yz_data[np.newaxis, :, :], (nx_s, ny_s, nz_s))
                phi_3d = np.sqrt(np.maximum(xz_3d * yz_3d, 0))

                ds = 3
                phi_ds = phi_3d[::ds, ::ds, ::ds]
                voxels = phi_ds > 0.5
                if voxels.any():
                    colors = np.empty(voxels.shape + (4,))
                    colors[voxels] = [0.2, 0.5, 0.85, 0.4]
                    ax.voxels(voxels, facecolors=colors, edgecolor='none', shade=True)

        ax.set_xlabel('x', fontsize=9)
        ax.set_ylabel('y', fontsize=9)
        ax.set_zlabel('z', fontsize=9)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.view_init(elev=25, azim=-60)
        ax.set_box_aspect([1, 1, 1])

fig.suptitle('3D Droplet Shape at Peak Spreading (Approximate)\n'
             'We=7.9, $\\theta$=162°, t*$\\approx$1.1',
             fontsize=13, fontweight='bold', y=1.05)
fig.savefig(fig_dir / 'fig14_3d_oblique.png', dpi=300)
plt.close()
print('  Saved fig14_3d_oblique.png')


print(f'\nAll {len(list(fig_dir.glob("fig*.png")))} droplet figures saved to {fig_dir}/')
