#!/usr/bin/env python3
"""Publication-quality droplet snapshot figure from 150³ VP data.

Generates:
1. Time-sequence x-z cross-sections (perpendicular to ridge)
2. Time-sequence x-y top-views (at surface level)
3. VP smooth boundary detail (inset)
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'mathtext.fontset': 'cm',
    'axes.labelsize': 11,
    'axes.titlesize': 11,
})

save_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(save_dir, 'snapshots_150_vp.npz')

if not os.path.exists(data_path):
    print(f"Data not found: {data_path}")
    print("Run run_snapshots_150.py first!")
    exit(1)

data = np.load(data_path, allow_pickle=True)
snapshots = data['snapshots']
steps = data['steps']
solid = data['solid']
fraction = data['fraction'] if 'fraction' in data else None
nx, ny, nz = int(data['nx']), int(data['ny']), int(data['nz'])
ridge_top = int(data['ridge_top'])
cx, cy, cz = float(data['cx']), float(data['cy']), float(data['cz'])
R_drop = float(data['R_drop'])
D0 = float(data['D0'])
amp = float(data['amp'])
n_base = int(data['n_base'])

print(f"Loaded: {len(snapshots)} snapshots, grid={nx}x{ny}x{nz}")

# Custom colormap: deep blue → white → deep red
colors = [
    (0.05, 0.15, 0.50),   # dark blue (gas)
    (0.20, 0.45, 0.75),   # medium blue
    (0.95, 0.95, 0.95),   # near-white (interface)
    (0.90, 0.25, 0.15),   # medium red
    (0.55, 0.05, 0.05),   # dark red (liquid)
]
cmap = LinearSegmentedColormap.from_list('phase', colors, N=256)

# ===== Figure 1: Time-sequence x-z cross-sections =====
# Select 6 key moments
n_snaps = len(snapshots)
key_indices = []
if n_snaps >= 6:
    # Pick: initial, early approach, impact, early spread, max spread, retraction
    key_indices = [0, 1, 3, 5, max(7, n_snaps//2+1), min(9, n_snaps-1)]
else:
    key_indices = list(range(n_snaps))

key_labels = ['(a) $t=0$', '(b) Approach', '(c) Impact',
              '(d) Spreading', '(e) Max spread', '(f) Retraction']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
cy_int = int(cy)

for idx, (ki, label) in enumerate(zip(key_indices, key_labels)):
    ax = axes[idx // 3, idx % 3]
    phi = snapshots[ki]
    step = steps[ki]

    phi_xz = phi[:, cy_int, :].T
    solid_xz = solid[:, cy_int, :].T

    phi_plot = np.ma.array(phi_xz, mask=solid_xz)

    im = ax.pcolormesh(phi_plot, cmap=cmap, vmin=0, vmax=1, shading='gouraud')

    # Solid boundary
    ax.contour(solid_xz.astype(float), levels=[0.5], colors='black',
               linewidths=1.5)

    # Interface contour (phi=0.5)
    ax.contour(phi_plot, levels=[0.5], colors='yellow', linewidths=1.2,
               linestyles='-')

    # Measure Dx, Dz
    interface = (phi_xz > 0.5) & ~solid_xz
    if interface.any():
        coords = np.argwhere(interface)
        Dx = coords[:, 1].max() - coords[:, 1].min() + 1
        Dz = coords[:, 0].max() - coords[:, 0].min() + 1
    else:
        Dx = Dz = 0

    ax.set_title(f'{label}\n$t={step}$, $D_x={Dx}$, $D_z={Dz}$', fontsize=11)
    ax.set_xlabel('$x$ (perp. to ridge)')
    ax.set_ylabel('$z$ (vertical)')
    ax.set_aspect('equal')

    # Zoom to region of interest
    margin = 10
    x_lo = max(0, int(cx - R_drop*1.5 - margin))
    x_hi = min(nx, int(cx + R_drop*1.5 + margin))
    z_hi = min(nz, int(cz + R_drop + margin))
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0, z_hi)

# Colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
cb = fig.colorbar(im, cax=cbar_ax)
cb.set_label('$\\phi$', fontsize=11)
cb.set_ticks([0, 0.5, 1])
cb.set_ticklabels(['0 (gas)', '0.5', '1 (liquid)'])

fig.suptitle(
    f'Droplet Impact on Ridge — Cross-Section ($x$-$z$)\n'
    f'$D/D_0=1.0$, $We=7.9$, $\\theta=162°$, '
    f'{n_base}$^3$ VP (amp={amp:.1f})',
    fontsize=13, y=0.99, fontweight='bold')
plt.subplots_adjust(right=0.90, hspace=0.35, wspace=0.25)

save_path = os.path.join(save_dir, 'paper_snapshots_xz.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Saved: {save_path}')
plt.close()

# ===== Figure 2: Time-sequence x-y top-views =====
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))

for idx, (ki, label) in enumerate(zip(key_indices, key_labels)):
    ax = axes2[idx // 3, idx % 3]
    phi = snapshots[ki]
    step = steps[ki]

    # Slice at a few grid points above ridge surface
    z_slice = min(ridge_top + 4, nz - 1)
    phi_xy = phi[:, :, z_slice].T
    solid_xy = solid[:, :, z_slice].T

    phi_plot2 = np.ma.array(phi_xy, mask=solid_xy)
    im2 = ax.pcolormesh(phi_plot2, cmap=cmap, vmin=0, vmax=1, shading='gouraud')

    if solid_xy.any():
        ax.contour(solid_xy.astype(float), levels=[0.5], colors='black',
                   linewidths=1.5)

    # Interface
    interface = (phi_xy > 0.5) & ~solid_xy
    if interface.any():
        coords = np.argwhere(interface)
        Dx = coords[:, 1].max() - coords[:, 1].min() + 1
        Dy = coords[:, 0].max() - coords[:, 0].min() + 1
        k = Dx / Dy if Dy > 0 else 0

        x_min, x_max = coords[:, 1].min(), coords[:, 1].max()
        y_min, y_max = coords[:, 0].min(), coords[:, 0].max()

        # Draw Dx/Dy measurement lines
        ax.annotate('', xy=(x_max+1, y_max+2), xytext=(x_min-1, y_max+2),
                    arrowprops=dict(arrowstyle='<->', color='lime', lw=1.5))
        ax.annotate('', xy=(x_max+2, y_max+1), xytext=(x_max+2, y_min-1),
                    arrowprops=dict(arrowstyle='<->', color='orange', lw=1.5))
    else:
        Dx = Dy = k = 0

    # k value label
    ax.text(0.03, 0.97, f'$k={k:.2f}$',
            transform=ax.transAxes, fontsize=11, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.9),
            fontweight='bold')

    ax.set_title(f'{label}\n$t={step}$, $k=D_x/D_y={k:.2f}$', fontsize=11)
    ax.set_xlabel('$x$ (perp. to ridge)')
    ax.set_ylabel('$y$ (along ridge)')
    ax.set_aspect('equal')
    ax.set_xlim(max(0, int(cx - R_drop*1.5 - 10)), min(nx, int(cx + R_drop*1.5 + 10)))
    ax.set_ylim(max(0, int(cy - R_drop*1.5 - 10)), min(ny, int(cy + R_drop*1.5 + 10)))

cbar_ax2 = fig2.add_axes([0.92, 0.15, 0.015, 0.7])
cb2 = fig2.colorbar(im2, cax=cbar_ax2)
cb2.set_label('$\\phi$', fontsize=11)

fig2.suptitle(
    f'Droplet Impact — Top View ($x$-$y$ at surface level)\n'
    f'$D/D_0=1.0$, $We=7.9$, $\\theta=162°$, '
    f'{n_base}$^3$ VP (amp={amp:.1f})',
    fontsize=13, y=0.99, fontweight='bold')
plt.subplots_adjust(right=0.90, hspace=0.35, wspace=0.25)

save_path2 = os.path.join(save_dir, 'paper_snapshots_xy.png')
fig2.savefig(save_path2, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Saved: {save_path2}')
plt.close(fig2)

print('\nDone!')
