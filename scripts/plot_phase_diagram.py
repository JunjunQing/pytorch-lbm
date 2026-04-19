#!/usr/bin/env python3
"""Generate We-D/D0 phase diagram from FULL measured sweep data.

Uses the complete 40-case phase_diagram_results.json with all
measured k values across 5 We x 8 D/D0 parameter space.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import griddata

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'mathtext.fontset': 'cm',
})

# ===== Load FULL measured data =====
import json, os
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'phase_diagram_results.json')
with open(data_path) as f:
    data = json.load(f)

dd0_raw = np.array(data['dd0_values'])
we_vals = np.array(data['we_values'])
k_matrix = np.array(data['k_matrix'])

# Replace flat (999) with 3.5 for plotting, use its k=1.0 data
dd0_plot = dd0_raw.copy()
flat_idx = np.where(dd0_plot == 999)[0]
dd0_plot[flat_idx] = 3.5

# Build scatter points for interpolation (exclude flat)
pts = []
vals = []
for i, we in enumerate(we_vals):
    for j, dd0 in enumerate(dd0_plot):
        pts.append([dd0, we])
        vals.append(k_matrix[i, j])

pts = np.array(pts)
vals = np.array(vals)

# Interpolate to fine grid
dd0_grid = np.logspace(np.log10(0.4), np.log10(3.5), 60)
we_grid = np.linspace(4, 25, 60)
DD0, WE = np.meshgrid(dd0_grid, we_grid)

K_cubic = griddata(pts, vals, (DD0, WE), method='cubic')
K_nearest = griddata(pts, vals, (DD0, WE), method='nearest')
K_interp = np.where(np.isnan(K_cubic), K_nearest, K_cubic)
K_interp = np.clip(K_interp, 0.5, 5.0)

# ===== Plot =====
fig, ax = plt.subplots(figsize=(11, 7.5))

# Contour fill
levels = np.arange(0.8, 4.5, 0.2)
cf = ax.contourf(DD0, WE, K_interp, levels=levels, cmap='RdYlBu_r', extend='both')
cb = plt.colorbar(cf, ax=ax, label='$k_{max} = D_x/D_y$ (spread asymmetry)', shrink=0.85)

# Contour lines
cs = ax.contour(DD0, WE, K_interp, levels=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
                colors='black', linewidths=0.8, alpha=0.6)
ax.clabel(cs, fmt='%.1f', fontsize=9)

# k=1 line (symmetric boundary)
ax.contour(DD0, WE, K_interp, levels=[1.0], colors='blue', linewidths=2, linestyles='--')

# All measured data points with k values
for i, we in enumerate(we_vals):
    for j, dd0 in enumerate(dd0_plot):
        k = k_matrix[i, j]
        size = 120 if (abs(dd0 - 1.0) < 0.01 and abs(we - 7.9) < 0.1) or \
                      (abs(dd0 - 2.76) < 0.01 and abs(we - 7.9) < 0.1) else 50
        ax.scatter(dd0, we, s=size, c='white', edgecolors='black',
                   linewidths=1, zorder=5)
        # Show k value at key points
        if abs(dd0 - 1.0) < 0.01 or abs(dd0 - 2.76) < 0.01:
            ax.annotate(f'{k:.2f}', (dd0, we), textcoords="offset points",
                       xytext=(8, 3), fontsize=7, color='black')

# Paper target annotations
ax.annotate('Liu 2015\n$k\\approx2.6$', xy=(1.0, 7.9), xytext=(1.6, 5.5),
            fontsize=10, fontweight='bold', color='darkred',
            arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5))
ax.annotate('Liu 2015\n$k\\approx1.33$', xy=(2.76, 7.9), xytext=(2.8, 12),
            fontsize=10, fontweight='bold', color='darkred',
            arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5))

# Flat plate label
ax.annotate('Flat plate\n($k=1$ always)', xy=(3.5, 14.5), fontsize=9,
            ha='center', color='gray',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))

ax.set_xlabel('$D/D_0$ (ridge/droplet diameter ratio)')
ax.set_ylabel('Weber number ($We$)')
ax.set_title('Spread Asymmetry Phase Diagram — Measured Data\n'
             '($\\theta=162°$, $\\rho_l/\\rho_g=828$, Allen-Cahn LBM, 40 cases)')
ax.set_xscale('log')
ax.set_xlim(0.4, 3.8)
ax.set_ylim(4, 25)
ax.grid(True, alpha=0.15)

# Legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
           markeredgecolor='black', markersize=7, label='Measured data (40 cases)'),
    Line2D([0], [0], color='blue', linestyle='--', linewidth=2, label='$k=1$ (symmetric)'),
    Line2D([0], [0], color='black', linewidth=0.8, alpha=0.6, label='$k$ contours'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
           markeredgecolor='darkred', markersize=12, label='Liu et al. 2015 targets'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'phase_diagram.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
print('Saved: phase_diagram.png')

plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'phase_diagram_preview.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
print('Saved: phase_diagram_preview.png')

# Print summary
print('\n' + '='*70)
print('  MEASURED k MATRIX (We rows x D/D0 columns)')
print('='*70)
header = '  We\\D/D0 |'
for dd0 in dd0_raw:
    label = "flat" if dd0 > 100 else f"{dd0:.2f}"
    header += f" {label:>6}"
print(header)
print('  ' + '-'*len(header))
for i, we in enumerate(we_vals):
    row = f"  {we:6.1f}  |"
    for j in range(len(dd0_raw)):
        row += f" {k_matrix[i, j]:6.3f}"
    print(row)
