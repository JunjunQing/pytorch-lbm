#!/usr/bin/env python3
"""Grid convergence analysis for reviewer response.

Two analyses:
1. Fixed amp=2.0: shows raw convergence (without calibration)
2. Calibrated amp: shows physical consistency

Key argument: amp is a closure parameter (like turbulence model coefficients)
that depends on grid resolution, but once calibrated gives correct physics.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'mathtext.fontset': 'cm',
})

k_target = 2.6

# Fixed amp=1.8 data (density-weighted tau: amp constant across resolutions)
n_fixed = [80, 100, 120, 150]
k_fixed = [2.600, 2.579, 2.565, 2.579]

# Density-weighted tau data: amp=1.8 at all resolutions
n_cal = [80, 100, 120, 150]
amp_cal = [1.8, 1.8, 1.8, 1.8]
k_cal = [2.600, 2.579, 2.565, 2.579]

# Grid spacing (inverse of resolution)
dx_fixed = [1.0/n for n in n_fixed]
dx_cal = [1.0/n for n in n_cal]

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# (a) k vs dx (log-log) — standard convergence plot
ax = axes[0]
ax.semilogx(dx_fixed, k_fixed, 'o-', color='#e74c3c', lw=2, ms=8,
            label='Fixed amp=1.8 (density-weighted τ)')
ax.semilogx(dx_cal, k_cal, 's-', color='#27ae60', lw=2.5, ms=9,
            label='Density-weighted τ (amp=1.8)')
ax.axhline(y=k_target, color='black', ls='--', lw=2, alpha=0.5,
           label='Liu 2015: $k\\approx2.6$')
ax.fill_between([0.005, 0.02], k_target*0.97, k_target*1.03,
                alpha=0.1, color='green', label='$\\pm3\\%$ band')

# Error band
for n, k in zip(n_cal, k_cal):
    err = abs(k - k_target) / k_target * 100
    ax.annotate(f'{err:.1f}%', xy=(1.0/n, k), xytext=(1.0/n*1.15, k+0.15),
                fontsize=9, color='#27ae60')

ax.set_xlabel('Grid spacing $\\Delta x = 1/N$', fontsize=12)
ax.set_ylabel('$k_{\\max} = D_x / D_y$', fontsize=12)
ax.set_title('(a) Grid Convergence\n($D/D_0=1.0$, We=7.9, density-weighted τ)', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.set_xlim(0.005, 0.02)
ax.grid(True, alpha=0.15)

# (b) Error vs dx (log-log) — Richardson convergence
ax = axes[1]
err_fixed = [abs(k - k_target)/k_target for k in k_fixed]
err_cal = [abs(k - k_target)/k_target for k in k_cal]

ax.loglog(dx_fixed, err_fixed, 'o-', color='#e74c3c', lw=2, ms=8,
          label='Fixed amp=1.8')
ax.loglog(dx_cal, err_cal, 's-', color='#27ae60', lw=2.5, ms=9,
          label='Density-weighted τ')

# Reference slopes
x_ref = np.array([0.006, 0.018])
for p, lbl in [(1, '$O(\\Delta x)$'), (2, '$O(\\Delta x^2)$')]:
    y_ref = 0.5 * (x_ref / 0.01)**p
    ax.loglog(x_ref, y_ref, '--', color='gray', alpha=0.5, lw=1.5, label=lbl)

ax.set_xlabel('Grid spacing $\\Delta x = 1/N$', fontsize=12)
ax.set_ylabel('Relative error $|k - k_{target}| / k_{target}$', fontsize=12)
ax.set_title('(b) Convergence Rate\n(density-weighted τ, amp=1.8)', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.15)
ax.set_ylim(0.001, 2)

# (c) amp vs resolution — closure parameter dependence
ax = axes[2]
ax.plot(n_cal, amp_cal, 'D-', color='#8e44ad', lw=2.5, ms=10, label='Optimal amp')

# Density-weighted tau: amp is constant!
ax.axhline(y=1.8, color='green', ls='--', lw=2, alpha=0.5, label='amp=1.8 (density-weighted τ)')

# Annotate each point
for n, a, k in zip(n_cal, amp_cal, k_cal):
    ax.annotate(f'amp={a:.1f}\n$k$={k:.2f}', xy=(n, a),
                xytext=(n+5, a+0.2), fontsize=8, color='#5B2C6F')

ax.set_xlabel('Grid resolution $N$', fontsize=12)
ax.set_ylabel('Optimal amp (closure parameter)', fontsize=12)
ax.set_title('(c) Closure Parameter: amp vs Resolution\n(density-weighted τ: amp stable at 1.8)', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.15)
ax.set_xlim(65, 170)
ax.set_ylim(0, 3)

# Add argument text box
textstr = ('With density-weighted τ interpolation,\n'
           'amp=1.8 works at all resolutions\n'
           '(100³–150³), eliminating the previous\n'
           'resolution dependence.')
props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
ax.text(0.55, 0.25, textstr, transform=ax.transAxes, fontsize=8,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('/mnt/simulation-projects/pytorch_lbm/paper/figures/grid_convergence.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print('Saved: grid_convergence.png')

# Print analysis
print('\n' + '='*60)
print('  GRID CONVERGENCE ANALYSIS')
print('='*60)
print(f'\n  Fixed amp=1.8 (density-weighted τ) convergence:')
for n, k, e in zip(n_fixed, k_fixed, err_fixed):
    print(f'    N={n:3d}: k={k:.3f}, err={e:.1%}')

print(f'\n  Density-weighted τ convergence:')
for n, k, a, e in zip(n_cal, k_cal, amp_cal, err_cal):
    print(f'    N={n:3d}: amp={a:.1f}, k={k:.3f}, err={e:.1%}')

print(f'\n  Key: amp=1.8 is constant across all resolutions (density-weighted τ)')
