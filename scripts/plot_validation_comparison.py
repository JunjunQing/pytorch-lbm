#!/usr/bin/env python3
"""Generate validation comparison figure for the paper.

Panel (a): k_max vs D/D0 compared with Liu 2015
Panel (b): k_max vs We power law scaling
Uses density-weighted tau data.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'mathtext.fontset': 'cm',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
})

save_dir = os.path.dirname(os.path.abspath(__file__))

# ===== D/D0 sweep data =====
# Combined from 100³ and 150³ with density-weighted tau
dd0_vals = [0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 2.76]
k_dd0 =    [5.263, 3.211, 3.353, 2.579, 1.914, 1.541, 1.308]
src_dd0 =  ['100³']*3 + ['150³']*4

# Liu 2015 targets
dd0_paper = [1.0, 2.76]
k_paper_dd0 = [2.6, 1.33]

# ===== We sweep data (from run_we_sweep_v2.py) =====
we_sweep_path = os.path.join(save_dir, 'we_sweep_v2_results.json')
if os.path.exists(we_sweep_path):
    with open(we_sweep_path) as f:
        we_results = json.load(f)
    we_vals = [r['We'] for r in we_results if r['stable']]
    k_we = [r['max_k'] for r in we_results if r['stable']]
else:
    # Placeholder: use old data until GPU run completes
    print("WARNING: we_sweep_v2_results.json not found, using placeholder data")
    we_vals = [5.0, 7.9, 12.0, 15.0, 23.6]
    k_we = [1.957, 2.579, 3.235, 3.471, 3.842]

Oh = 0.0068

# ===== Plot =====
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# (a) k vs D/D0
ax1.semilogy(dd0_vals, k_dd0, 'o-', color='#e74c3c', lw=2.5, ms=9,
             label='AC-LBM (this work)')
ax1.semilogy(dd0_paper, k_paper_dd0, '*', color='black', ms=20, zorder=5,
             label='Liu et al. 2015')
ax1.axhline(y=1.0, color='gray', ls=':', alpha=0.4, label='Perfect symmetry')

# Error annotations
for d, k_m, k_t in [(1.0, 2.579, 2.6), (2.76, 1.308, 1.33)]:
    err = abs(k_m - k_t) / k_t * 100
    ax1.annotate(f'{err:.1f}%', xy=(d, k_m), xytext=(d+0.12, k_m*1.12),
                 fontsize=9, color='#e74c3c', fontweight='bold')

ax1.set_xlabel('$D/D_0$ (ridge-to-droplet ratio)', fontsize=12)
ax1.set_ylabel('$k_{max} = D_x / D_y$', fontsize=12)
ax1.set_title('(a) Asymmetry vs Ridge Curvature\n($We=7.9$, $\\theta=162°$, density-weighted τ)',
              fontsize=12, fontweight='bold')
ax1.legend(fontsize=9, loc='upper right')
ax1.grid(True, alpha=0.15)
ax1.set_xlim(0.3, 3.2)

# (b) k vs We
Re_vals = np.sqrt(np.array(we_vals)) / Oh
ax2.plot(we_vals, k_we, 'D-', color='#8e44ad', lw=2.5, ms=9,
         label='AC-LBM (density-weighted τ)')

# Power law fit
if len(we_vals) >= 3:
    log_we = np.log(we_vals)
    log_k = np.log(k_we)
    coeffs = np.polyfit(log_we, log_k, 1)
    we_fit = np.linspace(min(we_vals)*0.8, max(we_vals)*1.1, 100)
    k_fit = np.exp(coeffs[1]) * we_fit ** coeffs[0]
    ax2.plot(we_fit, k_fit, '--', color='#8e44ad', alpha=0.5, lw=1.5,
             label=f'Fit: $k \\propto We^{{{coeffs[0]:.2f}}}$')

ax2.axhline(y=2.6, color='black', ls='--', lw=2, alpha=0.5, label='Liu 2015: $k\\approx 2.6$')

# Reference scaling lines
we_ref = np.linspace(4, 25, 50)
k_half = 2.0 * (we_ref / 7.9) ** 0.5
k_quarter = 2.0 * (we_ref / 7.9) ** 0.25
ax2.plot(we_ref, k_half, ':', color='gray', alpha=0.4, lw=1.2, label='$We^{1/2}$ (inertial)')
ax2.plot(we_ref, k_quarter, '-.', color='gray', alpha=0.4, lw=1.2, label='$We^{1/4}$ (Clanet)')

ax2.set_xlabel('Weber number $We$', fontsize=12)
ax2.set_ylabel('$k_{max}$', fontsize=12)
ax2.set_title('(b) Weber Number Dependence\n($D/D_0=1.0$, $100^3$, VP, amp=1.8)',
              fontsize=12, fontweight='bold')
ax2.legend(fontsize=9, loc='upper left')
ax2.grid(True, alpha=0.15)

plt.tight_layout()
save_path = os.path.join(save_dir, '..', 'paper', 'figures', 'validation_comparison.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Saved: {save_path}')
plt.close()

# Print summary
print('\nData used:')
print(f'  D/D0 sweep: {list(zip(dd0_vals, k_dd0))}')
print(f'  We sweep: {list(zip(we_vals, k_we))}')
if len(we_vals) >= 3:
    print(f'  Scaling: k ∝ We^{coeffs[0]:.2f}')
