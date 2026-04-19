#!/usr/bin/env python3
"""Generate resolution convergence publication figure with calibrated amp."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.labelsize': 12, 'axes.titlesize': 13,
    'legend.fontsize': 10, 'figure.dpi': 150,
    'savefig.dpi': 300, 'mathtext.fontset': 'cm',
})

save_dir = os.path.dirname(os.path.abspath(__file__))

# Calibrated convergence data (density-weighted tau, amp=1.8 at all resolutions)
calibrated = [
    {'N': 80,  'amp': 1.8, 'k': 2.600, 'Dx': 39, 'Dy': 15, 'time_s': 183},
    {'N': 100, 'amp': 1.8, 'k': 2.579, 'Dx': 49, 'Dy': 19, 'time_s': 262},
    {'N': 120, 'amp': 1.8, 'k': 2.565, 'Dx': 59, 'Dy': 23, 'time_s': 398},
    {'N': 150, 'amp': 1.8, 'k': 2.579, 'Dx': 83, 'Dy': 31, 'time_s': 710},
]

# Fixed amp=1.8 data (density-weighted tau: same as calibrated since amp is constant)
fixed_amp = [
    {'N': 80,  'amp': 1.8, 'k': 2.600},
    {'N': 100, 'amp': 1.8, 'k': 2.579},
    {'N': 120, 'amp': 1.8, 'k': 2.565},
    {'N': 150, 'amp': 1.8, 'k': 2.579},
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# ---- (a) k vs N: calibrated vs fixed amp ----
# Calibrated (best amp at each N)
n_cal = [d['N'] for d in calibrated]
k_cal = [d['k'] for d in calibrated]
err_cal = [abs(d['k'] - 2.6) / 2.6 * 100 for d in calibrated]

ax1.plot(n_cal, k_cal, 'rs-', ms=10, lw=2.5, label='Calibrated amp', zorder=5)

# Fixed amp=2.0
n_fix = [d['N'] for d in fixed_amp]
k_fix = [d['k'] for d in fixed_amp]
ax1.plot(n_fix, k_fix, 'b^--', ms=8, lw=1.5, alpha=0.6, label='Density-weighted τ (amp=1.8)')

# Target line and error band
ax1.axhline(y=2.6, color='black', ls='--', lw=2, alpha=0.6, label='Liu 2015: $k\\approx 2.6$')
ax1.fill_between([55, 160], 2.6 * 0.95, 2.6 * 1.05, alpha=0.12, color='green',
                 label='±5% error band')

# Annotate calibrated points with amp and error
for d in calibrated:
    n, k, amp = d['N'], d['k'], d['amp']
    err = abs(k - 2.6) / 2.6 * 100
    ax1.annotate(f'amp={amp}\n{err:.1f}%', (n, k), xytext=(0, 15),
                 fontsize=8, ha='center', textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.2', fc='lightyellow', ec='gray', alpha=0.9))

ax1.set_xlabel('Grid resolution $N^3$')
ax1.set_ylabel('$k_{max} = D_x / D_y$')
ax1.set_title('(a) Resolution Convergence\n($D/D_0=1.0$, $We=7.9$, density-weighted τ, amp=1.8)',
              fontsize=12)
ax1.legend(loc='upper right', fontsize=9)
ax1.set_xlim(55, 160)
ax1.set_ylim(1.5, 3.5)
ax1.grid(True, alpha=0.2)
ax1.set_xticks([60, 80, 100, 120, 150])

# ---- (b) amp calibration at 150³ (density-weighted tau) ----
amp_150 = [
    (1.0, 2.100), (1.5, 2.333), (1.8, 2.579),
    (2.0, 2.650), (2.3, 2.800),
]
amps = [a for a, k in amp_150]
ks = [k for a, k in amp_150]

ax2.plot(amps, ks, 'rs-', ms=9, lw=2.5, zorder=5)
ax2.axhline(y=2.6, color='black', ls='--', lw=2, alpha=0.6, label='$k=2.6$ (Liu 2015)')
ax2.fill_between(amps, 2.6 * 0.95, 2.6 * 1.05, alpha=0.12, color='green', label='±5%')

# Mark optimal
for a, k in amp_150:
    err = abs(k - 2.6) / 2.6 * 100
    if err < 5:
        ax2.plot(a, k, 'g*', ms=14, zorder=6)
        ax2.annotate(f'{err:.1f}%', (a, k), xytext=(8, -5),
                     fontsize=9, textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.2', fc='lightgreen', ec='green', alpha=0.8))

ax2.set_xlabel('Geometric amplification (amp)')
ax2.set_ylabel('$k_{max}$')
ax2.set_title('(b) Amp Calibration at $150^3$\n(VP, $D/D_0=1.0$, $We=7.9$, density-weighted τ)',
              fontsize=12)
ax2.legend(loc='upper left', fontsize=9)
ax2.set_xlim(0.8, 2.6)
ax2.set_ylim(1.8, 3.2)
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'resolution_convergence.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
print('Saved: resolution_convergence.png')
