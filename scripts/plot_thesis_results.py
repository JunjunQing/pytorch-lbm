#!/usr/bin/env python3
"""Generate thesis figures from sweep results."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 14,
    'legend.fontsize': 10, 'figure.dpi': 150,
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
})

ROOT = Path(__file__).resolve().parent.parent
results_dir = ROOT / 'results'
fig_dir = results_dir / 'thesis_figures'
fig_dir.mkdir(exist_ok=True)

with open(results_dir / 'thesis_sweep_results.json') as f:
    data = json.load(f)
with open(results_dir / 'thesis_k_history.json') as f:
    history = json.load(f)

# Also load existing sweep data
try:
    with open(results_dir / 'liu2015_sweep_results.json') as f:
        liu_data = json.load(f)
except FileNotFoundError:
    liu_data = []

main = [d for d in data if d['label'].startswith('s')]

# =====================================================================
# Figure 1: Substrate type comparison (bar chart)
# =====================================================================
fig, ax = plt.subplots(figsize=(8, 5))
subs = ['flat', 'ridge', 'convex', 'concave']
labels_cn = ['Flat', 'Ridge\n(R*=1.0)', 'Convex\n(R*=1.0)', 'Concave\n(R*=1.0)']
k_vals = []
for sub in subs:
    found = [d for d in main if d['substrate_type'] == sub and d['label'].startswith('s1_')]
    if found:
        k_vals.append(found[0]['max_k'])
    else:
        k_vals.append(0)

colors = ['#4ECDC4', '#FF6B6B', '#45B7D1', '#96CEB4']
bars = ax.bar(labels_cn, k_vals, color=colors, edgecolor='black', linewidth=0.8, width=0.6)
for bar, k in zip(bars, k_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{k:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel('Asymmetry factor $k = D_x / D_y$')
ax.set_title('Substrate Type Comparison (We=7.9, $\\theta$=162°)')
ax.set_ylim(0, 3.5)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='$k=1$ (symmetric)')
ax.legend()
fig.tight_layout()
fig.savefig(fig_dir / 'fig1_substrate_comparison.png')
plt.close()
print(f'Saved fig1_substrate_comparison.png')

# =====================================================================
# Figure 2: R* sweep on ridge (with existing data merged)
# =====================================================================
fig, ax = plt.subplots(figsize=(8, 5))

# Combine new + existing data
r_vals = [0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.4, 1.5, 2.0, 2.76, 2.8]
k_ridge = {}
for d in main:
    if d['label'].startswith('s2_'):
        k_ridge[d['R_star']] = d['max_k']
for d in liu_data:
    if d.get('We') == 7.9 and d.get('stable'):
        k_ridge[d['D_over_D0']] = d['max_k']

rs = sorted(k_ridge.keys())
ks = [k_ridge[r] for r in rs]

ax.plot(rs, ks, 'o-', color='#FF6B6B', markersize=8, linewidth=2, label='AC-LBM (100³)')
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Curvature ratio $R^* = R_{substrate}/R_{droplet}$')
ax.set_ylabel('Asymmetry factor $k = D_x / D_y$')
ax.set_title('Curvature Ratio Effect on Ridge (We=7.9, $\\theta$=162°)')
ax.set_xscale('log')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(fig_dir / 'fig2_rstar_sweep.png')
plt.close()
print(f'Saved fig2_rstar_sweep.png')

# =====================================================================
# Figure 3: Weber number sweep
# =====================================================================
fig, ax = plt.subplots(figsize=(8, 5))

we_flat = [(d['We'], d['max_k']) for d in main if d['label'].startswith('s3_flat')]
we_ridge = [(d['We'], d['max_k']) for d in main if d['label'].startswith('s3_ridge')]

we_flat.sort()
we_ridge.sort()

if we_flat:
    wf, kf = zip(*we_flat)
    ax.plot(wf, kf, 's-', color='#4ECDC4', markersize=8, linewidth=2, label='Flat')
if we_ridge:
    wr, kr = zip(*we_ridge)
    ax.plot(wr, kr, 'o-', color='#FF6B6B', markersize=8, linewidth=2, label='Ridge R*=1.0')

    # Clanet scaling fit: k ∝ We^{1/4}
    from numpy.polynomial import polynomial as P
    we_arr = np.array(wr)
    k_arr = np.array(kr)
    # Fit k = a * We^{1/4}
    coeffs = np.polyfit(we_arr**0.25, k_arr, 1)
    we_fit = np.linspace(4, 16, 100)
    k_fit = coeffs[0] * we_fit**0.25 + coeffs[1]
    ax.plot(we_fit, k_fit, '--', color='gray', alpha=0.7,
            label=f'Clanet fit: $k \\propto We^{{1/4}}$')

ax.set_xlabel('Weber number $We$')
ax.set_ylabel('Asymmetry factor $k$')
ax.set_title('Weber Number Effect ($\\theta$=162°)')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(fig_dir / 'fig3_we_sweep.png')
plt.close()
print(f'Saved fig3_we_sweep.png')

# =====================================================================
# Figure 4: Contact angle effect
# =====================================================================
fig, ax = plt.subplots(figsize=(8, 5))

theta_ridge = [(d['theta_eq'], d['max_k']) for d in main
               if d['label'].startswith('s4_ridge')]
theta_ridge.sort()

if theta_ridge:
    th, kth = zip(*theta_ridge)
    ax.plot(th, kth, 'o-', color='#FF6B6B', markersize=8, linewidth=2, label='Ridge R*=1.0')
    for t, k in zip(th, kth):
        ax.annotate(f'{k:.3f}', (t, k), textcoords="offset points",
                    xytext=(0, 12), ha='center', fontsize=9)

ax.set_xlabel('Contact angle $\\theta$ (degrees)')
ax.set_ylabel('Asymmetry factor $k$')
ax.set_title('Contact Angle Effect (We=7.9, Ridge R*=1.0)')
ax.set_xticks([60, 90, 120, 140, 162])
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(fig_dir / 'fig4_contact_angle.png')
plt.close()
print(f'Saved fig4_contact_angle.png')

# =====================================================================
# Figure 5: Time-resolved dynamics for key cases
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 5a: k(t) for different substrates
ax = axes[0]
key_cases = ['s1_ridge_R1.0', 's1_flat_Rflat', 's1_convex_R1.0', 's1_concave_R1.0']
colors_t = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
for case, color in zip(key_cases, colors_t):
    if case in history:
        h = history[case]
        steps = [d['step'] for d in h]
        ks = [d['k'] for d in h]
        label = case.replace('s1_', '').replace('_R1.0', '').replace('_Rflat', '')
        ax.plot(steps, ks, '-', color=color, linewidth=1.5, label=label.capitalize())

ax.set_xlabel('Time step')
ax.set_ylabel('$k = D_x / D_y$')
ax.set_title('Spreading Dynamics by Substrate (We=7.9)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 4)

# 5b: k(t) for different We on ridge
ax = axes[1]
we_cases = ['s3_ridge_R1.0_We5.0', 's3_ridge_R1.0_We7.9',
            's3_ridge_R1.0_We12.0', 's3_ridge_R1.0_We15.0']
colors_w = ['#4ECDC4', '#FF6B6B', '#45B7D1', '#FFD93D']
for case, color in zip(we_cases, colors_w):
    if case in history:
        h = history[case]
        steps = [d['step'] for d in h]
        ks = [d['k'] for d in h]
        we = case.split('We')[-1]
        ax.plot(steps, ks, '-', color=color, linewidth=1.5, label=f'We={we}')

ax.set_xlabel('Time step')
ax.set_ylabel('$k = D_x / D_y$')
ax.set_title('Spreading Dynamics by Weber Number (Ridge)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 5)

fig.tight_layout()
fig.savefig(fig_dir / 'fig5_time_dynamics.png')
plt.close()
print(f'Saved fig5_time_dynamics.png')

# =====================================================================
# Figure 6: Dx-Dy phase space
# =====================================================================
fig, ax = plt.subplots(figsize=(7, 7))

for d in main:
    if d['label'].startswith('s4_ridge'):
        color = '#FF6B6B'
        marker = 'o'
    elif d['label'].startswith('s3_ridge'):
        color = '#45B7D1'
        marker = 's'
    elif d['label'].startswith('s2_'):
        color = '#96CEB4'
        marker = '^'
    else:
        continue
    ax.scatter(d['Dx'], d['Dy'], c=color, marker=marker, s=100, zorder=5,
               edgecolors='black', linewidth=0.5)

# Add k=1 line
lims = [10, 70]
ax.plot(lims, lims, 'k--', alpha=0.3, label='$k=1$')
# Add k=2, k=3 lines
ax.plot(lims, [l/2 for l in lims], 'k:', alpha=0.3, label='$k=2$')
ax.plot(lims, [l/3 for l in lims], 'k-.', alpha=0.3, label='$k=3$')

# Legend patches
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#45B7D1',
           markersize=10, label='We sweep'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6B6B',
           markersize=10, label='Contact angle'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='#96CEB4',
           markersize=10, label='R* sweep'),
]
ax.legend(handles=legend_elements + ax.get_legend_handles_labels()[0], loc='upper left')

ax.set_xlabel('$D_x$ (across ridge)')
ax.set_ylabel('$D_y$ (along ridge)')
ax.set_title('$D_x$-$D_y$ Phase Space (Ridge Substrate)')
ax.set_xlim(30, 70)
ax.set_ylim(10, 35)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(fig_dir / 'fig6_dx_dy_phase.png')
plt.close()
print(f'Saved fig6_dx_dy_phase.png')

print(f'\nAll figures saved to {fig_dir}/')
