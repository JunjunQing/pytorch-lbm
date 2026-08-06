#!/usr/bin/env python3
"""Generate energy budget and physics analysis figures."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

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

# Load energy data
with open('/mnt/simulation-projects/pytorch_lbm/energy_analysis.pkl', 'rb') as f:
    data = pickle.load(f)

flat = data['flat']
ridge1 = data['ridge_d1']
ridge2 = data['ridge_d276']

# ===== Figure 1: Energy Budget =====
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

colors = {'Flat': '#2196F3', 'D/D₀=1.0': '#E91E63', 'D/D₀=2.76': '#4CAF50'}

# (a) Kinetic Energy vs time
ax = axes[0, 0]
for name, ehist, col in [('Flat', flat, colors['Flat']),
                          ('D/D₀=1.0', ridge1, colors['D/D₀=1.0']),
                          ('D/D₀=2.76', ridge2, colors['D/D₀=2.76'])]:
    steps = [s for s, e in ehist]
    KE = [e['KE'] for s, e in ehist]
    ax.plot(steps, KE, '-', linewidth=2, color=col, label=name)

ax.set_xlabel('Time step')
ax.set_ylabel('Kinetic Energy ($E_k$)')
ax.set_title('(a) Kinetic Energy Evolution')
ax.legend()
ax.set_xlim(0, 2000)
ax.grid(True, alpha=0.2)

# (b) Surface Energy vs time
ax = axes[0, 1]
for name, ehist, col in [('Flat', flat, colors['Flat']),
                          ('D/D₀=1.0', ridge1, colors['D/D₀=1.0']),
                          ('D/D₀=2.76', ridge2, colors['D/D₀=2.76'])]:
    steps = [s for s, e in ehist]
    SE = [e['SE'] for s, e in ehist]
    ax.plot(steps, SE, '-', linewidth=2, color=col, label=name)

ax.set_xlabel('Time step')
ax.set_ylabel('Surface Energy ($E_s$)')
ax.set_title('(b) Surface Energy Evolution')
ax.legend()
ax.set_xlim(0, 2000)
ax.grid(True, alpha=0.2)

# (c) k vs time
ax = axes[1, 0]
for name, ehist, col in [('Flat', flat, colors['Flat']),
                          ('D/D₀=1.0', ridge1, colors['D/D₀=1.0']),
                          ('D/D₀=2.76', ridge2, colors['D/D₀=2.76'])]:
    steps = [s for s, e in ehist]
    k = [e['k'] for s, e in ehist]
    ax.plot(steps, k, '-', linewidth=2, color=col, label=name)

ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4)
ax.axhline(y=2.6, color='red', linestyle='--', alpha=0.4, label='Paper target $k≈2.6$')
ax.set_xlabel('Time step')
ax.set_ylabel('$k = D_x/D_y$')
ax.set_title('(c) Spread Asymmetry Evolution')
ax.legend(fontsize=8)
ax.set_xlim(0, 2000)
ax.grid(True, alpha=0.2)

# (d) Dx and Dy vs time for D/D0=1.0
ax = axes[1, 1]
for name, ehist, col in [('D/D₀=1.0', ridge1, colors['D/D₀=1.0'])]:
    steps = [s for s, e in ehist]
    Dx = [e['Dx'] for s, e in ehist]
    Dy = [e['Dy'] for s, e in ehist]
    ax.plot(steps, Dx, '-', linewidth=2, color='red', label='$D_x$ (perp. to ridge)')
    ax.plot(steps, Dy, '-', linewidth=2, color='blue', label='$D_y$ (along ridge)')
    ax.fill_between(steps, Dx, Dy, alpha=0.1, color='purple')

ax.axhline(y=30, color='gray', linestyle=':', alpha=0.5, label='$D_0=30$')
ax.set_xlabel('Time step')
ax.set_ylabel('Spread diameter (lattice units)')
ax.set_title('(d) $D_x$ and $D_y$ Evolution ($D/D_0=1.0$)')
ax.legend()
ax.set_xlim(0, 2000)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('/mnt/simulation-projects/pytorch_lbm/energy_analysis.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print('Saved: energy_analysis.png')

# ===== Figure 2: Asymmetry mechanism =====
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# k_max comparison
cases = ['Flat', 'D/D₀=0.5', 'D/D₀=1.0', 'D/D₀=2.76']
k_max = [1.000, 2.684, 2.684, 1.320]
colors_bar = ['#2196F3', '#E91E63', '#E91E63', '#4CAF50']

bars = ax1.bar(cases, k_max, color=colors_bar, edgecolor='black', linewidth=0.5)

# Paper targets
ax1.axhline(y=2.6, color='red', linestyle='--', alpha=0.6, label='Liu 2015 $D/D_0=1$')
ax1.axhline(y=1.33, color='orange', linestyle='--', alpha=0.6, label='Liu 2015 $D/D_0=2.76$')

for bar, k in zip(bars, k_max):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{k:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax1.set_ylabel('$k_{max} = D_x/D_y$')
ax1.set_title('Maximum Spread Asymmetry')
ax1.legend(fontsize=8)
ax1.set_ylim(0, 3.2)
ax1.grid(True, alpha=0.2, axis='y')

# KE dissipation ratio
labels = ['Flat', 'D/D₀=1.0', 'D/D₀=2.76']
ke_ratios = [0.139, 0.088, 0.185]
ke_colors = ['#2196F3', '#E91E63', '#4CAF50']

bars2 = ax2.bar(labels, ke_ratios, color=ke_colors, edgecolor='black', linewidth=0.5)
for bar, r in zip(bars2, ke_ratios):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{r:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2.set_ylabel('$E_k^{final}/E_k^{initial}$')
ax2.set_title('Residual Kinetic Energy Ratio')
ax2.set_ylim(0, 0.25)
ax2.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
plt.savefig('/mnt/simulation-projects/pytorch_lbm/asymmetry_mechanism.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print('Saved: asymmetry_mechanism.png')

# Print physics analysis
print('\n' + '='*60)
print('  Physics Analysis')
print('='*60)
print(f'\n  Energy conservation (mass):')
for name, ehist in [('Flat', flat), ('D/D0=1.0', ridge1), ('D/D0=2.76', ridge2)]:
    mass_init = ehist[0][1]['mass']
    mass_final = ehist[-1][1]['mass']
    drift = abs(mass_final - mass_init) / mass_init * 100
    print(f'    {name:>10}: mass drift = {drift:.4f}%')

print(f'\n  Peak k and timing:')
for name, ehist in [('Flat', flat), ('D/D0=1.0', ridge1), ('D/D0=2.76', ridge2)]:
    max_k = 0
    max_step = 0
    for s, e in ehist:
        if e['k'] > max_k:
            max_k = e['k']
            max_step = s
    print(f'    {name:>10}: k_max={max_k:.3f} at step {max_step}')

print(f'\n  KE analysis:')
for name, ehist in [('Flat', flat), ('D/D0=1.0', ridge1), ('D/D0=2.76', ridge2)]:
    KE_min = min(e['KE'] for s, e in ehist)
    KE_min_step = [s for s, e in ehist if e['KE'] == KE_min][0]
    print(f'    {name:>10}: KE_min={KE_min:.4f} at step {KE_min_step}')
