#!/usr/bin/env python3
"""Generate benchmark comparison plots for Liu 2015 paper validation."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Data from sweep results (corrected half-cylinder geometry)
dd0_values = [0.6, 1.0, 1.2, 1.5, 2.0, 2.76]
k_values = [2.579, 2.684, 2.579, 2.143, 1.870, 1.296]
amp_values = [1.5, 1.5, 1.5, 1.0, 0.5, 0.0]

# Paper targets (from liu2015.md)
paper_dd0 = [1.0, 2.76]
paper_k = [2.6, 1.33]

# We sweep data (D/D0=1.0, amp=1.5)
we_values = [7.9, 12.0, 15.0, 23.6]
we_k = [2.684, 3.353, 3.471, 3.941]

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Plot 1: k vs D/D0 ---
ax1.plot(dd0_values, k_values, 'bo-', markersize=8, linewidth=2, label='AC-LBM (this work)')
ax1.scatter(paper_dd0, paper_k, marker='*', s=200, c='red', zorder=5, label='Liu 2015 (paper)')
ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Flat plate (k=1)')

# Annotate amp values
for i, (d, k, a) in enumerate(zip(dd0_values, k_values, amp_values)):
    ax1.annotate(f'amp={a}', (d, k), textcoords="offset points",
                xytext=(0, 12), fontsize=8, ha='center')

ax1.set_xlabel('D/D₀ (ridge/droplet diameter ratio)', fontsize=12)
ax1.set_ylabel('k = Dx/Dy (spread asymmetry)', fontsize=12)
ax1.set_title('Spread Asymmetry vs Ridge Curvature\n(We=7.9, θ=162°, ρ_l/ρ_g=828)', fontsize=13)
ax1.legend(fontsize=10, loc='upper right')
ax1.set_xlim(0.3, 3.0)
ax1.set_ylim(0.5, 3.2)
ax1.grid(True, alpha=0.3)

# --- Plot 2: k vs We ---
ax2.plot(we_values, we_k, 'rs-', markersize=8, linewidth=2, label='AC-LBM (D/D₀=1.0)')
ax2.scatter([7.9], [2.6], marker='*', s=200, c='blue', zorder=5, label='Liu 2015 target')

ax2.set_xlabel('Weber number (We)', fontsize=12)
ax2.set_ylabel('k = Dx/Dy (spread asymmetry)', fontsize=12)
ax2.set_title('Spread Asymmetry vs Weber Number\n(D/D₀=1.0, θ=162°, amp=1.5)', fontsize=13)
ax2.legend(fontsize=10)
ax2.set_xlim(5, 26)
ax2.set_ylim(2.0, 4.5)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/simulation-projects/pytorch_lbm/liu2015_benchmark.png', dpi=150, bbox_inches='tight')
print('Saved: liu2015_benchmark.png')

# --- Plot 3: Time evolution of k for key cases ---
fig2, ax3 = plt.subplots(1, 1, figsize=(10, 6))

# Paper grid data (150x150x200, D/D0=1.0, amp=1.5)
steps_paper = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400]
k_paper = [1.000, 1.108, 1.343, 1.690, 2.429, 2.684, 2.333, 1.800, 1.207, 0.758, 0.622, 0.657]

ax3.plot(steps_paper, k_paper, 'b-', linewidth=2, label='D/D₀=1.0, 150³×200 grid')
ax3.axhline(y=2.6, color='red', linestyle='--', alpha=0.5, label='Paper target k≈2.6')
ax3.fill_between(steps_paper, k_paper, alpha=0.1, color='blue')

ax3.set_xlabel('Time step', fontsize=12)
ax3.set_ylabel('k = Dx/Dy', fontsize=12)
ax3.set_title('Temporal Evolution of Spread Asymmetry\n(D/D₀=1.0, We=7.9, θ=162°, amp=1.5)', fontsize=13)
ax3.legend(fontsize=10)
ax3.set_xlim(0, 2500)
ax3.set_ylim(0, 3.2)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/simulation-projects/pytorch_lbm/liu2015_time_evolution.png', dpi=150, bbox_inches='tight')
print('Saved: liu2015_time_evolution.png')

# Print summary
print('\n' + '='*60)
print('  Liu 2015 Benchmark Summary')
print('='*60)
print(f'\n  {"Case":>15} {"Our k":>8} {"Paper k":>8} {"Error":>8} {"Status":>8}')
print('  ' + '-'*50)
cases = [
    ('D/D₀=1.0', 2.684, 2.6),
    ('D/D₀=2.76', 1.296, 1.33),
    ('Flat plate', 1.000, 1.0),
]
for name, ours, paper in cases:
    err = abs(ours - paper) / paper * 100
    status = '✓' if err < 5 else '~'
    print(f'  {name:>15} {ours:8.3f} {paper:8.2f} {err:7.1f}% {status:>8}')
