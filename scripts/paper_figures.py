#!/usr/bin/env python3
"""Generate publication-quality figures for the geo_amplification paper."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

# Output directory
fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'paper', 'figures')
os.makedirs(fig_dir, exist_ok=True)

# Publication style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.linewidth': 1.0,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.5,
    'ytick.minor.width': 0.5,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '0.8',
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# Colors
colors = {
    'blue': '#1f77b4', 'orange': '#ff7f0e', 'green': '#2ca02c',
    'red': '#d62728', 'purple': '#9467bd', 'brown': '#8c564b',
    'gray': '#7f7f7f',
}

# ============================================================
# Figure 1: Phase diagram — k vs We at different R*
# ============================================================
def fig1_phase_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))

    # Data: We sweep at R*=1.0 (N=150, α=1.5 — overnight sweep)
    We_r1 = [3.0, 5.0, 7.9, 10.0, 15.0, 20.0]
    k_r1  = [1.615, 1.971, 2.655, 3.074, 3.296, 2.657]

    # Plot k vs We for R*=1.0
    ax.plot(We_r1, k_r1, 'o-', color=colors['blue'], label='$R^*=1.0$', zorder=5)

    # Mark peak
    idx_peak = k_r1.index(max(k_r1))
    ax.annotate(f'$k_{{max}}={max(k_r1):.2f}$',
                xy=(We_r1[idx_peak], k_r1[idx_peak]),
                xytext=(We_r1[idx_peak]+2, k_r1[idx_peak]+0.15),
                fontsize=9, ha='left',
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

    # Flat reference
    ax.axhline(y=1.0, color=colors['gray'], linestyle='--', linewidth=0.8, alpha=0.6)
    ax.text(28, 1.02, 'flat', fontsize=8, color=colors['gray'], ha='right')

    # Regime thresholds (per main text: k<1.2 symmetric, 1.2<=k<2.0 moderate,
    # k>=2.0 highly asymmetric; no breakup claim)
    ax.axhline(y=1.2, color='0.5', linestyle=':', linewidth=0.8, alpha=0.8)
    ax.axhline(y=2.0, color='0.5', linestyle=':', linewidth=0.8, alpha=0.8)
    ax.text(1.2, 1.1, 'k = 1.2', fontsize=7, color='0.4')
    ax.text(1.2, 1.9, 'k = 2.0', fontsize=7, color='0.4')
    ax.text(16.5, 3.4, 'highly asymmetric', fontsize=8, ha='center', color='0.3',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='0.7', alpha=0.7))

    ax.set_xlabel('Weber number $\\mathrm{We}$')
    ax.set_ylabel('Asymmetry ratio $k_{max}$')
    ax.set_xlim(0, 23)
    ax.set_ylim(0.8, 3.7)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_title('(a) $k$ vs $\\mathrm{We}$ ($R^*=1.0$, $\\alpha_{geo}=1.5$, $N=150$)', fontsize=11)

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'fig1_we_sweep.pdf'))
    fig.savefig(os.path.join(fig_dir, 'fig1_we_sweep.png'))
    plt.close(fig)
    print('  fig1_we_sweep.pdf OK')

# ============================================================
# Figure 2: k vs R* at We=7.9
# ============================================================
def fig2_curvature():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Data: N=150, α=1.5, We=7.9 (validated R* sweep; R*=1.5/3.0 re-run 2026-08-05)
    R_vals = [0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    k_amp15 = [3.000, 3.148, 2.655, 2.290, 2.161, 1.410]

    # α=0 baseline (r5_phase3; R*=0.7 and R*=2.0 have NO alpha=0 run — left NaN)
    k_amp0 = [1.800, float('nan'), 1.541, 1.571, float('nan'), 1.293]

    # Panel (a): k vs R* with/without amp
    ax1.plot(R_vals, k_amp0, 's--', color=colors['gray'], label='$\\alpha_{geo}=0$', markersize=5)
    ax1.plot(R_vals, k_amp15, 'o-', color=colors['blue'], label='$\\alpha_{geo}=1.5$', zorder=5)

    # Fill between to show amp effect
    ax1.fill_between(R_vals, k_amp0, k_amp15, alpha=0.15, color=colors['blue'])

    ax1.set_xlabel('Curvature ratio $R^*=D/D_0$')
    ax1.set_ylabel('Asymmetry ratio $k_{max}$')
    ax1.set_xlim(0.3, 3.3)
    ax1.set_ylim(0.8, 3.3)
    ax1.legend(fontsize=9)
    ax1.set_title('(a) $k$ vs $R^*$ ($\\mathrm{We}=7.9$)', fontsize=11)

    # Panel (b): Δk (amp effect) vs R*
    dk = [k1 - k0 for k1, k0 in zip(k_amp15, k_amp0)]
    ax2.bar(R_vals, dk, width=0.4, color=colors['blue'], alpha=0.7, edgecolor='black', linewidth=0.5)

    ax2.set_xlabel('Curvature ratio $R^*=D/D_0$')
    ax2.set_ylabel('$\\Delta k$ ($\\alpha_{geo}=1.5$ vs $0$)')
    ax2.set_xlim(0.3, 3.3)
    ax2.set_ylim(0, 1.4)
    ax2.set_title('(b) Amplification effect $\\Delta k$', fontsize=11)

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'fig2_curvature.pdf'))
    fig.savefig(os.path.join(fig_dir, 'fig2_curvature.png'))
    plt.close(fig)
    print('  fig2_curvature.pdf OK')

# ============================================================
# Figure 3: Amp calibration (θ dependence)
# ============================================================
def fig3_amp_calibration():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Data: amp sweep at R*=1.0, We=7.9, N=150
    amp_vals = [0.0, 0.5, 1.0, 1.5]
    k_amp = [1.541, 2.290, 2.290, 2.655]

    # Panel (a): k vs amp
    ax1.plot(amp_vals, k_amp, 'o-', color=colors['blue'], zorder=5)
    ax1.axvline(x=1.5, color=colors['red'], linestyle=':', alpha=0.6)
    ax1.text(1.52, 2.85, 'recommended $\\alpha_{geo}=1.5$', fontsize=8, color=colors['red'],
             ha='left', va='top',
             bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='red', alpha=0.8))

    # Liu 2015 reference

    ax1.set_xlabel('Amplification factor $\\alpha_{geo}$')
    ax1.set_ylabel('Asymmetry ratio $k_{max}$')
    ax1.set_xlim(-0.1, 2.0)
    ax1.set_ylim(1.0, 3.0)
    ax1.set_title('(a) $k$ vs $\\alpha_{geo}$ ($R^*=1.0$, $\\mathrm{We}=7.9$)', fontsize=11)

    # Data: θ dependence (N=150, α=1.5 — overnight sweep)
    theta_vals = [90, 120, 140, 162]
    # alpha=0 has data ONLY at theta=162 (m1_amp_sweep); other angles NaN
    k_theta_amp0 = [float('nan'), float('nan'), float('nan'), 1.540]
    k_theta_amp15 = [1.970, 2.091, 2.586, 2.655]

    # Panel (b): k vs θ
    ax2.plot(theta_vals, k_theta_amp0, 's--', color='#555555', label='$\\alpha_{geo}=0$', markersize=6, linewidth=1.8)
    ax2.plot(theta_vals, k_theta_amp15, 'o-', color=colors['blue'], label='$\\alpha_{geo}=1.5$', zorder=5)

    # Mark threshold
    ax2.axvline(x=120, color=colors['red'], linestyle=':', alpha=0.6)
    ax2.text(122, 1.7, '$\\theta_{th}$', fontsize=9, color=colors['red'])

    ax2.set_xlabel('Contact angle $\\theta_{eq}$ (deg)')
    ax2.set_ylabel('Asymmetry ratio $k_{max}$')
    ax2.set_xlim(80, 175)
    ax2.set_ylim(1.0, 3.0)
    ax2.legend(fontsize=9)
    ax2.set_title('(b) $k$ vs $\\theta_{eq}$ ($R^*=1.0$, $\\mathrm{We}=7.9$)', fontsize=11)

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'fig3_amp_calibration.pdf'))
    fig.savefig(os.path.join(fig_dir, 'fig3_amp_calibration.png'))
    plt.close(fig)
    print('  fig3_amp_calibration.pdf OK')

# ============================================================
# Figure 4: Grid convergence + Resolution threshold
# ============================================================
def fig4_resolution():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Data: Grid convergence at amp=1.5, R*=1.0, We=7.9 (N=150 validated)
    N_vals = [80, 100, 120, 150, 180]
    k_amp15 = [2.355, 2.586, 2.586, 2.655, 2.655]
    k_inf = 2.66  # converged value (N>=150 plateau), internal benchmark
    # No error bars: run-to-run spread is the integer-extent quantization
    # half-width (~0.035 = (77/29 - 75/29)/2), not measurement uncertainty.

    # Panel (a): k vs N
    ax1.plot(N_vals, k_amp15, 'o-', color=colors['blue'],
             zorder=5, label='$\\alpha_{geo}=1.5$')

    # Converged line
    ax1.axhline(y=k_inf, color=colors['blue'], linestyle=':', alpha=0.4)
    ax1.text(82, k_inf + 0.04, 'converged $k_\\infty$', fontsize=8,
             color=colors['blue'], alpha=0.6)

    ax1.set_xlabel('Grid resolution $N$')
    ax1.set_ylabel('Asymmetry ratio $k_{max}$')
    ax1.set_xlim(60, 200)
    ax1.set_ylim(2.0, 3.0)
    ax1.legend(fontsize=9, loc='lower right')
    ax1.set_title('(a) Grid convergence ($R^*=1.0$, $\\mathrm{We}=7.9$)', fontsize=11)

    # Panel (b): % error vs converged value (internal benchmark)
    errors_pct = [(k / k_inf - 1) * 100 for k in k_amp15]
    bar_colors = ['#e74c3c' if e < 0 else '#e67e22' for e in errors_pct]
    bars = ax2.bar(N_vals, errors_pct, width=15, color=bar_colors, alpha=0.8,
                   edgecolor='black', linewidth=0.5)
    ax2.axhline(y=0, color='green', linestyle='--', linewidth=1.5, alpha=0.7,
                label='converged $k_\\infty$')
# ============================================================
# Figure 5: Full phase diagram heatmap
# ============================================================
def fig5_phase_diagram_heatmap():
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))

    # Complete phase diagram data (all 36 cells, N=150, alpha=1.5, theta=162;
    # gaps filled 2026-08-06: results/phase_gaps_fill.json + phase_cell_fill.json)
    We_vals = [3.0, 5.0, 7.9, 10.0, 15.0, 20.0]
    R_vals = [0.5, 0.7, 1.0, 1.5, 2.0, 3.0]

    # Build matrix (R x We) — full 36-cell table
    k_matrix = np.array([
        [2.152, 2.724, 3.000, 3.069, 2.758, 2.220],  # R*=0.5
        [1.857, 2.355, 3.148, 3.296, 3.000, 2.568],  # R*=0.7
        [1.615, 1.971, 2.655, 3.074, 3.296, 2.657],  # R*=1.0
        [1.488, 1.857, 2.290, 2.586, 2.455, 2.179],  # R*=1.5
        [1.341, 1.649, 2.161, 2.448, 2.394, 2.073],  # R*=2.0
        [1.041, 1.122, 1.410, 1.743, 2.290, 2.026],  # R*=3.0
    ], dtype=float)

    # Plot heatmap
    im = ax.imshow(k_matrix, cmap='YlOrRd', aspect='auto', vmin=1.0, vmax=3.5,
                   interpolation='nearest')

    # Add text annotations
    for i in range(len(R_vals)):
        for j in range(len(We_vals)):
            if not np.isnan(k_matrix[i, j]):
                color = 'white' if k_matrix[i, j] > 2.0 else 'black'
                ax.text(j, i, f'{k_matrix[i, j]:.2f}', ha='center', va='center',
                       fontsize=8, color=color, fontweight='bold')

    ax.set_xticks(range(len(We_vals)))
    ax.set_xticklabels([f'{w:.1f}' for w in We_vals], fontsize=9)
    ax.set_yticks(range(len(R_vals)))
    ax.set_yticklabels([f'{r:.1f}' for r in R_vals], fontsize=9)
    ax.set_xlabel('Weber number $\\mathrm{We}$')
    ax.set_ylabel('Curvature ratio $R^*=D/D_0$')

    cbar = fig.colorbar(im, ax=ax, label='$k_{max}$', shrink=0.8)
    cbar.ax.tick_params(labelsize=9)

    ax.set_title('Phase Diagram: $k_{max}$ ($\\alpha_{geo}=1.5$, $N=150$)', fontsize=11)

    fig.tight_layout()
    fig.subplots_adjust(top=0.88, bottom=0.10, left=0.12, right=0.88)
    fig.savefig(os.path.join(fig_dir, 'fig5_phase_diagram.pdf'))
    fig.savefig(os.path.join(fig_dir, 'fig5_phase_diagram.png'))
    plt.close(fig)
    print('  fig5_phase_diagram.pdf OK')

# ============================================================
# Figure 6: Conceptual schematic
# ============================================================
def fig6_schematic():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    # Panel (a): No correction (amp=0)
    ax = axes[0]
    theta = np.linspace(0, np.pi, 100)
    # Curved surface
    x_wall = np.cos(theta) * 0.8
    y_wall = np.sin(theta) * 0.8 - 0.3
    ax.plot(x_wall, y_wall, 'k-', linewidth=2)
    ax.fill_between(x_wall, y_wall, -1.5, color='lightgray', alpha=0.5)
    # Droplet (symmetric, small contact)
    x_drop = np.linspace(-0.4, 0.4, 50)
    y_drop = 0.1 + 0.3 * np.cos(np.pi * x_drop / 0.8)
    ax.plot(x_drop, y_drop, 'b-', linewidth=1.5)
    ax.fill_between(x_drop, y_drop, 0.1, color='blue', alpha=0.2)
    # Arrows showing AC sharpening
    for x in [-0.2, 0, 0.2]:
        ax.annotate('', xy=(x, 0.25), xytext=(x, 0.15),
                   arrowprops=dict(arrowstyle='->', color='red', lw=1.2))
    ax.text(0, 0.4, '$\\alpha_{geo}=0$', fontsize=10, ha='center', fontweight='bold')
    ax.text(0, -0.05, 'AC sharpening\nresists correction', fontsize=8, ha='center',
           color='0.3', style='italic')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.8, 0.6)
    ax.set_aspect('equal')
    ax.set_title('(a) Without amplification', fontsize=10)
    ax.axis('off')

    # Panel (b): With correction (amp=1.5)
    ax = axes[1]
    ax.plot(x_wall, y_wall, 'k-', linewidth=2)
    ax.fill_between(x_wall, y_wall, -1.5, color='lightgray', alpha=0.5)
    # Droplet (asymmetric, larger contact)
    x_drop2 = np.linspace(-0.5, 0.5, 50)
    y_drop2 = 0.05 + 0.25 * np.cos(np.pi * x_drop2 / 1.0)
    ax.plot(x_drop2, y_drop2, 'b-', linewidth=1.5)
    ax.fill_between(x_drop2, y_drop2, 0.05, color='blue', alpha=0.2)
    # Arrows showing amplified correction
    for x in [-0.2, 0, 0.2]:
        ax.annotate('', xy=(x, 0.2), xytext=(x, 0.08),
                   arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
    ax.text(0, 0.4, '$\\alpha_{geo}=1.5$', fontsize=10, ha='center', fontweight='bold')
    ax.text(0, -0.05, 'Amplified correction\novercomes resistance', fontsize=8, ha='center',
           color='0.3', style='italic')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.8, 0.6)
    ax.set_aspect('equal')
    ax.set_title('(b) With amplification', fontsize=10)
    ax.axis('off')

    # Panel (c): Phase diagram sketch
    ax = axes[2]
    We = np.linspace(0, 35, 100)
    k_theory = 1.0 + 1.5 * np.exp(-((We - 15) / 8)**2)
    ax.plot(We, k_theory, 'b-', linewidth=2)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8)
    ax.fill_between(We, 1.0, k_theory, alpha=0.15, color='blue')

    # Regime labels (B&W-friendly: use hatching patterns)
    ax.text(7, 1.15, 'symmetric', fontsize=8, color='0.3', ha='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='0.5', alpha=0.8))
    ax.text(15, 2.3, 'peak', fontsize=8, color='0.0', ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='0.0', alpha=0.8))
    ax.text(28, 1.3, 'declining', fontsize=8, color='0.4', ha='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='0.6', alpha=0.8))

    ax.set_xlabel('Weber number $\\mathrm{We}$')
    ax.set_ylabel('$k_{max}$')
    ax.set_xlim(0, 33)
    ax.set_ylim(0.8, 2.6)
    ax.set_title('(c) Regime diagram', fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'fig6_schematic.pdf'))
    fig.savefig(os.path.join(fig_dir, 'fig6_schematic.png'))
    plt.close(fig)
    print('  fig6_schematic.pdf OK')

# ============================================================
# Run all
# ============================================================
if __name__ == '__main__':
    print('Generating figures...')
    fig1_phase_diagram()
    fig2_curvature()
    fig3_amp_calibration()
    fig4_resolution()
    fig5_phase_diagram_heatmap()
    fig6_schematic()
    print(f'\nAll figures saved to {fig_dir}')
