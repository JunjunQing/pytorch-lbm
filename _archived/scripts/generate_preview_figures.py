#!/usr/bin/env python3
"""Quick preview figures from collected simulation data."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
OUT = RESULTS / 'preview_figures'
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'figure.dpi': 100,
    'savefig.dpi': 150,
})

# Color palette
COLORS = {
    'flat': '#0072B2', 'ridge': '#E69F00',
    'convex': '#009E73', 'concave': '#D55E00',
}
CN = {'flat': '平面', 'ridge': '脊状', 'convex': '凸面', 'concave': '凹面'}
MORPH_COLORS = {
    'bulging': '#D55E00', 'scalloped': '#E69F00',
    'uniform': '#009E73', 'isolated': '#0072B2',
}


def fig4_morphology_regime():
    """Fig: Line morphology regime map — p/D₀ vs substrate."""
    d = json.load(open(RESULTS / 'chapter4_line_results.json'))
    # Filter: single layer, We=7.9, theta=162, standard n_drops
    base = [r for r in d if r.get('n_layers', 1) == 1
            and r.get('We') == 7.9 and r.get('theta_eq') == 162.0
            and r.get('n_drops') == 5 and r.get('R_star') in (None, 1.0)]

    fig, ax = plt.subplots(figsize=(8, 4))
    substrates = ['flat', 'ridge', 'convex', 'concave']
    for i, sub in enumerate(substrates):
        cases = sorted([r for r in base if r['substrate_type'] == sub],
                       key=lambda r: r['p_ratio'])
        for r in cases:
            color = MORPH_COLORS.get(r['morphology'], 'gray')
            ax.scatter(r['p_ratio'], i, c=color, s=200, edgecolors='k',
                       linewidth=0.8, zorder=3, marker='o')
            ax.annotate(f"{r['w_mean']:.0f}",
                        (r['p_ratio'], i), textcoords='offset points',
                        xytext=(0, 12), ha='center', fontsize=7)

    ax.set_yticks(range(len(substrates)))
    ax.set_yticklabels([CN[s] for s in substrates])
    ax.set_xlabel('$p / D_0$')
    ax.set_title('第四章 线材形貌相图 ($We$=7.9, $\\theta$=162°)')
    ax.set_xlim(0.1, 1.8)

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                      markeredgecolor='k', markersize=10, label=m)
               for m, c in MORPH_COLORS.items()]
    ax.legend(handles=handles, loc='upper right', fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / 'fig4_morphology_regime.png')
    plt.close(fig)
    print(f'  Saved fig4_morphology_regime.png')


def fig4_spacing_width():
    """Fig: Mean line width vs p/D₀ for different substrates."""
    d = json.load(open(RESULTS / 'chapter4_line_results.json'))
    base = [r for r in d if r.get('n_layers', 1) == 1
            and r.get('We') == 7.9 and r.get('theta_eq') == 162.0
            and r.get('n_drops') == 5 and r.get('R_star') in (None, 1.0)]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for sub in ['flat', 'ridge', 'convex', 'concave']:
        cases = sorted([r for r in base if r['substrate_type'] == sub],
                       key=lambda r: r['p_ratio'])
        if not cases:
            continue
        p_vals = [r['p_ratio'] for r in cases]
        w_vals = [r['w_mean'] for r in cases]
        ax.plot(p_vals, w_vals, 'o-', color=COLORS[sub], label=CN[sub],
                linewidth=2, markersize=8, markeredgecolor='k', markeredgewidth=0.8)

    ax.set_xlabel('$p / D_0$')
    ax.set_ylabel('平均线宽 $\\bar{w}$ (格点)')
    ax.set_title('线宽 vs 液滴间距 ($We$=7.9, $\\theta$=162°)')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / 'fig4_spacing_width.png')
    plt.close(fig)
    print(f'  Saved fig4_spacing_width.png')


def fig4_we_effect():
    """Fig: Weber number effect on line morphology (ridge)."""
    d = json.load(open(RESULTS / 'chapter4_line_results.json'))
    we_cases = [r for r in d if r['label'].startswith('s11e')]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    if we_cases:
        we_cases.sort(key=lambda r: r['We'])
        we_vals = [r['We'] for r in we_cases]
        w_vals = [r['w_mean'] for r in we_cases]
        cv_vals = [r['cv'] for r in we_cases]
        morphs = [r['morphology'] for r in we_cases]

        ax1.plot(we_vals, w_vals, 'o-', color='#E69F00', linewidth=2,
                 markersize=9, markeredgecolor='k', markeredgewidth=0.8)
        ax1.set_xlabel('$We$')
        ax1.set_ylabel('平均线宽 $\\bar{w}$')
        ax1.set_title('线宽 vs $We$ (脊状, $p/D_0$=0.8)')
        ax1.grid(alpha=0.2)

        colors = [MORPH_COLORS.get(m, 'gray') for m in morphs]
        ax2.scatter(we_vals, cv_vals, c=colors, s=120, edgecolors='k', linewidth=0.8)
        for we, cv, m in zip(we_vals, cv_vals, morphs):
            ax2.annotate(m, (we, cv), textcoords='offset points',
                         xytext=(5, 5), fontsize=8)
        ax2.axhline(0.06, color='green', ls='--', alpha=0.5, label='uniform<0.06')
        ax2.axhline(0.15, color='orange', ls='--', alpha=0.5, label='scalloped<0.15')
        ax2.set_xlabel('$We$')
        ax2.set_ylabel('变异系数 CV')
        ax2.set_title('均匀性 vs $We$')
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(OUT / 'fig4_we_effect.png')
    plt.close(fig)
    print(f'  Saved fig4_we_effect.png')


def fig3_dual_spacing():
    """Fig: Dual droplet k vs d/D (supplementary + original)."""
    d = json.load(open(RESULTS / 'studies_9_10.json'))
    # Original s9b + supplementary s9f (ridge spacing scan)
    ridge_spacing = [r for r in d if 'ridge' in r.get('substrate_type', '')
                     and r.get('spacing_ratio') is not None
                     and r.get('max_k') is not None
                     and r.get('We') == 7.9
                     and r.get('theta_eq') == 162.0
                     and r.get('R_star') == 1.0]
    ridge_spacing.sort(key=lambda r: r['spacing_ratio'])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if ridge_spacing:
        d_vals = [r['spacing_ratio'] for r in ridge_spacing]
        k_vals = [r['max_k'] for r in ridge_spacing]
        coal = [r.get('coalescence_step') is not None for r in ridge_spacing]

        ax.plot(d_vals, k_vals, 'o-', color='#E69F00', linewidth=2,
                markersize=9, markeredgecolor='k', markeredgewidth=0.8)

        # Mark coalescence
        for r in ridge_spacing:
            marker = 's' if r.get('coalescence_step') else 'o'
            ax.scatter(r['spacing_ratio'], r['max_k'], marker=marker,
                       c='#E69F00', s=100, edgecolors='k', linewidth=0.8, zorder=3)

        # Annotate merge/separate
        for r in ridge_spacing:
            tag = f"merge@{r['coalescence_step']}" if r.get('coalescence_step') else 'separate'
            ax.annotate(tag, (r['spacing_ratio'], r['max_k']),
                        textcoords='offset points', xytext=(0, 12),
                        ha='center', fontsize=7)

    ax.axhline(1.0, color='gray', ls='--', alpha=0.4, label='$k$=1 (对称)')
    ax.set_xlabel('$d / D_0$')
    ax.set_ylabel('不对称因子 $k$')
    ax.set_title('双液滴不对称性 vs 间距 (脊状 $R^*$=1.0, $We$=7.9)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / 'fig3_dual_spacing.png')
    plt.close(fig)
    print(f'  Saved fig3_dual_spacing.png')


def fig4_theta_effect():
    """Fig: Contact angle effect on line width."""
    d = json.load(open(RESULTS / 'chapter4_line_results.json'))
    theta_cases = [r for r in d if r['label'].startswith('s11f')]
    theta_cases.sort(key=lambda r: r['theta_eq'])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if theta_cases:
        t_vals = [r['theta_eq'] for r in theta_cases]
        w_vals = [r['w_mean'] for r in theta_cases]
        cv_vals = [r['cv'] for r in theta_cases]

        ax.plot(t_vals, w_vals, 'o-', color='#0072B2', linewidth=2,
                markersize=9, markeredgecolor='k', markeredgewidth=0.8)
        ax.set_xlabel('接触角 $\\theta$ (°)')
        ax.set_ylabel('平均线宽 $\\bar{w}$', color='#0072B2')
        ax.tick_params(axis='y', labelcolor='#0072B2')

        ax2 = ax.twinx()
        ax2.plot(t_vals, cv_vals, 's--', color='#D55E00', linewidth=1.5,
                 markersize=8, markeredgecolor='k', markeredgewidth=0.8)
        ax2.set_ylabel('变异系数 CV', color='#D55E00')
        ax2.tick_params(axis='y', labelcolor='#D55E00')

    ax.set_title('接触角效应 (脊状, $p/D_0$=0.8, $We$=7.9)')
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / 'fig4_theta_effect.png')
    plt.close(fig)
    print(f'  Saved fig4_theta_effect.png')


if __name__ == '__main__':
    print('Generating preview figures...')
    fig4_morphology_regime()
    fig4_spacing_width()
    fig4_we_effect()
    fig3_dual_spacing()
    fig4_theta_effect()
    print(f'\nAll figures saved to {OUT}/')
