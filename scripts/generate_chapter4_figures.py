#!/usr/bin/env python3
"""Generate all Chapter 4 thesis figures from line formation + supplementary data.

Each PNG file contains exactly one figure.
Font rules: Chinese in 宋体 (Song), English/numbers/symbols in Times New Roman.

Usage: python3 scripts/generate_chapter4_figures.py [--skip-existing] [--only 4.1 4.3]
"""
import sys
import os
import json
import argparse
import time

import numpy as np
from scipy.ndimage import zoom as ndimage_zoom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

# ============================================================================
# Style Configuration — 宋体 + Times New Roman
# ============================================================================

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['AR PL SungtiL GB', 'Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'mathtext.fontset': 'stix',
    'axes.unicode_minus': False,
})

# Colors — consistent with Chapter 3
PRIMARY = '#0072B2'
COLORS = {
    'flat': '#0072B2', 'ridge': '#E69F00',
    'convex': '#009E73', 'concave': '#D55E00',
}
MARKERS = {'flat': 's', 'ridge': 'o', 'convex': '^', 'concave': 'D'}
SUBSTRATE_CN = {
    'flat': '平面', 'ridge': '脊状',
    'convex': '凸面', 'concave': '凹面',
}
MORPH_COLORS = {
    'bulging': '#D55E00', 'scalloped': '#E69F00',
    'uniform': '#009E73', 'isolated': '#0072B2',
}
MORPH_CN = {
    'bulging': '膨胀形', 'scalloped': '扇贝形',
    'uniform': '均匀形', 'isolated': '孤立形',
}
LINE_W = 2
MARKER_SZ = 9
MARKER_EDGE_W = 0.8
MARKER_EDGE = 'black'
REF_COLOR = 'gray'
REF_STYLE = '--'
REF_ALPHA = 0.4
GRID_ALPHA = 0.2

# Custom colormap for phase field
_PHASE_COLORS = ['#FFFFFF', '#D6EAF8', '#85C1E9', '#3498DB', '#2471A3', '#1A5276']
CMAP_PHASE = LinearSegmentedColormap.from_list('phase', _PHASE_COLORS, N=256)
SOLID_COLOR = '#7F8C8D'

# Physical constants
D0 = 45.0
U0 = 0.05

# Directories
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / 'results'
FIG_DIR = RESULTS_DIR / 'thesis_figures'
SNAP_DIR = RESULTS_DIR / 'chapter4_snapshots'
THESIS_SNAP_DIR = RESULTS_DIR / 'thesis_snapshots'
FIG_DIR.mkdir(exist_ok=True, parents=True)


def save_fig(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    size_kb = os.path.getsize(path) / 1024
    print(f"  Saved {name} ({size_kb:.0f} KB)")


def step_to_tstar(step):
    return step * U0 / D0


# ============================================================================
# Data Loading
# ============================================================================

def load_ch4_data():
    """Load chapter 4 line formation results."""
    fpath = RESULTS_DIR / 'chapter4_line_results.json'
    if not fpath.exists():
        print("  WARNING: chapter4_line_results.json not found")
        return []
    with open(fpath) as f:
        return json.load(f)


def load_s910_data():
    """Load supplementary dual/multi droplet results."""
    fpath = RESULTS_DIR / 'studies_9_10.json'
    if not fpath.exists():
        print("  WARNING: studies_9_10.json not found")
        return []
    with open(fpath) as f:
        return json.load(f)


def filter_by(records, prefix=None, substrate=None, we=None, theta=None,
              p_ratio=None, rstar=None, n_drops=None, n_layers=None):
    """Filter records by multiple criteria."""
    out = records
    if prefix:
        out = [r for r in out if r['label'].startswith(prefix)]
    if substrate:
        out = [r for r in out if r.get('substrate_type') == substrate]
    if we is not None:
        out = [r for r in out if r.get('We') == we]
    if theta is not None:
        out = [r for r in out if r.get('theta_eq') == theta]
    if p_ratio is not None:
        out = [r for r in out if r.get('p_ratio') == p_ratio]
    if rstar is not None:
        out = [r for r in out if r.get('R_star') == rstar]
    if n_drops is not None:
        out = [r for r in out if r.get('n_drops') == n_drops]
    if n_layers is not None:
        out = [r for r in out if r.get('n_layers', 1) == n_layers]
    return out


# ============================================================================
# Fig 4.1: 形貌相图 (p/D₀ vs substrate)
# ============================================================================

def fig_4_01(ch4):
    """Line morphology regime map — p/D₀ vs substrate type."""
    base = [r for r in ch4
            if r.get('n_layers', 1) == 1 and r.get('We') == 7.9
            and r.get('theta_eq') == 162.0 and r.get('n_drops') == 5
            and r.get('R_star') in (None, 1.0)]
    if not base:
        print("  SKIP fig_4_01: no base case data")
        return

    substrates = ['flat', 'ridge', 'convex', 'concave']
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, sub in enumerate(substrates):
        cases = sorted([r for r in base if r['substrate_type'] == sub],
                       key=lambda r: r['p_ratio'])
        for r in cases:
            color = MORPH_COLORS.get(r['morphology'], 'gray')
            ax.scatter(r['p_ratio'], i, c=color, s=250, edgecolors='k',
                       linewidth=MARKER_EDGE_W, zorder=3, marker='o')
            ax.annotate(f"$\\bar{{w}}$={r['w_mean']:.0f}",
                        (r['p_ratio'], i), textcoords='offset points',
                        xytext=(0, 14), ha='center', fontsize=8)

    ax.set_yticks(range(len(substrates)))
    ax.set_yticklabels([SUBSTRATE_CN[s] for s in substrates])
    ax.set_xlabel('归一化液滴间距 $p / D_0$')
    ax.set_title('线材形貌相图 ($We$=7.9, $\\theta$=162°, $n$=5)')
    ax.set_xlim(0.1, 2.0)

    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                       markeredgecolor='k', markersize=10, label=MORPH_CN.get(m, m))
               for m, c in MORPH_COLORS.items()]
    ax.legend(handles=handles, loc='upper right', fontsize=9)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.set_ylim(-0.5, len(substrates) - 0.5)
    save_fig(fig, 'fig4_01.png')


# ============================================================================
# Fig 4.2: 线宽 vs p/D₀
# ============================================================================

def fig_4_02(ch4):
    """Mean line width vs p/D₀ for different substrates."""
    base = [r for r in ch4
            if r.get('n_layers', 1) == 1 and r.get('We') == 7.9
            and r.get('theta_eq') == 162.0 and r.get('n_drops') == 5
            and r.get('R_star') in (None, 1.0)]
    if not base:
        print("  SKIP fig_4_02: no data")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for sub in ['flat', 'ridge', 'convex', 'concave']:
        cases = sorted([r for r in base if r['substrate_type'] == sub],
                       key=lambda r: r['p_ratio'])
        if not cases:
            continue
        p_vals = [r['p_ratio'] for r in cases]
        w_vals = [r['w_mean'] for r in cases]
        ax.plot(p_vals, w_vals, marker=MARKERS.get(sub, 'o'), markersize=MARKER_SZ,
                linewidth=LINE_W, color=COLORS[sub], markeredgecolor=MARKER_EDGE,
                markeredgewidth=MARKER_EDGE_W, label=SUBSTRATE_CN[sub])
        for p, w in zip(p_vals, w_vals):
            ax.annotate(f'{w:.0f}', (p, w), textcoords='offset points',
                        xytext=(0, 10), ha='center', fontsize=8)

    ax.set_xlabel('归一化液滴间距 $p / D_0$')
    ax.set_ylabel('平均线宽 $\\bar{w}$（格点单位）')
    ax.set_title('线宽随液滴间距变化 ($We$=7.9, $\\theta$=162°, $n$=5)')
    ax.legend()
    ax.grid(True, alpha=GRID_ALPHA)
    ax.margins(y=0.12)
    save_fig(fig, 'fig4_02.png')


# ============================================================================
# Fig 4.3: Weber数效应
# ============================================================================

def fig_4_03(ch4):
    """Weber number effect on line width and uniformity (ridge, p/D=0.8)."""
    we_cases = sorted([r for r in ch4 if r['label'].startswith('s11e')],
                      key=lambda r: r['We'])
    if not we_cases:
        print("  SKIP fig_4_03: no We sweep data")
        return

    fig, ax1 = plt.subplots(figsize=(7, 5))
    we_vals = [r['We'] for r in we_cases]
    w_vals = [r['w_mean'] for r in we_cases]
    cv_vals = [r['cv'] for r in we_cases]

    ax1.plot(we_vals, w_vals, 'o-', color=PRIMARY, markersize=MARKER_SZ,
             linewidth=LINE_W, markeredgecolor=MARKER_EDGE, markeredgewidth=MARKER_EDGE_W,
             label='平均线宽 $\\bar{w}$')
    ax1.set_xlabel('韦伯数 $We$')
    ax1.set_ylabel('平均线宽 $\\bar{w}$（格点单位）', color=PRIMARY)
    ax1.tick_params(axis='y', labelcolor=PRIMARY)

    ax2 = ax1.twinx()
    ax2.plot(we_vals, cv_vals, 's--', color='#D55E00', markersize=8,
             linewidth=1.5, markeredgecolor=MARKER_EDGE, markeredgewidth=MARKER_EDGE_W,
             label='变异系数 CV')
    ax2.set_ylabel('变异系数 CV', color='#D55E00')
    ax2.tick_params(axis='y', labelcolor='#D55E00')

    # Morphology annotations
    for r in we_cases:
        morph = r['morphology']
        ax1.annotate(MORPH_CN.get(morph, morph), (r['We'], r['w_mean']),
                     textcoords='offset points', xytext=(8, 5), fontsize=7,
                     color=MORPH_COLORS.get(morph, 'gray'))

    ax1.set_title('韦伯数对线形成的影响（脊状 $p/D_0$=0.8, $\\theta$=162°）')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    ax1.grid(True, alpha=GRID_ALPHA)
    fig.tight_layout()
    save_fig(fig, 'fig4_03.png')


# ============================================================================
# Fig 4.4: 接触角效应
# ============================================================================

def fig_4_04(ch4):
    """Contact angle effect on line width (ridge, p/D=0.8, We=7.9)."""
    theta_cases = sorted([r for r in ch4 if r['label'].startswith('s11f')],
                         key=lambda r: r['theta_eq'])
    if not theta_cases:
        print("  SKIP fig_4_04: no theta sweep data")
        return

    fig, ax1 = plt.subplots(figsize=(7, 5))
    t_vals = [r['theta_eq'] for r in theta_cases]
    w_vals = [r['w_mean'] for r in theta_cases]
    cv_vals = [r['cv'] for r in theta_cases]

    ax1.plot(t_vals, w_vals, 'o-', color=PRIMARY, markersize=MARKER_SZ,
             linewidth=LINE_W, markeredgecolor=MARKER_EDGE, markeredgewidth=MARKER_EDGE_W,
             label='平均线宽 $\\bar{w}$')
    ax1.set_xlabel('接触角 $\\theta$ (°)')
    ax1.set_ylabel('平均线宽 $\\bar{w}$（格点单位）', color=PRIMARY)
    ax1.tick_params(axis='y', labelcolor=PRIMARY)

    ax2 = ax1.twinx()
    ax2.plot(t_vals, cv_vals, 's--', color='#D55E00', markersize=8,
             linewidth=1.5, markeredgecolor=MARKER_EDGE, markeredgewidth=MARKER_EDGE_W,
             label='变异系数 CV')
    ax2.set_ylabel('变异系数 CV', color='#D55E00')
    ax2.tick_params(axis='y', labelcolor='#D55E00')

    for t, w in zip(t_vals, w_vals):
        ax1.annotate(f'{w:.0f}', (t, w), textcoords='offset points',
                     xytext=(0, 10), ha='center', fontsize=8)

    ax1.set_title('接触角对线形成的影响（脊状 $p/D_0$=0.8, $We$=7.9）')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax1.grid(True, alpha=GRID_ALPHA)
    fig.tight_layout()
    save_fig(fig, 'fig4_04.png')


# ============================================================================
# Fig 4.5: 曲率比R*效应
# ============================================================================

def fig_4_05(ch4):
    """Curvature ratio R* effect on line width (ridge, We=7.9, p/D=0.8)."""
    rstar_cases = sorted([r for r in ch4 if r['label'].startswith('s11g')],
                         key=lambda r: r.get('R_star', 1.0))
    if not rstar_cases:
        print("  SKIP fig_4_05: no R* sweep data")
        return

    fig, ax1 = plt.subplots(figsize=(7, 5))
    rs = [r.get('R_star', 1.0) for r in rstar_cases]
    ws = [r['w_mean'] for r in rstar_cases]
    cvs = [r['cv'] for r in rstar_cases]

    ax1.plot(rs, ws, 'o-', color=PRIMARY, markersize=MARKER_SZ,
             linewidth=LINE_W, markeredgecolor=MARKER_EDGE, markeredgewidth=MARKER_EDGE_W,
             label='平均线宽 $\\bar{w}$')
    ax1.set_xlabel('曲率比 $R^*$')
    ax1.set_ylabel('平均线宽 $\\bar{w}$（格点单位）', color=PRIMARY)
    ax1.tick_params(axis='y', labelcolor=PRIMARY)

    ax2 = ax1.twinx()
    ax2.plot(rs, cvs, 's--', color='#D55E00', markersize=8,
             linewidth=1.5, markeredgecolor=MARKER_EDGE, markeredgewidth=MARKER_EDGE_W,
             label='变异系数 CV')
    ax2.set_ylabel('变异系数 CV', color='#D55E00')
    ax2.tick_params(axis='y', labelcolor='#D55E00')

    for r_s, w, m in zip(rs, ws, [r['morphology'] for r in rstar_cases]):
        ax1.annotate(MORPH_CN.get(m, m), (r_s, w), textcoords='offset points',
                     xytext=(8, 5), fontsize=7, color=MORPH_COLORS.get(m, 'gray'))

    ax1.set_title('曲率比对线形成的影响（脊状 $p/D_0$=0.8, $We$=7.9）')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax1.grid(True, alpha=GRID_ALPHA)
    fig.tight_layout()
    save_fig(fig, 'fig4_05.png')


# ============================================================================
# Fig 4.6: 液滴数量效应
# ============================================================================

def fig_4_06(ch4):
    """Number of drops effect on line width and uniformity."""
    # s11a ridge has n_drops=5, s11h has n_drops=3,7,10
    base_ridge = [r for r in ch4
                  if r['substrate_type'] == 'ridge' and r.get('We') == 7.9
                  and r.get('theta_eq') == 162.0 and r.get('p_ratio') == 0.8
                  and r.get('R_star') in (None, 1.0) and r.get('n_layers', 1) == 1]
    # Include s11h (n_drops sweep)
    n_cases = sorted(base_ridge, key=lambda r: r.get('n_drops', 5))
    if not n_cases:
        print("  SKIP fig_4_06: no n_drops data")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ns = [r.get('n_drops', 5) for r in n_cases]
    ws = [r['w_mean'] for r in n_cases]
    cvs = [r['cv'] for r in n_cases]

    ax.plot(ns, ws, 'o-', color=PRIMARY, markersize=MARKER_SZ,
            linewidth=LINE_W, markeredgecolor=MARKER_EDGE, markeredgewidth=MARKER_EDGE_W)
    for n, w in zip(ns, ws):
        ax.annotate(f'{w:.0f}', (n, w), textcoords='offset points',
                    xytext=(0, 10), ha='center', fontsize=9)

    ax.set_xlabel('液滴数量 $n$')
    ax.set_ylabel('平均线宽 $\\bar{w}$（格点单位）')
    ax.set_title('液滴数量对线宽的影响（脊状 $p/D_0$=0.8, $We$=7.9）')
    ax.grid(True, alpha=GRID_ALPHA)
    ax.margins(y=0.12)
    save_fig(fig, 'fig4_06.png')


# ============================================================================
# Fig 4.7: 多层对比
# ============================================================================

def fig_4_07(ch4):
    """Multi-layer comparison: single vs aligned vs staggered."""
    # Single layer reference
    single_flat = [r for r in ch4
                   if r['substrate_type'] == 'flat' and r.get('We') == 7.9
                   and r.get('theta_eq') == 162.0 and r.get('p_ratio') == 0.8
                   and r.get('n_layers', 1) == 1 and r.get('n_drops') == 5
                   and r.get('R_star') in (None, 1.0)]
    single_ridge = [r for r in ch4
                    if r['substrate_type'] == 'ridge' and r.get('We') == 7.9
                    and r.get('theta_eq') == 162.0 and r.get('p_ratio') == 0.8
                    and r.get('n_layers', 1) == 1 and r.get('n_drops') == 5
                    and r.get('R_star') in (None, 1.0)]
    multi = [r for r in ch4 if r.get('n_layers', 1) > 1]
    if not multi:
        print("  SKIP fig_4_07: no multi-layer data")
        return

    categories = []
    widths = []
    cvs = []
    colors_bar = []

    # Single layer flat
    if single_flat:
        categories.append('平面\n(单层)')
        widths.append(single_flat[0]['w_mean'])
        cvs.append(single_flat[0]['cv'])
        colors_bar.append(COLORS['flat'])

    # Single layer ridge
    if single_ridge:
        categories.append('脊状\n(单层)')
        widths.append(single_ridge[0]['w_mean'])
        cvs.append(single_ridge[0]['cv'])
        colors_bar.append(COLORS['ridge'])

    # Multi-layer cases
    for r in sorted(multi, key=lambda x: x['label']):
        sub = r['substrate_type']
        offset = r.get('stagger_offset', 0)
        tag = '对齐' if offset == 0 else '交错'
        categories.append(f'{SUBSTRATE_CN[sub]}\n(双层{tag})')
        widths.append(r['w_mean'])
        cvs.append(r['cv'])
        colors_bar.append(COLORS[sub])

    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(categories))
    bars = ax1.bar(x, widths, color=colors_bar, edgecolor='black', linewidth=0.8, alpha=0.8)
    for bar, w in zip(bars, widths):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f'{w:.0f}', ha='center', fontsize=9, fontweight='bold')

    ax2 = ax1.twinx()
    ax2.plot(x, cvs, 'D-', color='#CC79A7', markersize=8, linewidth=1.5,
             markeredgecolor='black', markeredgewidth=0.8, label='CV')
    ax2.set_ylabel('变异系数 CV', color='#CC79A7')
    ax2.tick_params(axis='y', labelcolor='#CC79A7')

    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=9)
    ax1.set_ylabel('平均线宽 $\\bar{w}$（格点单位）')
    ax1.set_title('单层与多层沉积对比 ($We$=7.9, $p/D_0$=0.8)')
    ax1.grid(True, alpha=GRID_ALPHA, axis='y')
    ax2.legend(loc='upper left')
    fig.tight_layout()
    save_fig(fig, 'fig4_07.png')


# ============================================================================
# Fig 4.8: 双液滴不对称性 vs 间距
# ============================================================================

def fig_4_08(s910):
    """Dual droplet asymmetry k vs d/D (ridge, R*=1.0)."""
    s9b = sorted([r for r in s910 if r['label'].startswith('s9b')
                  and r.get('spacing_ratio') is not None],
                 key=lambda r: r['spacing_ratio'])
    if not s9b:
        print("  SKIP fig_4_08: no dual spacing data")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ds = [r['spacing_ratio'] for r in s9b]
    ks = [r['max_k'] for r in s9b]

    ax.plot(ds, ks, 'o-', color=PRIMARY, markersize=MARKER_SZ,
            linewidth=LINE_W, markeredgecolor=MARKER_EDGE, markeredgewidth=MARKER_EDGE_W)
    for d, k, r in zip(ds, ks, s9b):
        coal = '聚结' if r.get('coalescence_step') else '分离'
        ax.annotate(f'k={k:.2f}\n({coal})', (d, k), textcoords='offset points',
                    xytext=(0, 14), ha='center', fontsize=8)

    ax.axhline(y=1.0, color=REF_COLOR, linestyle=REF_STYLE, alpha=REF_ALPHA,
               label='$k$=1（对称）')
    ax.set_xlabel('归一化间距 $d / D_0$')
    ax.set_ylabel('不对称因子 $k = D_x / D_y$')
    ax.set_title('双液滴不对称性随间距变化（脊状 $R^*$=1.0, $We$=7.9）')
    ax.legend()
    ax.grid(True, alpha=GRID_ALPHA)
    ax.margins(y=0.12)
    save_fig(fig, 'fig4_08.png')


# ============================================================================
# Fig 4.9: 多液滴累积效应
# ============================================================================

def fig_4_09(s910):
    """Multi-droplet cumulative spreading: Dx, Dy vs n."""
    s10c = sorted([r for r in s910 if r['label'].startswith('s10c')],
                  key=lambda r: r.get('n_drops', 0))
    # Also get s10a ridge for n=2
    s10a_ridge = [r for r in s910 if r['label'].startswith('s10a')
                  and r['substrate_type'] == 'ridge']
    all_cases = s10c[:]
    if s10a_ridge:
        all_cases.append(s10a_ridge[0])
    all_cases.sort(key=lambda r: r.get('n_drops', 0))

    if not all_cases:
        print("  SKIP fig_4_09: no multi-drop n data")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ns = [r.get('n_drops', 0) for r in all_cases]
    dxs = [r.get('max_Dx', r.get('Dx', 0)) for r in all_cases]
    dys = [r.get('max_Dy', r.get('Dy', 0)) for r in all_cases]

    ax.plot(ns, dxs, 'o-', color=PRIMARY, markersize=MARKER_SZ,
            linewidth=LINE_W, markeredgecolor=MARKER_EDGE, markeredgewidth=MARKER_EDGE_W,
            label='$D_x$（横向铺展）')
    ax.plot(ns, dys, 's--', color=COLORS['ridge'], markersize=MARKER_SZ,
            linewidth=LINE_W, markeredgecolor=MARKER_EDGE, markeredgewidth=MARKER_EDGE_W,
            label='$D_y$（纵向铺展）')

    for n, dx, dy in zip(ns, dxs, dys):
        ax.annotate(f'{dx:.0f}', (n, dx), textcoords='offset points',
                    xytext=(0, 10), ha='center', fontsize=8, color=PRIMARY)
        ax.annotate(f'{dy:.0f}', (n, dy), textcoords='offset points',
                    xytext=(0, -14), ha='center', fontsize=8, color=COLORS['ridge'])

    if len(ns) >= 2:
        z = np.polyfit(ns, dxs, 1)
        n_fit = np.linspace(min(ns) - 0.5, max(ns) + 0.5, 50)
        ax.plot(n_fit, np.polyval(z, n_fit), ':', color=REF_COLOR, alpha=0.5,
                label=f'线性拟合: {z[0]:.0f}n + {z[1]:.0f}')

    ax.set_xlabel('液滴数量 $n$')
    ax.set_ylabel('铺展量（格点单位）')
    ax.set_title('多液滴累积铺展效应（脊状 $R^*$=1.0, $\\Delta t^*$=1.0）')
    ax.legend()
    ax.grid(True, alpha=GRID_ALPHA)
    ax.margins(y=0.12)
    save_fig(fig, 'fig4_09.png')


# ============================================================================
# Fig 4.10: 双液滴基底对比
# ============================================================================

def fig_4_10(s910):
    """Dual droplet: substrate type comparison."""
    s9a = [r for r in s910 if r['label'].startswith('s9a')]
    if not s9a:
        print("  SKIP fig_4_10: no s9a data")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    substrates = ['flat', 'ridge', 'convex', 'concave']
    ks = []
    labels = []
    colors = []
    coal_tags = []
    for sub in substrates:
        found = [r for r in s9a if r['substrate_type'] == sub]
        if found:
            r = found[0]
            ks.append(r['max_k'])
            labels.append(SUBSTRATE_CN[sub])
            colors.append(COLORS[sub])
            coal_tags.append('聚结' if r.get('coalescence_step') else '分离')

    x = np.arange(len(labels))
    bars = ax.bar(x, ks, color=colors, edgecolor='black', linewidth=0.8)
    for bar, k, ct in zip(bars, ks, coal_tags):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{k:.2f}\n({ct})', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(y=1.0, color=REF_COLOR, linestyle=REF_STYLE, alpha=REF_ALPHA,
               label='$k$=1（对称）')
    ax.set_ylabel('不对称因子 $k = D_x / D_y$')
    ax.set_title('双液滴聚结——基底类型对比 ($d/D_0$=1.2, $We$=7.9)')
    ax.legend()
    ax.grid(True, alpha=GRID_ALPHA, axis='y')
    ax.set_ylim(0, max(ks) * 1.3)
    save_fig(fig, 'fig4_10.png')


# ============================================================================
# Fig 4.11: 多液滴We效应
# ============================================================================

def fig_4_11(s910):
    """Multi-droplet We effect on spreading."""
    s10d = sorted([r for r in s910 if r['label'].startswith('s10d')],
                  key=lambda r: r['We'])
    if not s10d:
        print("  SKIP fig_4_11: no s10d data")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    wes = [r['We'] for r in s10d]
    dxs = [r.get('max_Dx', r.get('Dx', 0)) for r in s10d]
    dys = [r.get('max_Dy', r.get('Dy', 0)) for r in s10d]

    ax.plot(wes, dxs, 'o-', color=PRIMARY, markersize=MARKER_SZ,
            linewidth=LINE_W, markeredgecolor=MARKER_EDGE, markeredgewidth=MARKER_EDGE_W,
            label='$D_x$（横向）')
    ax.plot(wes, dys, 's--', color=COLORS['ridge'], markersize=MARKER_SZ,
            linewidth=LINE_W, markeredgecolor=MARKER_EDGE, markeredgewidth=MARKER_EDGE_W,
            label='$D_y$（纵向）')
    for w, dx, dy in zip(wes, dxs, dys):
        ax.annotate(f'{dx:.0f}', (w, dx), textcoords='offset points',
                    xytext=(0, 10), ha='center', fontsize=8, color=PRIMARY)
        ax.annotate(f'{dy:.0f}', (w, dy), textcoords='offset points',
                    xytext=(0, -14), ha='center', fontsize=8, color=COLORS['ridge'])

    ax.set_xlabel('韦伯数 $We$')
    ax.set_ylabel('铺展量（格点单位）')
    ax.set_title('多液滴沉积——韦伯数影响（脊状 $R^*$=1.0, $n$=2）')
    ax.legend()
    ax.grid(True, alpha=GRID_ALPHA)
    ax.margins(y=0.12)
    save_fig(fig, 'fig4_11.png')


# ============================================================================
# Fig 4.12: 线形成快照（形貌对比）
# ============================================================================

def _upsample_field(field, factor=4):
    return ndimage_zoom(field, factor, order=1)


def _find_closest_step(snap_dir, prefix, target_step):
    candidates = sorted(snap_dir.glob(f'{prefix}_step*.npz'))
    if not candidates:
        return None
    return min(candidates, key=lambda f: abs(int(f.name.split('_step')[1].split('.')[0]) - target_step))


def _plot_droplet_smooth(ax, xz, solid_xz, title='', upsampling=4):
    nx, nz = xz.shape
    if upsampling > 1:
        xz_up = _upsample_field(xz.T, upsampling)
        ext_up = [0, nx, 0, nz]
    else:
        xz_up = xz.T
        ext_up = [0, nx, 0, nz]
    im = ax.imshow(xz_up, origin='lower', cmap=CMAP_PHASE, vmin=0, vmax=1,
                   aspect='equal', extent=ext_up, interpolation='bilinear')
    if solid_xz is not None:
        if upsampling > 1:
            solid_up = _upsample_field(solid_xz.T.astype(float), upsampling)
            solid_mask = solid_up > 0.5
        else:
            solid_mask = solid_xz.T
        masked = np.ma.masked_where(~solid_mask, np.ones_like(solid_mask, dtype=float))
        ax.imshow(masked, origin='lower',
                  cmap=LinearSegmentedColormap.from_list('s', [SOLID_COLOR, SOLID_COLOR], N=2),
                  vmin=0, vmax=1, aspect='equal', alpha=0.9, extent=ext_up)
    try:
        ax.contour(xz_up, levels=[0.5], colors='#E74C3C', linewidths=1.2,
                   origin='lower', extent=ext_up)
    except ValueError:
        pass
    if title:
        ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def fig_4_12():
    """Line formation morphology snapshots — substrate comparison at final state."""
    if not SNAP_DIR.exists():
        print("  SKIP fig_4_12: no chapter4_snapshots directory")
        return
    snap_files = list(SNAP_DIR.glob('*.npz'))
    if not snap_files:
        print("  SKIP fig_4_12: no snapshot files")
        return

    # Find final-step snapshots for each substrate at p/D=0.8
    prefixes = {}
    for f in snap_files:
        name = f.name.split('_step')[0]
        step = int(f.name.split('_step')[1].split('.')[0])
        if name not in prefixes or step > prefixes[name][1]:
            prefixes[name] = (f, step)

    # Substudy labels for p/D=0.8 single layer
    targets = {
        's11a_flat_p0.8': ('平面', COLORS['flat']),
        's11a_ridge_p0.8': ('脊状', COLORS['ridge']),
        's11b_convex_p0.8': ('凸面', COLORS['convex']),
        's11c_concave_p0.8': ('凹面', COLORS['concave']),
    }
    # Try alternate naming patterns
    alt_targets = {}
    for f, step in prefixes.values():
        name = f.name.split('_step')[0]
        # Check if it's a ch4 case with p=0.8
        for sub in ['flat', 'ridge', 'convex', 'concave']:
            if sub in name and 'p0.8' in name:
                alt_targets[name] = (sub, f, step)

    cases = []
    for prefix, (label, color) in targets.items():
        if prefix in prefixes:
            cases.append((label, color, prefixes[prefix][0]))
        # Try alternative pattern matching
        for name, (sub, f, step) in alt_targets.items():
            if sub in prefix:
                cases.append((label, color, f))
                break

    if not cases:
        # Fallback: just grab one snapshot per substrate type
        for sub in ['flat', 'ridge', 'convex', 'concave']:
            matching = [(f, s) for n, (f, s) in prefixes.items() if sub in n and 'p0.8' in n]
            if matching:
                best = max(matching, key=lambda x: x[1])
                cases.append((SUBSTRATE_CN[sub], COLORS[sub], best[0]))

    if not cases:
        print("  SKIP fig_4_12: no matching snapshots found")
        return

    n = len(cases)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 6))
    if n == 1:
        axes = [axes]
    im_ref = None
    for ax, (label, color, snap_file) in zip(axes, cases):
        data = dict(np.load(snap_file, allow_pickle=True))
        if 'xz_slice' in data:
            im = _plot_droplet_smooth(ax, data['xz_slice'],
                                       data.get('xz_solid', None), upsampling=4)
            if im_ref is None:
                im_ref = im
        step = int(snap_file.name.split('_step')[1].split('.')[0])
        ax.set_title(f'{label}\n(step {step}, $t^*$={step_to_tstar(step):.2f})', fontsize=10)

    if im_ref is not None:
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.93, 0.15, 0.012, 0.7])
        cbar = fig.colorbar(im_ref, cax=cbar_ax)
        cbar.set_label('相场 $\\phi$', fontsize=11)
        cbar.set_ticks([0, 0.5, 1.0])
        cbar.set_ticklabels(['气相', '界面', '液相'])
    fig.suptitle('线材形貌对比 ($We$=7.9, $\\theta$=162°, $p/D_0$=0.8)',
                 fontsize=13, fontweight='bold', y=1.01)
    save_fig(fig, 'fig4_12.png')


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='生成第四章论文图表')
    parser.add_argument('--skip-existing', action='store_true')
    parser.add_argument('--only', nargs='+', default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("  生成第四章论文图表（中文宋体 + Times New Roman）")
    print("=" * 70)

    t_start = time.time()
    print("\n  加载数据...")
    ch4 = load_ch4_data()
    s910 = load_s910_data()
    print(f"  加载 {len(ch4)} 条第四章数据, {len(s910)} 条双/多液滴数据")

    figures = [
        ('4.1', lambda: fig_4_01(ch4)),
        ('4.2', lambda: fig_4_02(ch4)),
        ('4.3', lambda: fig_4_03(ch4)),
        ('4.4', lambda: fig_4_04(ch4)),
        ('4.5', lambda: fig_4_05(ch4)),
        ('4.6', lambda: fig_4_06(ch4)),
        ('4.7', lambda: fig_4_07(ch4)),
        ('4.8', lambda: fig_4_08(s910)),
        ('4.9', lambda: fig_4_09(s910)),
        ('4.10', lambda: fig_4_10(s910)),
        ('4.11', lambda: fig_4_11(s910)),
        ('4.12', lambda: fig_4_12()),
    ]

    generated = []
    for fig_id, fig_func in figures:
        if args.only and fig_id not in args.only:
            continue
        if args.skip_existing:
            # Normalize fig_id to filename pattern
            fname = f'fig{fig_id.replace(".", "_")}.png'
            if (FIG_DIR / fname).exists():
                print(f"  SKIP fig_{fig_id} (exists)")
                generated.append((fig_id, 0, 'SKIPPED'))
                continue
        print(f"\n  生成 Fig {fig_id}...")
        t0 = time.time()
        try:
            fig_func()
            generated.append((fig_id, time.time() - t0, 'OK'))
        except Exception as e:
            generated.append((fig_id, time.time() - t0, f'ERROR: {e}'))
            print(f"  ERROR: {e}")

    total_time = time.time() - t_start
    print("\n" + "=" * 70)
    print("  生成完成")
    print("=" * 70)
    ok = sum(1 for _, _, s in generated if s == 'OK')
    print(f"  成功: {ok}/{len(generated)}, 耗时: {total_time:.1f}s")

    # List all fig4 files
    fig_files = sorted(FIG_DIR.glob('fig4_*.png'))
    print(f"\n  第四章输出文件 ({len(fig_files)}):")
    for f in fig_files:
        print(f"    {f.name} ({os.path.getsize(f)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
