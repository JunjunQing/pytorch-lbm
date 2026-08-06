#!/usr/bin/env python3
"""Generate all Chapter 3 thesis figures from 158 simulation cases.

Each PNG file contains exactly one figure. Multi-panel figures are split
into separate files (e.g. fig3_04_(a).png, fig3_04_(b).png).

Font rules: Chinese in 宋体 (Song), English/numbers/symbols in Times New Roman.

Usage: python3 scripts/generate_chapter3_figures.py [--skip-existing] [--only 3.4 3.7]
"""
import sys
import os
import json
import re
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

# Colors
# Wong/Okabe-Ito 色盲友好调色板 (Nature Methods, 2011)
PRIMARY = '#0072B2'          # 深蓝 — 单系列主色
COLORS = {                   # 基底类型配色
    'flat': '#0072B2',       # 蓝
    'ridge': '#E69F00',      # 橙
    'convex': '#009E73',     # 绿
    'concave': '#D55E00',    # 朱红
}
MARKERS = {
    'flat': 's',
    'ridge': 'o',
    'convex': '^',
    'concave': 'D',
}
WE_COLORS = {                # We分组配色（统一）
    5.0: '#0072B2', 7.9: '#E69F00', 10.0: '#009E73',
    12.0: '#D55E00', 15.0: '#CC79A7', 20.0: '#56B4E9',
    25.0: '#000000', 30.0: '#F0E442',
}
WE_MARKERS = {               # We分组标记（统一）
    5.0: 'o', 7.9: 's', 10.0: '^',
    12.0: 'D', 15.0: 'v', 20.0: 'P',
}
# 统一样式常量
LINE_W = 2            # 数据线宽
MARKER_SZ = 9         # 标记大小
MARKER_EDGE_W = 0.8   # 标记边框宽度
MARKER_EDGE = 'black'
REF_COLOR = 'gray'
REF_STYLE = '--'
REF_ALPHA = 0.4
GRID_ALPHA = 0.2
SUBSTRATE_CN = {
    'flat': '平面',
    'ridge': '脊状',
    'convex': '凸面',
    'concave': '凹面',
}

# Physical constants
D0 = 45.0
U0 = 0.05

# Directories
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / 'results'
FIG_DIR = RESULTS_DIR / 'thesis_figures'
SNAP_DIR = RESULTS_DIR / 'thesis_snapshots'
FIG_DIR.mkdir(exist_ok=True, parents=True)

# Custom colormap
_PHASE_COLORS = ['#FFFFFF', '#D6EAF8', '#85C1E9', '#3498DB', '#2471A3', '#1A5276']
CMAP_PHASE = LinearSegmentedColormap.from_list('phase', _PHASE_COLORS, N=256)
SOLID_COLOR = '#7F8C8D'


# ============================================================================
# Data Loading (unchanged)
# ============================================================================

def load_all_results():
    records = []
    schemas = [
        ('thesis_sweep_results.json', 'sweep'),
        ('thesis_bigblack_results.json', 'bigblack'),
        ('thesis_qingqing_results.json', 'qingqing'),
        ('studies_9_10.json', 'dual_multi'),
        ('supplementary_we_results.json', 'supp_we'),
        ('supplementary_theta_phase_results.json', 'supp_theta_phase'),
    ]
    for fname, src in schemas:
        fpath = RESULTS_DIR / fname
        if not fpath.exists():
            print(f"  WARNING: {fname} not found, skipping")
            continue
        with open(fpath) as f:
            data = json.load(f)
        for r in data:
            r['_source'] = src
            r.setdefault('D0', 45.0)
            r.setdefault('machine', 'local')
            r.setdefault('spacing_ratio', None)
            r.setdefault('n_drops', 1)
            r.setdefault('dt_star', None)
            r.setdefault('coalescence_step', None)
            if 'max_Dx' in r and 'Dx' not in r:
                r['Dx'] = r['max_Dx']
            if 'max_Dy' in r and 'Dy' not in r:
                r['Dy'] = r['max_Dy']
            if 'max_k' not in r:
                r['max_k'] = r.get('Dx', 0) / r.get('Dy', 1) if r.get('Dy', 0) > 0 else 0
            records.append(r)
    return records


def load_all_histories():
    merged = {}
    for fname in ['thesis_k_history.json', 'thesis_bigblack_history.json',
                  'thesis_qingqing_history.json',
                  'supplementary_we_history.json',
                  'supplementary_theta_phase_history.json']:
        fpath = RESULTS_DIR / fname
        if fpath.exists():
            with open(fpath) as f:
                merged.update(json.load(f))
    return merged


def filter_study(records, prefix):
    return [r for r in records if r['label'].startswith(prefix)]


def step_to_tstar(step):
    return step * U0 / D0


def save_fig(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    size_kb = os.path.getsize(path) / 1024
    print(f"  Saved {name} ({size_kb:.0f} KB)")


def _find_closest_step(prefix, target_step):
    candidates = sorted(SNAP_DIR.glob(f'{prefix}_step*.npz'))
    if not candidates:
        return None
    best = min(candidates, key=lambda f: abs(int(f.name.split('_step')[1].split('.')[0]) - target_step))
    return best


def _load_peak_steps():
    peak_steps = {}
    for fname in ['supplementary_we_results.json', 'supplementary_theta_phase_results.json']:
        fpath = RESULTS_DIR / fname
        if fpath.exists():
            with open(fpath) as f:
                data = json.load(f)
            for r in data:
                step = r.get('max_k_step', r.get('max_Dx_step', None))
                if step:
                    peak_steps[r['label']] = step
    for fname in ['thesis_sweep_results.json']:
        fpath = RESULTS_DIR / fname
        if fpath.exists():
            with open(fpath) as f:
                data = json.load(f)
            for r in data:
                step = r.get('max_k_step', None)
                if step:
                    peak_steps[r['label']] = step
    return peak_steps


# ============================================================================
# Fig 3.1: 网格收敛性
# ============================================================================

def fig_3_01(records):
    cases = filter_study(records, 's0_')
    if len(cases) < 3:
        print("  SKIP fig_3_01: insufficient s0 data")
        return
    cases.sort(key=lambda r: r.get('grid_level', 0))
    Ns = np.array([r.get('grid_level', 0) for r in cases], dtype=float)
    ks = np.array([r['max_k'] for r in cases])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(1.0 / Ns, ks, 'o-', color=PRIMARY, markersize=MARKER_SZ, linewidth=LINE_W,
            markeredgecolor=MARKER_EDGE, markeredgewidth=MARKER_EDGE_W)
    k_fine = ks[-1]
    ax.axhline(y=k_fine, color=REF_COLOR, linestyle=REF_STYLE, alpha=0.5,
               label=f'$k_{{150}}$ = {k_fine:.3f}')
    for n, k in zip(Ns, ks):
        ax.annotate(f'{k:.3f}', (1.0/n, k), textcoords="offset points",
                    xytext=(8, 8), fontsize=9)
    ax.set_xlabel('$1/N$（网格分辨率倒数）')
    ax.set_ylabel('不对称因子 $k_{max} = D_x / D_y$')
    ax.set_title('网格收敛性验证 ($We$=7.9, $\\theta$=162°, 脊状 $R^*$=1.0)')
    ax.legend()
    ax.grid(True, alpha=GRID_ALPHA)
    save_fig(fig, 'fig3_01.png')


# ============================================================================
# Fig 3.2: Liu 2015 验证
# ============================================================================

def fig_3_02(records):
    cases = filter_study(records, 's1_')
    if not cases:
        print("  SKIP fig_3_02: no s1 data")
        return
    ridge_cases = [r for r in cases if r['substrate_type'] == 'ridge']
    flat_cases = [r for r in cases if r['substrate_type'] == 'flat']

    rs = np.array([r['R_star'] for r in ridge_cases])
    ks = np.array([r['max_k'] for r in ridge_cases])
    sort_idx = np.argsort(rs)
    rs, ks = rs[sort_idx], ks[sort_idx]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(rs, ks, 'o-', color=PRIMARY, markersize=MARKER_SZ, linewidth=LINE_W,
            markeredgecolor=MARKER_EDGE, markeredgewidth=MARKER_EDGE_W, label='AC-LBM (150³+VP)')

    if flat_cases:
        k_flat = np.mean([r['max_k'] for r in flat_cases])
        ax.scatter([4.0], [k_flat], s=120, c='#0072B2', marker='s',
                   edgecolors='black', linewidths=0.8, zorder=5)
        ax.annotate(f'平面\n(k={k_flat:.2f})', (4.0, k_flat),
                    textcoords="offset points", xytext=(-10, 15),
                    fontsize=9, ha='center')

    for r_ref, k_ref in [(1.0, 2.6), (2.76, 1.33)]:
        ax.scatter([r_ref], [k_ref], s=200, c='#D55E00', marker='*',
                   edgecolors='darkred', linewidths=MARKER_EDGE_W, zorder=6)
        idx = np.argmin(np.abs(rs - r_ref))
        if abs(rs[idx] - r_ref) < 0.1:
            err_pct = abs(ks[idx] - k_ref) / k_ref * 100
            ax.annotate(f'Liu: {k_ref:.2f}\n(Δ={err_pct:.1f}%)',
                        (r_ref, k_ref), textcoords="offset points",
                        xytext=(12, -5), fontsize=9, color='darkred', fontweight='bold')

    ax.scatter([4.0], [1.0], s=200, c='#D55E00', marker='*',
               edgecolors='darkred', linewidths=MARKER_EDGE_W, zorder=6)
    ax.axhline(y=1.0, color=REF_COLOR, linestyle=REF_STYLE, alpha=REF_ALPHA, label='$k=1$（对称）')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=PRIMARY,
               markeredgecolor=MARKER_EDGE, markersize=MARKER_SZ, label='AC-LBM'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#D55E00',
               markeredgecolor='darkred', markersize=14, label='Liu et al. 2015'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    ax.set_xlabel('曲率比 $R^*$（$R_{\\mathrm{sub}}$ / $R_{\\mathrm{drop}}$）')
    ax.set_ylabel('不对称因子 $k = D_x / D_y$')
    ax.set_title('AC-LBM与Liu et al. (2015)实验数据验证')
    ax.set_xscale('log')
    ax.set_xlim(0.4, 5.0)
    ax.set_ylim(0, 4.5)
    ax.grid(True, alpha=GRID_ALPHA)
    save_fig(fig, 'fig3_02.png')


# ============================================================================
# Fig 3.3: 基底类型对比
# ============================================================================

def fig_3_03(records):
    cases = filter_study(records, 's2_')
    if not cases:
        print("  SKIP fig_3_03: no s2 data")
        return
    substrates = ['flat', 'ridge', 'convex', 'concave']
    rstar_levels = [1.0, 2.0]
    k_data = {}
    for sub in substrates:
        for rstar in rstar_levels:
            found = [r for r in cases if r['substrate_type'] == sub and r.get('R_star') == rstar]
            if sub == 'flat' and not found:
                found = [r for r in cases if r['substrate_type'] == sub and r.get('R_star') is None]
            if found:
                k_data[(sub, rstar)] = found[0]['max_k']

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(substrates))
    width = 0.3
    for i, rstar in enumerate(rstar_levels):
        ks = [k_data.get((sub, rstar), 0) for sub in substrates]
        bars = ax.bar(x + i * width - width/2, ks, width,
                      color=[COLORS[s] for s in substrates],
                      alpha=0.7 + 0.3 * i, edgecolor='black', linewidth=0.8,
                      label=f'$R^*$={rstar:.1f}')
        for bar, k in zip(bars, ks):
            if k > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        f'{k:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([SUBSTRATE_CN[s] for s in substrates])
    ax.set_ylabel('不对称因子 $k = D_x / D_y$')
    ax.set_title('不同基底类型下液滴铺展不对称性对比 ($We$=7.9, $\\theta$=162°)')
    ax.axhline(y=1.0, color=REF_COLOR, linestyle=REF_STYLE, alpha=0.5, label='$k=1$（对称）')
    ax.set_ylim(0, 3.5)
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')
    save_fig(fig, 'fig3_03.png')


# ============================================================================
# Fig 3.4(a): 韦伯数对不同基底的影响
# ============================================================================

def fig_3_04_a(records):
    cases_3a = filter_study(records, 's3a_')
    cases_3b = filter_study(records, 's3b_')
    supp_we = [r for r in records if r.get('_source') == 'supp_we' and r['substrate_type'] == 'ridge']

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for sub in ['flat', 'ridge', 'convex', 'concave']:
        sub_cases = [r for r in cases_3a + cases_3b if r['substrate_type'] == sub]
        if sub == 'ridge':
            sub_cases.extend(supp_we)
        seen = {}
        for r in sub_cases:
            we = r['We']
            if we not in seen:
                seen[we] = r
        sub_cases = sorted(seen.values(), key=lambda r: r['We'])
        if not sub_cases:
            continue
        ax.plot([r['We'] for r in sub_cases], [r['max_k'] for r in sub_cases],
                marker=MARKERS.get(sub, 'o'), markersize=MARKER_SZ, linewidth=LINE_W,
                color=COLORS.get(sub, 'gray'), markeredgecolor=MARKER_EDGE,
                markeredgewidth=MARKER_EDGE_W, label=SUBSTRATE_CN.get(sub, sub))

    ax.axhline(y=1.0, color=REF_COLOR, linestyle=REF_STYLE, alpha=REF_ALPHA)
    ax.set_xlabel('韦伯数 $We$')
    ax.set_ylabel('不对称因子 $k = D_x / D_y$')
    ax.set_title('韦伯数对不同基底上液滴铺展不对称性的影响')
    ax.legend()
    ax.grid(True, alpha=GRID_ALPHA)
    save_fig(fig, 'fig3_04_(a).png')


# ============================================================================
# Fig 3.4(b): Clanet标度律
# ============================================================================

def fig_3_04_b(records):
    cases_3a = filter_study(records, 's3a_')
    supp_we = [r for r in records if r.get('_source') == 'supp_we' and r['substrate_type'] == 'ridge']

    ridge_cases = [r for r in cases_3a + supp_we if r['substrate_type'] == 'ridge']
    seen_we = {}
    for r in ridge_cases:
        we = r['We']
        if we not in seen_we:
            seen_we[we] = r
    ridge_cases = sorted(seen_we.values(), key=lambda r: r['We'])
    if not ridge_cases:
        return

    we_arr = np.array([r['We'] for r in ridge_cases])
    k_arr = np.array([r['max_k'] for r in ridge_cases])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.loglog(we_arr, k_arr, 'o', color=PRIMARY, markersize=MARKER_SZ,
              markeredgecolor=MARKER_EDGE, markeredgewidth=MARKER_EDGE_W, label='脊状 $R^*$=1.0')

    if len(we_arr) >= 3:
        mask = we_arr <= 15.0
        log_we = np.log(we_arr[mask] if mask.sum() >= 3 else we_arr)
        log_k = np.log(k_arr[mask] if mask.sum() >= 3 else k_arr)
        coeffs = np.polyfit(log_we, log_k, 1)
        alpha, A = coeffs[0], np.exp(coeffs[1])

        we_fit = np.logspace(np.log10(we_arr.min() * 0.8), np.log10(we_arr.max() * 1.2), 100)
        k_fit = A * we_fit ** alpha
        ax.loglog(we_fit, k_fit, '--', color=REF_COLOR, linewidth=1.5,
                  label=f'拟合: $k = {A:.2f} \\cdot We^{{{alpha:.2f}}}$')

        k_quarter = k_arr[0] * (we_fit / we_arr[0]) ** 0.25
        ax.loglog(we_fit, k_quarter, ':', color='#999999', linewidth=1.0,
                  label='$k \\propto We^{1/4}$ (Clanet)')

        we_fit_arr = we_arr[mask] if mask.sum() >= 3 else we_arr
        k_fit_arr = k_arr[mask] if mask.sum() >= 3 else k_arr
        r2 = 1 - np.sum((k_fit_arr - A*we_fit_arr**alpha)**2) / np.sum((k_fit_arr - k_fit_arr.mean())**2)
        ax.text(0.95, 0.05, f'$\\alpha$ = {alpha:.3f}\n$R^2$ = {r2:.3f}',
                transform=ax.transAxes, fontsize=10, ha='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))

    ax.set_xlabel('韦伯数 $We$')
    ax.set_ylabel('不对称因子 $k = D_x / D_y$')
    ax.set_title('Clanet标度律验证（脊状 $R^*$=1.0）')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, which='both')
    save_fig(fig, 'fig3_04_(b).png')


# ============================================================================
# Fig 3.5: 接触角影响
# ============================================================================

def fig_3_05(records):
    cases = filter_study(records, 's4')
    supp_theta = [r for r in records if r.get('_source') == 'supp_theta_phase'
                  and r.get('We') == 7.9 and r['substrate_type'] == 'ridge']

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for sub in ['flat', 'ridge', 'convex', 'concave']:
        sub_cases = [r for r in cases if r['substrate_type'] == sub]
        if sub == 'ridge':
            sub_cases.extend(supp_theta)
        seen = {}
        for r in sub_cases:
            t = r['theta_eq']
            if t not in seen:
                seen[t] = r
        sub_cases = sorted(seen.values(), key=lambda r: r['theta_eq'])
        if not sub_cases:
            continue
        ax.plot([r['theta_eq'] for r in sub_cases], [r['max_k'] for r in sub_cases],
                marker=MARKERS.get(sub, 'o'), markersize=MARKER_SZ, linewidth=LINE_W,
                color=COLORS.get(sub, 'gray'), markeredgecolor=MARKER_EDGE,
                markeredgewidth=MARKER_EDGE_W, label=SUBSTRATE_CN.get(sub, sub))

    ax.axhline(y=1.0, color=REF_COLOR, linestyle=REF_STYLE, alpha=REF_ALPHA, label='$k=1$（对称）')
    ax.set_xlabel('接触角 $\\theta$ (°)')
    ax.set_ylabel('不对称因子 $k = D_x / D_y$')
    ax.set_title('接触角对铺展不对称性的影响 ($We$=7.9, $R^*$=1.0)')
    ax.set_xticks(range(30, 180, 15))
    ax.set_xlim(25, 170)
    ax.legend()
    ax.grid(True, alpha=GRID_ALPHA)
    ax.set_ylim(0, None)
    save_fig(fig, 'fig3_05.png')


# ============================================================================
# Fig 3.6: 曲率比影响
# ============================================================================

def fig_3_06(records):
    s1_ridge = [r for r in filter_study(records, 's1_') if r['substrate_type'] == 'ridge']
    s5_ridge = filter_study(records, 's5_')
    if not s1_ridge and not s5_ridge:
        print("  SKIP fig_3_06: no s1/s5 ridge data")
        return

    fig, ax = plt.subplots(figsize=(8, 5.5))
    we_groups = {}
    for r in s1_ridge + s5_ridge:
        we_groups.setdefault(r['We'], []).append(r)

    for we in sorted(we_groups.keys()):
        cases = we_groups[we]
        rs = np.array([r['R_star'] for r in cases])
        ks = np.array([r['max_k'] for r in cases])
        sort_idx = np.argsort(rs)
        ax.plot(rs[sort_idx], ks[sort_idx], marker=WE_MARKERS.get(we, 'o'),
                markersize=MARKER_SZ, linewidth=LINE_W, color=WE_COLORS.get(we, '#999999'),
                markeredgecolor=MARKER_EDGE, markeredgewidth=MARKER_EDGE_W, label=f'$We$={we:.1f}')

    ax.axhline(y=1.0, color=REF_COLOR, linestyle=REF_STYLE, alpha=REF_ALPHA, label='$k=1$')
    ax.axvline(x=1.0, color=REF_COLOR, linestyle=':', alpha=0.3)
    ax.set_xlabel('曲率比 $R^*$（$R_{\\mathrm{sub}}$ / $R_{\\mathrm{drop}}$）')
    ax.set_ylabel('不对称因子 $k = D_x / D_y$')
    ax.set_title('曲率比对脊状基底铺展的影响 ($\\theta$=162°)')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=GRID_ALPHA)
    ax.set_xlim(0.4, 6.0)
    save_fig(fig, 'fig3_06.png')


# ============================================================================
# Fig 3.7: We-θ 相图
# ============================================================================

def fig_3_07(records):
    from scipy.interpolate import griddata
    seen = {}
    for r in records:
        if r['substrate_type'] == 'ridge' and r.get('R_star') == 1.0:
            theta, we, k = r.get('theta_eq', 162.0), r.get('We', 7.9), r['max_k']
            if k > 0:
                key = (round(theta, 1), round(we, 1))
                if key not in seen:
                    seen[key] = (theta, we, k)
    ridge_pts = list(seen.values())
    if len(ridge_pts) < 10:
        print(f"  SKIP fig_3_07: only {len(ridge_pts)} data points")
        return

    pts = np.array([(p[0], p[1]) for p in ridge_pts])
    vals = np.array([p[2] for p in ridge_pts])

    theta_grid = np.linspace(30, 170, 100)
    we_grid = np.linspace(2, 25, 100)
    THETA, WE = np.meshgrid(theta_grid, we_grid)
    K_cubic = griddata(pts, vals, (THETA, WE), method='cubic')
    K_nearest = griddata(pts, vals, (THETA, WE), method='nearest')
    K_interp = np.where(np.isnan(K_cubic), K_nearest, K_cubic)
    K_interp = np.clip(K_interp, 0.5, 5.0)

    fig, ax = plt.subplots(figsize=(10, 7))
    levels = np.arange(0.8, 4.5, 0.2)
    cf = ax.contourf(THETA, WE, K_interp, levels=levels, cmap='RdYlBu_r', extend='both')
    cb = plt.colorbar(cf, ax=ax, label='不对称因子 $k_{max} = D_x / D_y$', shrink=0.85)
    cs = ax.contour(THETA, WE, K_interp, levels=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
                    colors='black', linewidths=0.8, alpha=0.6)
    ax.clabel(cs, fmt='%.1f', fontsize=8)
    ax.contour(THETA, WE, K_interp, levels=[1.0], colors='blue', linewidths=2.5, linestyles='--')
    ax.scatter(pts[:, 0], pts[:, 1], s=40, c='white', edgecolors='black', linewidths=0.8, zorder=5)

    ax.set_xlabel('接触角 $\\theta$ (°)')
    ax.set_ylabel('韦伯数 $We$')
    ax.set_title('铺展不对称性相图（脊状 $R^*$=1.0）')
    ax.set_xlim(30, 170)
    ax.set_ylim(2, 25)
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='black', markersize=7, label=f'模拟数据 ({len(ridge_pts)}个)'),
        Line2D([0], [0], color='blue', linestyle='--', linewidth=2.5, label='$k=1$ 对称边界'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.15)
    save_fig(fig, 'fig3_07.png')


# ============================================================================
# Fig 3.8(a): k(t*) 不同基底
# ============================================================================

def fig_3_08_a(records, histories):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sub_cases = {
        's7_flat_Rflat_We7.9': ('平面', COLORS['flat']),
        's7_ridge_R1.0_We7.9': ('脊状', COLORS['ridge']),
        's7_convex_R1.0_We7.9': ('凸面', COLORS['convex']),
        's7_concave_R1.0_We7.9': ('凹面', COLORS['concave']),
    }
    alt_cases = {
        's1_flat': ('平面', COLORS['flat']),
        's2_flat_Rflat': ('平面', COLORS['flat']),
        's1_ridge_R1.0': ('脊状', COLORS['ridge']),
        's2_convex_R1.0': ('凸面', COLORS['convex']),
        's2_concave_R1.0': ('凹面', COLORS['concave']),
    }
    plotted = set()
    for case_dict in [sub_cases, alt_cases]:
        for case_key, (label, color) in case_dict.items():
            if label in plotted:
                continue
            if case_key in histories:
                h = histories[case_key]
                ax.plot([step_to_tstar(d['step']) for d in h], [d['k'] for d in h],
                        '-', color=color, linewidth=1.5, label=label)
                plotted.add(label)
    ax.axhline(y=1.0, color=REF_COLOR, linestyle=REF_STYLE, alpha=REF_ALPHA)
    ax.set_xlabel('无量纲时间 $t^* = t \\cdot |U_0| / D_0$')
    ax.set_ylabel('不对称因子 $k = D_x / D_y$')
    ax.set_title('不同基底上铺展动力学 ($We$=7.9)')
    ax.legend()
    ax.grid(True, alpha=GRID_ALPHA)
    ax.set_ylim(0, 4)
    save_fig(fig, 'fig3_08_(a).png')


# ============================================================================
# Fig 3.8(b): k(t*) 不同We
# ============================================================================

def fig_3_08_b(records, histories):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    we_cases = {}
    for key in histories:
        if 's3a_ridge' in key or 's3b_ridge' in key:
            we_str = key.split('We')[-1] if 'We' in key else None
            if we_str:
                try:
                    we_cases[key] = float(we_str)
                except ValueError:
                    pass
    for case_key, we in sorted(we_cases.items(), key=lambda x: x[1]):
        h = histories[case_key]
        ax.plot([step_to_tstar(d['step']) for d in h], [d['k'] for d in h],
                '-', color=WE_COLORS.get(we, '#999999'), linewidth=LINE_W, label=f'$We$={we:.1f}')
    ax.axhline(y=1.0, color=REF_COLOR, linestyle=REF_STYLE, alpha=REF_ALPHA)
    ax.set_xlabel('无量纲时间 $t^* = t \\cdot |U_0| / D_0$')
    ax.set_ylabel('不对称因子 $k = D_x / D_y$')
    ax.set_title('不同韦伯数下铺展动力学（脊状 $R^*$=1.0）')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.set_ylim(0, 5)
    save_fig(fig, 'fig3_08_(b).png')


# ============================================================================
# Fig 3.9(a-d): 双液滴聚结（各拆一张）
# ============================================================================

def fig_3_09_a(records):
    cases = [r for r in records if r['label'].startswith('s9')]
    s9a = [r for r in cases if r['label'].startswith('s9a_')]
    if not s9a:
        return
    fig, ax = plt.subplots(figsize=(8, 5.5))
    subs = [r['substrate_type'] for r in s9a]
    ks = [r['max_k'] for r in s9a]
    merged = ['聚结' if r.get('coalescence_step') else '未聚结' for r in s9a]
    colors = [COLORS.get(s, '#999') for s in subs]
    bars = ax.bar([SUBSTRATE_CN.get(s, s) for s in subs], ks, color=colors, edgecolor='black', linewidth=0.8)
    for bar, k, m in zip(bars, ks, merged):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{k:.2f}\n({m})', ha='center', va='bottom', fontsize=9)
    ax.axhline(y=1.0, color=REF_COLOR, linestyle=REF_STYLE, alpha=REF_ALPHA)
    ax.set_ylabel('不对称因子 $k = D_x / D_y$')
    ax.set_title('双液滴聚结——基底类型对比 ($d/D$=1.2, $We$=7.9)')
    ax.grid(True, alpha=GRID_ALPHA, axis='y')
    save_fig(fig, 'fig3_09_(a).png')


def fig_3_09_b(records):
    cases = [r for r in records if r['label'].startswith('s9')]
    s9b = sorted([r for r in cases if r['label'].startswith('s9b_')], key=lambda r: r.get('spacing_ratio', 0))
    if not s9b:
        return
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ds = [r['spacing_ratio'] for r in s9b]
    ks = [r['max_k'] for r in s9b]
    ax.plot(ds, ks, 'o-', color=PRIMARY, markersize=MARKER_SZ, linewidth=LINE_W,
            markeredgecolor=MARKER_EDGE, markeredgewidth=MARKER_EDGE_W)
    for d, k in zip(ds, ks):
        ax.annotate(f'{k:.2f}', (d, k), textcoords="offset points", xytext=(0, 12), ha='center', fontsize=9)
    ax.set_xlabel('间距比 $d/D_0$')
    ax.set_ylabel('不对称因子 $k = D_x / D_y$')
    ax.set_title('双液滴聚结——间距影响（脊状 $R^*$=1.0, $We$=7.9）')
    ax.grid(True, alpha=GRID_ALPHA)
    save_fig(fig, 'fig3_09_(b).png')


def fig_3_09_c(records):
    cases = [r for r in records if r['label'].startswith('s9')]
    s9c = sorted([r for r in cases if r['label'].startswith('s9c_')], key=lambda r: r['We'])
    if not s9c:
        return
    fig, ax = plt.subplots(figsize=(8, 5.5))
    wes = [r['We'] for r in s9c]
    ks = [r['max_k'] for r in s9c]
    ax.plot(wes, ks, 'o-', color=PRIMARY, markersize=MARKER_SZ, linewidth=LINE_W,
            markeredgecolor=MARKER_EDGE, markeredgewidth=MARKER_EDGE_W)
    for w, k in zip(wes, ks):
        ax.annotate(f'{k:.2f}', (w, k), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
    ax.set_xlabel('韦伯数 $We$')
    ax.set_ylabel('不对称因子 $k = D_x / D_y$')
    ax.set_title('双液滴聚结——韦伯数影响（脊状 $R^*$=1.0, $d/D$=1.2）')
    ax.grid(True, alpha=GRID_ALPHA)
    ax.margins(y=0.15)
    save_fig(fig, 'fig3_09_(c).png')


def fig_3_09_d(records):
    cases = [r for r in records if r['label'].startswith('s9')]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')
    table_data = []
    for r in sorted(cases, key=lambda x: x['label']):
        coal = r.get('coalescence_step')
        coal_str = f'Step {coal}' if coal else '未聚结'
        table_data.append([
            r['label'].replace('s9a_dual_', '').replace('s9b_ridge_R1.0_', '').replace('s9c_ridge_R1.0_', ''),
            f"{r.get('We', 7.9):.1f}", f"{r.get('spacing_ratio', 1.2):.1f}",
            f"{r['max_k']:.2f}", coal_str,
        ])
    col_labels = ['案例', '$We$', '$d/D$', '$k$', '聚结状态']
    table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.3)
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#1565C0')
        table[0, j].set_text_props(color='white', fontweight='bold')
    ax.set_title('双液滴聚结实验汇总', fontsize=13, fontweight='bold', pad=20)
    save_fig(fig, 'fig3_09_(d).png')


# ============================================================================
# Fig 3.10(a-d): 多液滴沉积（各拆一张）
# ============================================================================

def fig_3_10_a(records):
    cases = [r for r in records if r['label'].startswith('s10')]
    s10a = [r for r in cases if r['label'].startswith('s10a_')]
    if not s10a:
        return
    fig, ax = plt.subplots(figsize=(8, 5.5))
    subs = [r['substrate_type'] for r in s10a]
    dxs = [r.get('max_Dx', r.get('Dx', 0)) for r in s10a]
    dys = [r.get('max_Dy', r.get('Dy', 0)) for r in s10a]
    x = np.arange(len(subs))
    width = 0.3
    ax.bar(x - width/2, dxs, width, label='$D_x$（横向）',
           color=[COLORS.get(s, '#999') for s in subs], edgecolor='black', linewidth=0.8, alpha=0.8)
    ax.bar(x + width/2, dys, width, label='$D_y$（纵向）',
           color=[COLORS.get(s, '#999') for s in subs], edgecolor='black', linewidth=0.8, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([SUBSTRATE_CN.get(s, s) for s in subs])
    for i, (dx, dy) in enumerate(zip(dxs, dys)):
        ax.text(i - width/2, dx + 2, f'{dx:.0f}', ha='center', fontsize=8)
        ax.text(i + width/2, dy + 2, f'{dy:.0f}', ha='center', fontsize=8)
    ax.set_ylabel('铺展量（格点单位）')
    ax.set_title('多液滴沉积——基底类型对比 ($n$=2, $\\Delta t^*$=1.0, $We$=7.9)')
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')
    save_fig(fig, 'fig3_10_(a).png')


def fig_3_10_b(records):
    cases = [r for r in records if r['label'].startswith('s10')]
    s10b = sorted([r for r in cases if r['label'].startswith('s10b_')], key=lambda r: r.get('dt_star', 0))
    if not s10b:
        return
    fig, ax = plt.subplots(figsize=(8, 5.5))
    dts = [r['dt_star'] for r in s10b]
    dxs = [r.get('max_Dx', r.get('Dx', 0)) for r in s10b]
    dys = [r.get('max_Dy', r.get('Dy', 0)) for r in s10b]
    ax.plot(dts, dxs, 'o-', color=PRIMARY, markersize=MARKER_SZ, linewidth=LINE_W,
            markeredgecolor=MARKER_EDGE, markeredgewidth=MARKER_EDGE_W, label='$D_x$（横向）')
    ax.plot(dts, dys, 's--', color=COLORS['ridge'], markersize=MARKER_SZ, linewidth=LINE_W,
            markeredgecolor=MARKER_EDGE, markeredgewidth=MARKER_EDGE_W, label='$D_y$（纵向）')
    for dt, dx, dy in zip(dts, dxs, dys):
        ax.annotate(f'{dx:.0f}', (dt, dx), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=8, color=PRIMARY)
        ax.annotate(f'{dy:.0f}', (dt, dy), textcoords="offset points",
                    xytext=(0, -14), ha='center', fontsize=8, color=COLORS['ridge'])
    ax.set_xlabel('沉积间隔 $\\Delta t^*$')
    ax.set_ylabel('铺展量（格点单位）')
    ax.set_title('多液滴沉积——沉积间隔影响（脊状 $R^*$=1.0, $n$=2）')
    ax.legend()
    ax.grid(True, alpha=GRID_ALPHA)
    ax.margins(y=0.12)
    save_fig(fig, 'fig3_10_(b).png')


def fig_3_10_c(records):
    cases = [r for r in records if r['label'].startswith('s10')]
    s10a_ridge = [r for r in cases if r['label'].startswith('s10a_') and r['substrate_type'] == 'ridge']
    s10c = sorted([r for r in cases if r['label'].startswith('s10c_')], key=lambda r: r.get('n_drops', 0))
    n_cases = s10c + (s10a_ridge[:1] if s10a_ridge else [])
    n_cases.sort(key=lambda r: r.get('n_drops', 0))
    if not n_cases:
        return
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ns = [r.get('n_drops', 0) for r in n_cases]
    dxs = [r.get('max_Dx', r.get('Dx', 0)) for r in n_cases]
    ax.plot(ns, dxs, 'o-', color=PRIMARY, markersize=MARKER_SZ, linewidth=LINE_W,
            markeredgecolor=MARKER_EDGE, markeredgewidth=MARKER_EDGE_W)
    for n, dx in zip(ns, dxs):
        ax.annotate(f'{dx:.0f}', (n, dx), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
    if len(ns) >= 2:
        z = np.polyfit(ns, dxs, 1)
        n_fit = np.linspace(min(ns) - 0.5, max(ns) + 0.5, 50)
        ax.plot(n_fit, np.polyval(z, n_fit), '--', color=REF_COLOR, alpha=0.5,
                label=f'线性拟合: {z[0]:.0f}n + {z[1]:.0f}')
        ax.legend()
    ax.set_xlabel('液滴数量 $n$')
    ax.set_ylabel('$D_x$（格点单位）')
    ax.set_title('累积铺展效应（脊状 $R^*$=1.0, $\\Delta t^*$=1.0）')
    ax.grid(True, alpha=GRID_ALPHA)
    save_fig(fig, 'fig3_10_(c).png')


def fig_3_10_d(records):
    cases = [r for r in records if r['label'].startswith('s10')]
    s10d = sorted([r for r in cases if r['label'].startswith('s10d_')], key=lambda r: r['We'])
    if not s10d:
        return
    fig, ax = plt.subplots(figsize=(8, 5.5))
    wes = [r['We'] for r in s10d]
    dxs = [r.get('max_Dx', r.get('Dx', 0)) for r in s10d]
    ax.plot(wes, dxs, 'o-', color=PRIMARY, markersize=MARKER_SZ, linewidth=LINE_W,
            markeredgecolor=MARKER_EDGE, markeredgewidth=MARKER_EDGE_W)
    for w, dx in zip(wes, dxs):
        ax.annotate(f'{dx:.0f}', (w, dx), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
    ax.set_xlabel('韦伯数 $We$')
    ax.set_ylabel('$D_x$（格点单位）')
    ax.set_title('多液滴沉积——韦伯数影响（脊状 $R^*$=1.0, $n$=2）')
    ax.grid(True, alpha=GRID_ALPHA)
    ax.margins(y=0.12)
    save_fig(fig, 'fig3_10_(d).png')


# ============================================================================
# Fig 3.11: 形貌快照
# ============================================================================

def _upsample_field(field, factor=4):
    return ndimage_zoom(field, factor, order=1)


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


def fig_3_11():
    if not SNAP_DIR.exists():
        print("  SKIP fig_3_11: no snapshot directory")
        return
    snap_files = list(SNAP_DIR.glob('*.npz'))
    if not snap_files:
        print("  SKIP fig_3_11: no snapshot files")
        return
    prefixes = sorted(set(f.name.split('_step')[0] for f in snap_files))
    peak_steps = _load_peak_steps()

    # --- fig3_11a: 基底演化 (4行×4列) ---
    sub_prefixes = ['s1_flat_Rflat', 's1_ridge_R1.0', 's1_convex_R1.0', 's1_concave_R1.0']
    sub_labels = ['平面', '脊状 ($R^*$=1.0)', '凸面 ($R^*$=1.0)', '凹面 ($R^*$=1.0)']
    key_steps = [200, 1000, 1400, 2000]
    n_rows, n_cols = len(sub_prefixes), len(key_steps)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows),
                             gridspec_kw={'hspace': 0.1, 'wspace': 0.05})
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    im_ref = None
    for row, (prefix, label) in enumerate(zip(sub_prefixes, sub_labels)):
        for col, step in enumerate(key_steps):
            ax = axes[row, col]
            snap_file = _find_closest_step(prefix, step)
            if snap_file is not None:
                data = dict(np.load(snap_file, allow_pickle=True))
                im = _plot_droplet_smooth(ax, data['xz_slice'], data.get('xz_solid', None), upsampling=4)
                if im_ref is None:
                    im_ref = im
                actual_step = int(snap_file.name.split('_step')[1].split('.')[0])
                ax.axhline(y=0, color='#555555', linewidth=1.5, alpha=0.7)
            else:
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes, ha='center', va='center', color='gray')
                actual_step = step
            if row == 0:
                ax.set_title(f'$t^*$={step_to_tstar(actual_step):.2f}', fontsize=10)
            if col == 0:
                ax.set_ylabel(label, fontsize=10, fontweight='bold', rotation=0, labelpad=80, ha='right', va='center')
    if im_ref is not None:
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.93, 0.15, 0.012, 0.7])
        cbar = fig.colorbar(im_ref, cax=cbar_ax)
        cbar.set_label('相场 $\\phi$', fontsize=11)
        cbar.set_ticks([0, 0.5, 1.0])
        cbar.set_ticklabels(['气相', '界面', '液相'])
    fig.suptitle('不同基底上液滴冲击演化过程\n($We$=7.9, $\\theta$=162°, $D_0$=45)', fontsize=13, fontweight='bold', y=1.01)
    save_fig(fig, 'fig3_11a.png')

    # --- fig3_11c: We效应形貌 ---
    we_snap_prefixes = sorted([p for p in prefixes if p.startswith('morph_We') or p.startswith('sup_We')])
    if we_snap_prefixes:
        we_cases = []
        for we in [5, 10, 15, 20]:
            p1, p2 = f'morph_We{we:.1f}', f'sup_We{we:.1f}'
            if p1 in prefixes:
                we_cases.append((p1, f'$We$={we}'))
            elif p2 in prefixes:
                we_cases.append((p2, f'$We$={we}'))
        if 's1_ridge_R1.0' in prefixes:
            we_cases.insert(1, ('s1_ridge_R1.0', '$We$=7.9'))
        if len(we_cases) >= 3:
            n = len(we_cases)
            fig, axes = plt.subplots(1, n, figsize=(5*n, 6))
            if n == 1:
                axes = [axes]
            for ax, (prefix, label) in zip(axes, we_cases):
                target_step = peak_steps.get(prefix, 1000)
                snap_file = _find_closest_step(prefix, target_step)
                if snap_file is not None:
                    data = dict(np.load(snap_file, allow_pickle=True))
                    _plot_droplet_smooth(ax, data['xz_slice'], data.get('xz_solid', None), upsampling=4)
                    actual_step = int(snap_file.name.split('_step')[1].split('.')[0])
                    ax.set_title(f'{label}\n($t^*$={step_to_tstar(actual_step):.2f})', fontsize=10)
                else:
                    ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes, ha='center', va='center')
                    ax.set_title(label, fontsize=11, fontweight='bold')
            fig.suptitle('韦伯数对液滴形貌的影响（脊状 $R^*$=1.0, $\\theta$=162°）', fontsize=13, fontweight='bold')
            save_fig(fig, 'fig3_11c.png')

    # --- fig3_11d: θ效应形貌 (xy俯视图) ---
    theta_snap_prefixes = sorted([p for p in prefixes if p.startswith('morph_theta') or p.startswith('sup_theta')])
    if theta_snap_prefixes:
        theta_cases = []
        for theta in [60, 90, 120, 150]:
            p1, p2 = f'morph_theta{theta:.0f}', f'sup_theta{theta:.0f}'
            if p1 in prefixes:
                theta_cases.append((p1, f'$\\theta$={theta}°'))
            elif p2 in prefixes:
                theta_cases.append((p2, f'$\\theta$={theta}°'))
        if len(theta_cases) >= 3:
            n = len(theta_cases)
            fig, axes = plt.subplots(1, n, figsize=(5*n, 6))
            if n == 1:
                axes = [axes]
            for ax, (prefix, label) in zip(axes, theta_cases):
                target_step = peak_steps.get(prefix, 1000)
                snap_file = _find_closest_step(prefix, target_step)
                if snap_file is not None:
                    data = dict(np.load(snap_file, allow_pickle=True))
                    if 'phi_3d' in data:
                        phi_3d = data['phi_3d']
                        solid_3d = data.get('solid_3d', np.zeros_like(phi_3d))
                        mid_i = phi_3d.shape[0] // 2
                        ridge_z = 0
                        for kz in range(phi_3d.shape[2] - 1, -1, -1):
                            if solid_3d[mid_i, phi_3d.shape[1] // 2, kz]:
                                ridge_z = kz
                                break
                        _plot_droplet_smooth(ax, phi_3d[:, :, ridge_z], solid_3d[:, :, ridge_z], upsampling=4)
                    else:
                        _plot_droplet_smooth(ax, data['xz_slice'], data.get('xz_solid', None), upsampling=4)
                    actual_step = int(snap_file.name.split('_step')[1].split('.')[0])
                    ax.set_title(f'{label}\n($t^*$={step_to_tstar(actual_step):.2f})', fontsize=10)
                else:
                    ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes, ha='center', va='center')
                    ax.set_title(label, fontsize=11, fontweight='bold')
            fig.suptitle('接触角对液滴形貌的影响——俯视图（脊状 $R^*$=1.0, $We$=7.9）', fontsize=13, fontweight='bold')
            save_fig(fig, 'fig3_11d.png')


# ============================================================================
# Fig 3.12: Dx-Dy 相空间
# ============================================================================

def fig_3_12(records):
    ridge_data = [r for r in records if r['substrate_type'] == 'ridge'
                  and r.get('Dx', 0) > 0 and r.get('Dy', 0) > 0
                  and not r['label'].startswith('s9') and not r['label'].startswith('s10')]
    if not ridge_data:
        print("  SKIP fig_3_12: no ridge Dx/Dy data")
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    study_colors = {'s1': '#E69F00', 's2': '#999999', 's3a': '#0072B2', 's3b': '#56B4E9',
                    's4a': '#009E73', 's4b': '#009E73', 's5': '#D55E00', 's6': '#CC79A7'}
    study_markers = {'s1': 'o', 's2': 'v', 's3a': 's', 's3b': 's', 's4a': '^', 's4b': '^', 's5': 'D', 's6': 'P'}
    study_labels = {'s1': '$R^*$扫描', 's2': '基底类型', 's3a': '$We$扫描', 's3b': '$We$(凸/凹)',
                    's4a': '接触角', 's4b': '$\\theta$补充', 's5': '$R^*$扩展', 's6': '$We$-$\\theta$相图'}
    plotted_studies = set()
    for r in ridge_data:
        m = re.match(r'(s\d+[abc]?)', r['label'])
        study = m.group(1) if m else 'other'
        color = study_colors.get(study, '#999999')
        marker = study_markers.get(study, 'o')
        label = study_labels.get(study, study) if study not in plotted_studies else ''
        plotted_studies.add(study)
        ax.scatter(r['Dx'], r['Dy'], c=color, marker=marker, s=80, zorder=5,
                   edgecolors=MARKER_EDGE, linewidth=MARKER_EDGE_W, label=label)
    lims = [5, max(r['Dx'] for r in ridge_data) + 10]
    ax.plot(lims, lims, 'k--', alpha=0.3, label='$k=1$')
    ax.plot(lims, [l/2 for l in lims], 'k:', alpha=0.3, label='$k=2$')
    ax.plot(lims, [l/3 for l in lims], 'k-.', alpha=0.3, label='$k=3$')
    ax.set_xlabel('$D_x$（横向铺展，格点单位）')
    ax.set_ylabel('$D_y$（纵向铺展，格点单位）')
    ax.set_title('$D_x$-$D_y$相空间（脊状基底, $R^*$=1.0）')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_xlim(lims[0], lims[1])
    ax.set_ylim(lims[0], lims[1])
    ax.set_aspect('equal')
    ax.grid(True, alpha=GRID_ALPHA)
    save_fig(fig, 'fig3_12.png')


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='生成第三章论文图表')
    parser.add_argument('--skip-existing', action='store_true')
    parser.add_argument('--only', nargs='+', default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("  生成第三章论文图表（中文宋体 + Times New Roman）")
    print("=" * 70)

    t_start = time.time()
    print("\n  加载数据...")
    records = load_all_results()
    histories = load_all_histories()
    print(f"  加载 {len(records)} 条记录, {len(histories)} 条时间序列")

    figures = [
        ('3.1', lambda: fig_3_01(records)),
        ('3.2', lambda: fig_3_02(records)),
        ('3.3', lambda: fig_3_03(records)),
        ('3.4a', lambda: fig_3_04_a(records)),
        ('3.4b', lambda: fig_3_04_b(records)),
        ('3.5', lambda: fig_3_05(records)),
        ('3.6', lambda: fig_3_06(records)),
        ('3.7', lambda: fig_3_07(records)),
        ('3.8a', lambda: fig_3_08_a(records, histories)),
        ('3.8b', lambda: fig_3_08_b(records, histories)),
        ('3.9a', lambda: fig_3_09_a(records)),
        ('3.9b', lambda: fig_3_09_b(records)),
        ('3.9c', lambda: fig_3_09_c(records)),
        ('3.9d', lambda: fig_3_09_d(records)),
        ('3.10a', lambda: fig_3_10_a(records)),
        ('3.10b', lambda: fig_3_10_b(records)),
        ('3.10c', lambda: fig_3_10_c(records)),
        ('3.10d', lambda: fig_3_10_d(records)),
        ('3.11', lambda: fig_3_11()),
        ('3.12', lambda: fig_3_12(records)),
    ]

    generated = []
    for fig_id, fig_func in figures:
        if args.only and fig_id not in args.only:
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

    fig_files = sorted(FIG_DIR.glob('fig3_*.png'))
    print(f"\n  输出文件 ({len(fig_files)}):")
    for f in fig_files:
        print(f"    {f.name} ({os.path.getsize(f)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
