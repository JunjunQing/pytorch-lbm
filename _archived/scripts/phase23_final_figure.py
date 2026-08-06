#!/usr/bin/env python3
"""Final Phase 2.3 figure with N=150 ridge data + N=150 convex data.

4-panel: (a) Ridge k_max bar, (b) Ridge k evolution,
         (c) Convex spread_R bar, (d) Convex spread evolution
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'mathtext.fontset': 'cm',
    'axes.labelsize': 11, 'axes.titlesize': 12,
    'legend.fontsize': 9,
})

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# N=150 ridge comparison data
with open(os.path.join(root, 'results', 'phase4_ridge_n150_comparison.json')) as f:
    ridge_n150 = json.load(f)

# N=150 convex sweep data
with open(os.path.join(root, 'results', 'phase3_convex_sweep_n150.json')) as f:
    convex_n150 = json.load(f)
convex_n150 = [r for r in convex_n150 if r['amp'] in [0.0, 1.0, 1.5]]

# Color scheme
colors = {
    0.0: ('Ghost-fluid\nlinear (amp=0)', '#95a5a6', '--', 1.5),
    1.0: ('Zhang 2023\n(amp=1.0)', '#3498db', '-.', 1.5),
    1.5: ('This work\n(amp=1.5)', '#e74c3c', '-', 2.5),
}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# ===== (a) Ridge k_max bar chart =====
ax = axes[0, 0]
amp_order = [0.0, 1.0, 1.5]
labels = [colors[a][0] for a in amp_order]
k_vals = []
errors = []
# Grid convergence uncertainty for amp=1.5 is ±0.035 (from N=100..180 spread)
# For amp=0 and amp=1.0, no grid convergence data available → use 0
error_map = {0.0: 0, 1.0: 0, 1.5: 0.035}
for a in amp_order:
    for r in ridge_n150:
        if abs(r['amp'] - a) < 0.01:
            k_vals.append(r['max_k'])
            errors.append(error_map[a])
            break

bar_colors = [colors[a][1] for a in amp_order]
bars = ax.bar(range(len(amp_order)), k_vals, color=bar_colors,
              width=0.45, edgecolor='gray', yerr=errors, capsize=4)
ax.set_xticks(range(len(amp_order)))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('$k_{max} = D_x / D_y$', fontsize=12)
ax.set_title('(a) Ridge — Maximum spreading asymmetry', fontsize=12)

# Target line
ax.axhline(y=2.6, color='green', ls='--', lw=1.5, alpha=0.7,
           label='Liu 2015 exp.: $k \\approx 2.6$')
for i, v in enumerate(k_vals):
    ax.text(i, v + 0.08, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
ax.legend(fontsize=8, loc='upper left')
ax.set_ylim(0, 3.3)
ax.grid(True, alpha=0.1, axis='y')

# ===== (b) Ridge k time evolution =====
ax = axes[0, 1]
for r in ridge_n150:
    a = r['amp']
    h = r.get('history', [])
    steps = [p['step'] for p in h]
    kvals = [p['k'] for p in h]
    ax.plot(steps, kvals, color=colors[a][1], ls=colors[a][2],
            lw=colors[a][3], label=f"amp={a:.1f}")
ax.axhline(y=2.6, color='green', ls='--', lw=1.5, alpha=0.5,
           label='Target $k\\approx 2.6$')
ax.axhline(y=1.0, color='gray', ls=':', alpha=0.3)
ax.set_xlabel('LBM step')
ax.set_ylabel('$k = D_x / D_y$')
ax.set_title('(b) Ridge — Time evolution of $k$')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.1)

# ===== (c) Convex spread radius bar =====
ax = axes[1, 0]
sr_vals = []
sr_errors = []
for a in amp_order:
    for r in convex_n150:
        if abs(r['amp'] - a) < 0.01:
            h = r.get('history', [])
            final_sr = h[-1]['spread_radius'] if h else 0
            all_sr = [p['spread_radius'] for p in h[-10:]] if h else [0]
            sr_vals.append(final_sr)
            sr_errors.append(np.std(all_sr) if len(all_sr) > 1 else 0)
            break

bars = ax.bar(range(len(amp_order)), sr_vals, color=bar_colors,
              width=0.45, edgecolor='gray', yerr=sr_errors, capsize=4)
ax.set_xticks(range(len(amp_order)))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('Spread radius (lattice units)', fontsize=12)
ax.set_title('(c) Convex hemisphere — Equilibrium spread', fontsize=12)

# Droplet radius reference
ax.axhline(y=22.5, color='gray', ls=':', alpha=0.4, label='$R_{drop}$ (initial)')
for i, v in enumerate(sr_vals):
    ax.text(i, v + 1.2, f'{v:.1f}', ha='center', fontsize=10, fontweight='bold')
ax.legend(fontsize=8)
ax.set_ylim(0, max(sr_vals) * 1.35)
ax.grid(True, alpha=0.1, axis='y')

# ===== (d) Convex spread radius evolution =====
ax = axes[1, 1]
for r in convex_n150:
    a = r['amp']
    h = r.get('history', [])
    steps = [p['step'] for p in h]
    sr = [p['spread_radius'] for p in h]
    ax.plot(steps, sr, color=colors[a][1], ls=colors[a][2],
            lw=colors[a][3], label=f"amp={a:.1f}")
ax.axhline(y=22.5, color='gray', ls=':', alpha=0.4, label='$R_{drop}$')
ax.set_xlabel('LBM step')
ax.set_ylabel('Spread radius (LU)')
ax.set_title('(d) Convex hemisphere — Spread evolution')
ax.legend(fontsize=7, ncol=2, loc='upper center')
ax.grid(True, alpha=0.1)

plt.tight_layout()
save_dir = os.path.join(root, 'paper', 'figures')
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'fig_method_comparison.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {save_path}")

# Also save as PDF
save_pdf = save_path.replace('.png', '.pdf')
plt.savefig(save_pdf, bbox_inches='tight')
print(f"Saved: {save_pdf}")
plt.close()

# Print summary
print("\n" + "="*60)
print("METHOD COMPARISON — RIDGE N=150")
print("="*60)
for r in ridge_n150:
    f = r['final']
    target = 2.6
    err = (r['max_k'] - target) / target * 100
    print(f"  amp={r['amp']:.1f}: k_max={r['max_k']:.4f} "
          f"(err={err:+.1f}% vs Liu 2.6), "
          f"Dx={f['Dx']:.0f}, Dy={f['Dy']:.0f}")

print("\n" + "="*60)
print("METHOD COMPARISON — CONVEX N=150")
print("="*60)
for r in convex_n150:
    h = r.get('history', [])
    f = h[-1] if h else {}
    a = r['amp']
    print(f"  amp={a:.1f}: spread_R={f.get('spread_radius',0):.1f}, "
          f"aspect={f.get('aspect_ratio',0):.4f}, "
          f"Dx={f.get('Dx',0):.0f}")
