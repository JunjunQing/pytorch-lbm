#!/usr/bin/env python3
"""Phase 2.3: Method comparison figure.

Three methods at θ=162°, ridge R*=1.0:
  - Ghost-fluid linear only (amp=0)
  - Zhang 2023 geometric correction, no amp (amp=1.0)
  - This work: geometric correction with amp=1.5

Uses N=80 sweep data + supplementary Quadratic-mode comparison at θ=90°.
Also generates convex comparison sidebar.
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

# Load Phase 3 sweep data (ridge at N=80)
with open(os.path.join(root, 'results', 'phase3_geometry_sweep.json')) as f:
    sweep = json.load(f)

ridge_data = [r for r in sweep if r['geometry'] == 'ridge' and r['amp'] in [0.0, 1.0, 1.5] and r.get('stable', True)]
convex_data = [r for r in sweep if r['geometry'] == 'convex' and r['amp'] in [0.0, 1.0, 1.5] and r.get('stable', True)]

# Load convex N=150 data (amp=0, 1.0, 1.5)
cv150_path = os.path.join(root, 'results', 'phase3_convex_sweep_n150.json')
if os.path.exists(cv150_path):
    with open(cv150_path) as f:
        cv150 = json.load(f)
    cv150 = [r for r in cv150 if r['amp'] in [0.0, 1.0, 1.5]]

print("=== RIDGE (N=80) DATA ===")
for r in ridge_data:
    h = r.get('history', [])
    k_history = [p['k'] for p in h]
    print(f"  amp={r['amp']:.1f}: k_max={r.get('max_k',0):.4f}, "
          f"final_k={k_history[-1]:.4f}, spread_R={h[-1]['spread_radius']:.1f}")

print("\n=== CONVEX (N=150) DATA ===")
for r in cv150:
    h = r.get('history', [])
    print(f"  amp={r['amp']:.1f}: final_spread_R={h[-1]['spread_radius']:.1f}, "
          f"final_aspect={h[-1]['aspect_ratio']:.4f}, "
          f"max_spread={r.get('max_spread',0):.1f}")

# ---- Figure ----
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# (a) Ridge k_max comparison bar chart
ax = axes[0, 0]
amps = [0.0, 1.0, 1.5]
labels = ['Ghost-fluid\nlinear (amp=0)', 'Zhang 2023\n(amp=1.0)', 'This work\n(amp=1.5)']
k_vals = []
for a in amps:
    for r in ridge_data:
        if abs(r['amp'] - a) < 0.01:
            k_vals.append(r.get('max_k', 0))
            break

colors = ['#bdc3c7', '#3498db', '#e74c3c']
bars = ax.bar(range(len(amps)), k_vals, color=colors, width=0.5, edgecolor='gray')
ax.set_xticks(range(len(amps)))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('$k_{max} = D_x / D_y$')
ax.set_title('(a) Ridge — Spreading asymmetry')
ax.axhline(y=1.0, color='gray', ls=':', alpha=0.4, label='$k=1$ (symmetry)')
for i, v in enumerate(k_vals):
    ax.text(i, v + 0.03, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')

# Add "target k=2.6 (Liu 2015)" reference
# Note: N=80 gives different absolute k than N=150, so target not directly applicable
ax.legend(fontsize=8)
ax.set_ylim(0, max(k_vals) * 1.25)

# (b) Ridge k time evolution
ax = axes[0, 1]
for r in ridge_data:
    h = r.get('history', [])
    steps = [p['step'] for p in h]
    kvals = [p['k'] for p in h]
    amp = r['amp']
    if amp == 0.0:
        c = '#bdc3c7'; ls = '--'
    elif amp == 1.0:
        c = '#3498db'; ls = '-.'
    else:
        c = '#e74c3c'; ls = '-'
    ax.plot(steps, kvals, color=c, ls=ls, lw=2,
            label=f"amp={amp:.1f}")
ax.axhline(y=1.0, color='gray', ls=':', alpha=0.4)
ax.set_xlabel('Step')
ax.set_ylabel('$k = D_x / D_y$')
ax.set_title('(b) Ridge — k evolution')
ax.legend()
ax.grid(True, alpha=0.15)

# (c) Convex spread radius comparison bar chart
ax = axes[1, 0]
sr_vals = []
for a in amps:
    for r in cv150:
        if abs(r['amp'] - a) < 0.01:
            h = r.get('history', [])
            sr_vals.append(h[-1]['spread_radius'] if h else 0)
            break

bars = ax.bar(range(len(amps)), sr_vals, color=colors, width=0.5, edgecolor='gray')
ax.set_xticks(range(len(amps)))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('Spread radius (LU)')
ax.set_title('(c) Convex hemisphere — Equilibrium spread')
for i, v in enumerate(sr_vals):
    ax.text(i, v + 0.5, f'{v:.1f}', ha='center', fontsize=10, fontweight='bold')

# Mark droplet radius reference
ax.axhline(y=22.5, color='gray', ls=':', alpha=0.4, label='$R_{drop}$ (initial)')
ax.legend(fontsize=8)
ax.set_ylim(0, max(sr_vals) * 1.3)

# (d) Convex spread radius time evolution
ax = axes[1, 1]
for r in cv150:
    h = r.get('history', [])
    steps = [p['step'] for p in h]
    sr = [p['spread_radius'] for p in h]
    amp = r['amp']
    if amp == 0.0:
        c = '#bdc3c7'; ls = '--'
    elif amp == 1.0:
        c = '#3498db'; ls = '-.'
    else:
        c = '#e74c3c'; ls = '-'
    ax.plot(steps, sr, color=c, ls=ls, lw=2,
            label=f"amp={amp:.1f}")
ax.axhline(y=22.5, color='gray', ls=':', alpha=0.4, label='$R_{drop}$')
ax.set_xlabel('Step')
ax.set_ylabel('Spread radius (LU)')
ax.set_title('(d) Convex hemisphere — spread evolution')
ax.legend()
ax.grid(True, alpha=0.15)

plt.tight_layout()
save_dir = os.path.join(root, 'paper', 'figures')
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'phase23_method_comparison.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\nSaved: {save_path}")
plt.close()

# ---- Summary Table ----
print("\n" + "=" * 70)
print("PHASE 2.3 — METHOD COMPARISON SUMMARY")
print("=" * 70)
print(f"\n{'Method':>30} {'Ridge k_max':>13} {'Convex spread_R':>17}")
print("-" * 65)
data_rows = [
    ("Ghost-fluid linear (amp=0)", ridge_data, cv150, 'max_k', 'spread_radius'),
    ("Zhang 2023 (amp=1.0)", ridge_data, cv150, 'max_k', 'spread_radius'),
    ("This work (amp=1.5)", ridge_data, cv150, 'max_k', 'spread_radius'),
]
for name, rd, cd, rk, ck in data_rows:
    rv = None
    for r in rd:
        if abs(r['amp'] - float(name.split('=')[-1].rstrip(')'))) < 0.01 or \
           (name == "Ghost-fluid linear (amp=0)" and abs(r['amp']) < 0.01):
            rv = r.get(rk, 0)
            break
    cv = None
    for r in cd:
        if abs(r['amp'] - float(name.split('=')[-1].rstrip(')'))) < 0.01 or \
           (name == "Ghost-fluid linear (amp=0)" and abs(r['amp']) < 0.01):
            h = r.get('history', [])
            cv = h[-1]['spread_radius'] if h else 0
            break
    print(f"  {name:>30} {rv:13.4f} {cv:17.1f}")

# Inset: cross-method comparison info
print(f"\n{'Observations':}")
print(f"  Ridge: amp=0→1.0: k_max {ridge_data[0]['max_k']:.3f}→{ridge_data[1]['max_k']:.3f} "
      f"(+{(ridge_data[1]['max_k']/ridge_data[0]['max_k']-1)*100:.0f}%)")
print(f"  Ridge: amp=1.0→1.5: k_max {ridge_data[1]['max_k']:.3f}→{ridge_data[2]['max_k']:.3f} "
      f"(+{(ridge_data[2]['max_k']/ridge_data[1]['max_k']-1)*100:.0f}%)")
if len(cv150) >= 3:
    sr0 = cv150[0]['history'][-1]['spread_radius']
    sr1 = cv150[1]['history'][-1]['spread_radius']
    sr2 = cv150[2]['history'][-1]['spread_radius']
    print(f"  Convex: amp=0→1.0: spread_R {sr0:.1f}→{sr1:.1f} "
          f"(+{(sr1/sr0-1)*100:.0f}%)")
    print(f"  Convex: amp=1.0→1.5: spread_R {sr1:.1f}→{sr2:.1f} "
          f"(+{(sr2/sr1-1)*100:.0f}%)")
print(f"\n  Note: Quadratic (Connington-Lee) mode requires θ∈[50°,130°];")
print(f"  at θ=162° the solver auto-selects linear mode.")
print(f"  See Kang & Lee (2022) for quadratic limitations at high θ.")
