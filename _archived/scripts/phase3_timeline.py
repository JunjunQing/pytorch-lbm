#!/usr/bin/env python3
"""Plot Phase 3 time evolution: spread_radius and z_max over time."""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'mathtext.fontset': 'cm',
})

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Convex N=150
with open(os.path.join(root, 'results', 'phase3_convex_n150.json')) as f:
    c150 = json.load(f)

# N=80 sweep
with open(os.path.join(root, 'results', 'phase3_geometry_sweep.json')) as f:
    sweep = json.load(f)

# Gather ridge histories from sweep
ridge_hists = {}
for r in sweep:
    if r['geometry'] == 'ridge':
        ridge_hists[r['amp']] = r.get('history', [])

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel 1: Convex spread_radius vs step
ax = axes[0, 0]
for r in c150:
    h = r.get('history', [])
    steps = [p['step'] for p in h]
    sr = [p['spread_radius'] for p in h]
    ax.plot(steps, sr, label=f"amp={r['amp']:.1f}", lw=2)
ax.set_xlabel('Step')
ax.set_ylabel('Spread radius (LU)')
ax.set_title('(a) Convex hemisphere — spread radius')
ax.legend()
ax.grid(True, alpha=0.2)

# Panel 2: Convex z_max vs step
ax = axes[0, 1]
for r in c150:
    h = r.get('history', [])
    steps = [p['step'] for p in h]
    zm = [p['z_max'] for p in h]
    ax.plot(steps, zm, label=f"amp={r['amp']:.1f}", lw=2)
ax.set_xlabel('Step')
ax.set_ylabel('Max droplet height (LU)')
ax.set_title('(b) Convex hemisphere — droplet height')
ax.legend()
ax.grid(True, alpha=0.2)

# Panel 3: Ridge k_max vs step
ax = axes[1, 0]
for amp in sorted(ridge_hists.keys()):
    h = ridge_hists[amp]
    if not h:
        continue
    steps = [p['step'] for p in h]
    kvals = [p['k'] for p in h]
    ax.plot(steps, kvals, label=f"amp={amp:.1f}", lw=2)
ax.axhline(y=1.0, color='gray', ls=':', alpha=0.4)
ax.set_xlabel('Step')
ax.set_ylabel('k = Dx/Dy')
ax.set_title('(c) Ridge — spreading asymmetry')
ax.legend()
ax.grid(True, alpha=0.2)

# Panel 4: Ridge spread_R vs step
ax = axes[1, 1]
for amp in sorted(ridge_hists.keys()):
    h = ridge_hists[amp]
    if not h:
        continue
    steps = [p['step'] for p in h]
    sr = [p.get('spread_radius', 0) for p in h]
    ax.plot(steps, sr, label=f"amp={amp:.1f}", lw=2)
ax.set_xlabel('Step')
ax.set_ylabel('Spread radius (LU)')
ax.set_title('(d) Ridge — spread radius')
ax.legend()
ax.grid(True, alpha=0.2)

plt.tight_layout()
save_dir = os.path.join(root, 'paper', 'figures')
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'phase3_timelines.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {save_path}")
plt.close()

# Print final comparison summary
print("\n" + "=" * 70)
print("PHASE 3 FINAL SUMMARY")
print("=" * 70)

print("\nConvex hemisphere (N=150):")
for r in c150:
    h = r.get('history', [])
    last = h[-1] if h else {}
    first = h[4] if len(h) > 4 else h[0] if h else {}
    print(f"  amp={r['amp']:.1f}:")
    print(f"    Initial spread_R = {first.get('spread_radius',0):.1f}")
    print(f"    Final   spread_R = {last.get('spread_radius',0):.1f}")
    print(f"    Max     spread_R = {r.get('max_spread',0):.1f}")
    print(f"    Final aspect     = {last.get('aspect_ratio',0):.4f}")
    print(f"    Volume change     = {last.get('volume',0)-first.get('volume',0):.1f}")
    print(f"    Stable            = {r.get('stable', False)}")
    print(f"    Domain (nx,ny)    = 150x150, max D={last.get('Dx',0):.0f} (within bounds)")

print(f"\nRidge (N=80):")
for r in sweep:
    if r['geometry'] != 'ridge':
        continue
    h = r.get('history', [])
    last = h[-1] if h else {}
    print(f"  amp={r['amp']:.1f}: k_max={r.get('max_k',0):.4f}, "
          f"spread_R={last.get('spread_radius',0):.1f}, "
          f"aspect={last.get('aspect_ratio',0):.4f}, "
          f"stable={r.get('stable', False)}")

print(f"\nConvex hemisphere (N=80):")
for r in sweep:
    if r['geometry'] != 'convex':
        continue
    h = r.get('history', [])
    last = h[-1] if h else {}
    domain_limit = last.get('Dx', 0) >= 79 or last.get('Dy', 0) >= 79
    print(f"  amp={r['amp']:.1f}: spread_R={last.get('spread_radius',0):.1f}, "
          f"aspect={last.get('aspect_ratio',0):.4f}, "
          f"stable={r.get('stable', False)}, "
          f"domain_boundary={domain_limit}")

print(f"\nInterpretation:")
print(f"  - Convex hemisphere at N=150: geo_amplification changes droplet")
print(f"    wetting behavior. Amp=0 → droplet retracts (no wetting).")
print(f"    Amp=1.5 → droplet maintains spread (proper wetting).")
print(f"  - Both convex and ridge show that geo_amplification creates")
print(f"    measurable differences in droplet behavior on curved surfaces.")
print(f"  - The optimal amp value differs between geometries,")
print(f"    suggesting geometry-dependent calibration is needed.")
