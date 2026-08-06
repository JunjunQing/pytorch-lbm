#!/usr/bin/env python3
"""Analyze Phase 3 results and print history for both convex + ridge."""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Convex N=150
path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    '..', 'results', 'phase3_convex_n150.json')
with open(path) as f:
    convex_data = json.load(f)

print("=" * 70)
print("CONVEX HEMISPHERE (N=150) — TIME EVOLUTION")
print("=" * 70)

for r in convex_data:
    amp = r['amp']
    history = r.get('history', [])
    print(f"\n  amp={amp}: {len(history)} snapshots")
    print(f"  {'step':>6} {'Dx':>5} {'Dy':>5} {'Dz':>5} {'spread_R':>9} "
          f"{'z_max':>6} {'aspect':>7} {'vol':>8}")
    print(f"  " + "-" * 55)
    for h in history[::5]:  # every 5th snapshot
        print(f"  {h['step']:6d} {h['Dx']:5.0f} {h['Dy']:5.0f} {h['Dz']:5.0f} "
              f"{h['spread_radius']:9.1f} {h['z_max']:6.0f} "
              f"{h['aspect_ratio']:7.4f} {h['volume']:8.1f}")

# Geometry sweep data (convex + ridge at N=80)
print("\n" + "=" * 70)
print("GEOMETRY SWEEP (N=80) — AMP COMPARISON")
print("=" * 70)

path2 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     '..', 'results', 'phase3_geometry_sweep.json')
with open(path2) as f:
    sweep = json.load(f)

for geom in ['convex', 'ridge']:
    sub = [r for r in sweep if r['geometry'] == geom]
    print(f"\n  {geom.upper()}")
    print(f"  {'amp':>5} {'aspect':>7} {'k_max':>7} {'spread_R':>9} "
          f"{'z_max':>6} {'Dx':>5} {'Dy':>5} {'vol':>8}")
    print(f"  " + "-" * 55)
    for r in sorted(sub, key=lambda x: x['amp']):
        f = r['final']
        t = "OK" if r.get('stable', True) else "FAIL"
        vol = f.get('volume', 0)
        print(f"  {r['amp']:5.1f} {f.get('aspect_ratio',0):7.4f} "
              f"{r.get('max_k',0):7.4f} {f.get('spread_radius',0):9.1f} "
              f"{f.get('z_max',0):6.0f} {f.get('Dx',0):5.0f} "
              f"{f.get('Dy',0):5.0f} {vol:8.1f} {t:>4}")
