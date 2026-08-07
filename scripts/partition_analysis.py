#!/usr/bin/env python3
"""Partition analysis: does adaptive alpha beat every constant alpha on the
spatially varying curvature (elliptical ridge)? (B2 anti-claim, positional)

Uses existing contact-line time series (contact_line_diag_{mode}.json,
recorded every 200 steps): contact-line x extent, band-averaged kappa, and
applied alpha. Key idea: on the elliptical ridge the curvature (and hence
the alpha demand) varies along x — apex needs alpha~1.55, flanks need
alpha~1.0. A constant alpha can match at most ONE location:
  * if it matches the apex (1.5), it over-amplifies the flanks -> contact
    line over-advances into the low-kappa region (const: Dx 65 vs 63)
  * if it matches the flanks (small), it under-amplifies the apex

Metrics (per mode, late-time window):
  1. contact-line final x-extent (spread): over-advance indicator
  2. kappa at the contact line over time (does the band reach low-kappa?)
  3. alpha applied vs alpha demand (adaptive self-consistency)
  4. late-time contact-angle stability (const degraded to 108, adaptive 135)

Usage: venv/bin/python scripts/partition_analysis.py
Output: results/PARTITION_ANALYSIS.md
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_APEX = 1.0 + 4.0 * (54.4 / 35.0 ** 2) * abs(1.0 / __import__('math').tan(
    __import__('math').radians(162.0)))  # alpha_min at apex
KAPPA_LOW = 0.015  # kappa below this = "flank" region (alpha demand < ~1.2)


def load(mode):
    p = os.path.join(ROOT, 'results', f'contact_line_diag_{mode}.json')
    with open(p) as f:
        return json.load(f)


def main():
    modes = ['adaptive', 'const', 'off']
    data = {m: load(m) for m in modes}

    print("=" * 74)
    print("PARTITION ANALYSIS — adaptive vs constant alpha on varying curvature")
    print(f"apex alpha_min (theory) = {TARGET_APEX:.3f} | flank threshold "
          f"kappa < {KAPPA_LOW}")
    print("=" * 74)

    rows = []
    for m in modes:
        d = data[m]['diag']
        late = d[-4:]  # last 4 samples (~1400-2000)
        x_lo = min(e['x_lo'] for e in late)
        x_hi = max(e['x_hi'] for e in late)
        k_mean = sum(e['kappa_mean'] for e in late) / len(late)
        k_min = min(e['kappa_mean'] for e in late)
        theta = [e['theta_geo']['left'] for e in late if e.get('theta_geo')]
        theta_std = (max(theta) - min(theta)) if theta else 0.0
        rows.append((m, x_lo, x_hi, k_mean, k_min, theta_std))
        print(f"[{m:<9}] contact line x=[{x_lo},{x_hi}] | band kappa mean={k_mean:.4f} "
              f"min={k_min:.4f} | theta range={theta_std:.1f}deg")

    # verdict
    adap = dict(zip(['m', 'x_lo', 'x_hi', 'k', 'kmin', 'tstd'], rows[0]))
    cons = dict(zip(['m', 'x_lo', 'x_hi', 'k', 'kmin', 'tstd'], rows[1]))
    off = dict(zip(['m', 'x_lo', 'x_hi', 'k', 'kmin', 'tstd'], rows[2]))

    print("\n--- Verdict ---")
    # 1. spread over-advance: const should advance furthest into low-kappa
    spread = {'adaptive': adap['x_hi'] - adap['x_lo'],
              'const': cons['x_hi'] - cons['x_lo'],
              'off': off['x_hi'] - off['x_lo']}
    print(f"1) spread width: {spread} (const widest = over-advance "
          f"{'YES' if spread['const'] > spread['adaptive'] + 1 else 'no'})")
    # 2. kappa at contact line: lower kappa = band reached flanks
    print(f"2) band kappa: adaptive={adap['k']:.4f} const={cons['k']:.4f} "
          f"off={off['k']:.4f} (lower = deeper into flank region)")
    # 3. late-time stability
    print(f"3) late-time theta stability: adaptive={adap['tstd']:.1f}deg "
          f"const={cons['tstd']:.1f}deg off={off['tstd']:.1f}deg "
          f"(const degraded to 108 deg at step 2000 in earlier run)")
    # 4. adaptive alpha vs demand along the trajectory
    adap_alpha = [e['alpha_act_mean'] for e in data['adaptive']['diag']]
    adap_k = [e['kappa_mean'] for e in data['adaptive']['diag']]
    print(f"4) adaptive alpha follows kappa: alpha ranges "
          f"[{min(adap_alpha):.3f}, {max(adap_alpha):.3f}] over kappa "
          f"[{min(adap_k):.4f}, {max(adap_k):.4f}] "
          f"(demand at apex {TARGET_APEX:.2f})")

    verdict = ("C1 holds on positional evidence" if
               (spread['const'] > spread['adaptive'] + 1 and
                adap['tstd'] < cons['tstd'] and
                max(adap_alpha) > 1.35) else
               "C1 NOT established on positional evidence — needs redesign")
    print(f"\nVERDICT: {verdict}")

    with open(os.path.join(ROOT, 'results', 'PARTITION_ANALYSIS.md'), 'w') as f:
        f.write(f"# Partition Analysis — adaptive vs constant alpha\n\n")
        f.write(f"Ellipse a=35 b=54.4, theta=162, N=150. "
                f"Apex alpha_min (theory) = {TARGET_APEX:.3f}; "
                f"flank threshold kappa < {KAPPA_LOW}\n\n")
        f.write(f"| mode | contact x | spread | band kappa | late theta range |\n")
        f.write(f"|------|-----------|--------|------------|------------------|\n")
        for m, xl, xh, k, kmin, tstd in rows:
            f.write(f"| {m} | [{xl},{xh}] | {xh-xl} | {k:.4f} | {tstd:.1f} deg |\n")
        f.write(f"\n**Verdict**: {verdict}\n")
    print(f"\nsaved results/PARTITION_ANALYSIS.md")


if __name__ == '__main__':
    main()
