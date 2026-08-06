#!/usr/bin/env python3
"""Analyze R*=2.76 resolution experiment results.

Compares the new deconfounded runs (results/r276_resolution.json) against:
  * M4 sweep  (results/m4_r276_amp_sweep_results.json)   — amp 0.2..1.5, nz=146
  * phase4a   (_archived/results/phase4_ridge_n150_comparison.json) — amp=0, nz=107
  * Liu 2015 experiment target k=1.33 (D/D0=2.76, We=7.9)

Prints a consolidated table plus per-hypothesis verdicts.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = 1.33


def load(name):
    with open(os.path.join(ROOT, name)) as f:
        return json.load(f)


def main():
    new = load('results/r276_resolution.json')
    m4 = load('results/m4_r276_amp_sweep_results.json')
    try:
        p4 = load('_archived/results/phase4_ridge_n150_comparison.json')
    except FileNotFoundError:
        p4 = []

    print("=" * 88)
    print(f"R*=2.76 RESOLUTION ANALYSIS (target k={TARGET} ± 0.15, Liu 2015)")
    print("=" * 88)

    print(f"\n{'src':<10} {'label':<18} {'amp':>4} {'th':>4} {'grid':>14} "
          f"{'max_k':>7} {'@step':>6} {'Dx':>5} {'Dy':>5} {'err%':>7} {'fit':>4}")
    print("-" * 88)

    rows = []
    for r in new:
        rows.append(('NEW', r['label'], r['amp'], r['theta_eq'], r['grid'],
                     r['max_k'], r['max_k_step'], r['Dx'], r['Dy'],
                     r['error_pct'], 'Y' if r['in_range'] else 'N'))
    for r in m4:
        rows.append(('M4', f"amp{r['amp']}", r['amp'], 162.0, '150x150x146',
                     r['max_k'], r['max_k_step'], r['Dx'], r['Dy'],
                     r['error_pct'], 'Y' if r['in_range'] else 'N'))
    for r in p4:
        rows.append(('P4a', r.get('geometry', ''), r.get('amp', '?'), 162.0,
                     r['grid'], r.get('max_k'), '-',
                     r['final']['Dx'], r['final']['Dy'],
                     round((r['final']['k'] - TARGET) / TARGET * 100, 1),
                     'Y' if abs(r['final']['k'] - TARGET) <= 0.15 else 'N'))

    for row in rows:
        print(f"{row[0]:<10} {row[1]:<18} {row[2]:>4} {row[3]:>4} {row[4]:>14} "
              f"{row[5]:>7} {str(row[6]):>6} {row[7]:>5} {row[8]:>5} "
              f"{row[9]:>+6.1f}% {row[10]:>4}")

    # --- Hypothesis verdicts ---
    print("\n" + "=" * 88)
    print("HYPOTHESIS VERDICTS")
    print("=" * 88)

    by = {r['label']: r for r in new}
    a, e, b, d, c = (by.get(k) for k in
                     ['A_amp0_th162', 'E_amp0.1_th162', 'B_amp0.5_th140',
                      'D_amp0_th140', 'C_amp0.5_th150'])

    print("\n[H1] amp=0 (wetting OFF) is the correct operating point at R*=2.76")
    if a:
        fit = abs(a['max_k'] - TARGET) <= 0.15
        print(f"  Case A (amp=0, th=162, 150³): max_k={a['max_k']} "
              f"k(2000)={a['k_final_2000']} -> {'MATCHES Liu' if fit else 'does NOT match'}")

    print("\n[H2] The 0->0.2 jump is the wetting ON/OFF switch (any amp>0 = wetting ON)")
    if e and a:
        same_as_on = abs(e['max_k'] - 1.6857) < 0.01
        diff_from_off = abs(e['max_k'] - a['max_k']) > 0.05
        print(f"  Case E (amp=0.1): max_k={e['max_k']} vs Case A (amp=0): {a['max_k']}")
        print(f"  -> amp=0.1 already at M4 level ({same_as_on}): "
              f"switch confirmed {'YES' if same_as_on and diff_from_off else 'NO'}")

    print("\n[H3] R036: lower theta restores amp sensitivity at R*=2.76")
    if b and d:
        amp_effect_low_th = abs(b['max_k'] - d['max_k']) > 0.05
        print(f"  th=140: amp=0.5 -> {b['max_k']} vs amp=0 -> {d['max_k']}")
        print(f"  -> amp effect at low theta: {'YES (sensitivity restored)' if amp_effect_low_th else 'NO (still insensitive)'}")
    if c:
        print(f"  th=150, amp=0.5 -> {c['max_k']}")

    print("\n[Summary] Resolution of the archived R*=2.76 mystery + R036 status:")

    # Suggest tracker update text
    print("\n--- Suggested tracker note ---")
    print("R027-R035 (M4): amp 0.2-1.0 share one code path (amp>1.0 gate); "
          "amp=0 (wetting OFF) never tested -> 'invariance' is a design artifact.")
    print("R036: lower-theta sweep re-evaluated 2026-08-05 (150³, nz=146): ...")


if __name__ == '__main__':
    main()
