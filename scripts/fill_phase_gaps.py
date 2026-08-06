#!/usr/bin/env python3
"""Fill ALL remaining empty cells of the We x R* phase diagram.

The supplementary table (tab:full_phase_diagram) marks uncomputed cells with
"---". This script computes every remaining cell at the paper's protocol:
N=150, N_STEPS=2000, amp=1.5, theta=162 (same as fill_phase_cells.py).

Missing cells (21):
  We=3 :  R*=0.7, 1.5, 2.0
  We=5 :  R*=0.5, 0.7, 1.5, 2.0, 3.0
  We=10:  R*=0.5, 0.7, 1.5, 2.0, 3.0
  We=15:  R*=0.7, 1.5, 2.0
  We=20:  R*=0.5, 0.7, 1.5, 2.0, 3.0

Usage: venv/bin/python scripts/fill_phase_gaps.py
Saves results/phase_gaps_fill.json
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.run_validation_12h import run_impact

CASES = [
    # We=3
    dict(label='W3_R0.7', We=3.0, R_star=0.7),
    dict(label='W3_R1.5', We=3.0, R_star=1.5),
    dict(label='W3_R2.0', We=3.0, R_star=2.0),
    # We=5
    dict(label='W5_R0.5', We=5.0, R_star=0.5),
    dict(label='W5_R0.7', We=5.0, R_star=0.7),
    dict(label='W5_R1.5', We=5.0, R_star=1.5),
    dict(label='W5_R2.0', We=5.0, R_star=2.0),
    dict(label='W5_R3.0', We=5.0, R_star=3.0),
    # We=10
    dict(label='W10_R0.5', We=10.0, R_star=0.5),
    dict(label='W10_R0.7', We=10.0, R_star=0.7),
    dict(label='W10_R1.5', We=10.0, R_star=1.5),
    dict(label='W10_R2.0', We=10.0, R_star=2.0),
    dict(label='W10_R3.0', We=10.0, R_star=3.0),
    # We=15
    dict(label='W15_R0.7', We=15.0, R_star=0.7),
    dict(label='W15_R1.5', We=15.0, R_star=1.5),
    dict(label='W15_R2.0', We=15.0, R_star=2.0),
    # We=20
    dict(label='W20_R0.5', We=20.0, R_star=0.5),
    dict(label='W20_R0.7', We=20.0, R_star=0.7),
    dict(label='W20_R1.5', We=20.0, R_star=1.5),
    dict(label='W20_R2.0', We=20.0, R_star=2.0),
    dict(label='W20_R3.0', We=20.0, R_star=3.0),
]

if __name__ == '__main__':
    results = []
    for c in CASES:
        t0 = time.time()
        r = run_impact(c['We'], c['R_star'], 162.0, 1.5,
                       N=150, N_STEPS=2000)
        out = {
            'label': c['label'], 'R_star': c['R_star'], 'We': c['We'],
            'theta_eq': 162.0, 'amp': 1.5,
            'k_max': r.get('k_max'), 'mass_drift_pct': r.get('mass_drift_pct'),
            'step_peak': r.get('step_peak'),
            'elapsed_s': round(time.time() - t0, 1),
        }
        results.append(out)
        print(json.dumps(out), flush=True)

    json.dump(results, open('results/phase_gaps_fill.json', 'w'), indent=1)
    print('saved results/phase_gaps_fill.json')
