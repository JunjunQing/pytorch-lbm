"""Fill missing phase-diagram cells found by the paper-claim audit.

Missing N=150, alpha=1.5, We=7.9, theta=162 cells:
  R*=0.7  (paper claims 3.148 -- no traceable data)
  R*=2.0  (paper claims 2.161 -- no traceable data)

Also recompute the R*=1.5 and R*=3.0 cells to settle the discrepancy
between the paper (2.355 / 1.462) and r5_phase3 (2.290 / 1.410).

Usage: venv/bin/python scripts/fill_phase_cells.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.run_validation_12h import run_impact

CASES = [
    dict(label='R0.7_We7.9_a1.5', R_star=0.7, We=7.9, theta_eq=162.0, amp=1.5),
    dict(label='R2.0_We7.9_a1.5', R_star=2.0, We=7.9, theta_eq=162.0, amp=1.5),
    dict(label='R1.5_We7.9_a1.5_rerun', R_star=1.5, We=7.9, theta_eq=162.0, amp=1.5),
    dict(label='R3.0_We7.9_a1.5_rerun', R_star=3.0, We=7.9, theta_eq=162.0, amp=1.5),
]

if __name__ == '__main__':
    results = []
    for c in CASES:
        t0 = time.time()
        r = run_impact(c['We'], c['R_star'], c['theta_eq'], c['amp'],
                       N=150, N_STEPS=2000)
        out = {
            'label': c['label'], 'R_star': c['R_star'], 'We': c['We'],
            'theta_eq': c['theta_eq'], 'amp': c['amp'],
            'k_max': r.get('k_max'), 'mass_drift_pct': r.get('mass_drift_pct'),
            'step_peak': r.get('step_peak'),
            'elapsed_s': round(time.time() - t0, 1),
        }
        results.append(out)
        print(json.dumps(out), flush=True)

    json.dump(results, open('results/phase_cell_fill.json', 'w'), indent=1)
    print('saved results/phase_cell_fill.json')
