#!/usr/bin/env python3
"""3D follow-up validation: resolution refinement (N=100) + We sweep (N=80).

Paper limitation (review round 5.3): "3D study is preliminary (single
resolution at N=80)". This adds:
  * N=100 refinement of the five published 3D cases (k_resolution check)
  * N=80 We sweep at R*=1.0 (We=5, 10, 20) — 3D support for the phase diagram

Usage: venv/bin/python scripts/run_3d_followup.py
Saves results/3d_followup_results.json
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.run_3d_validation import run_3d_case

CASES_N100 = [
    dict(label='3d_n100_ridge_R1.0_We7.9', substrate_type='ridge',
         R_star=1.0, We=7.9, theta_eq=162.0, amp=1.5),
    dict(label='3d_n100_ridge_R1.0_We15', substrate_type='ridge',
         R_star=1.0, We=15.0, theta_eq=162.0, amp=1.5),
    dict(label='3d_n100_ridge_R2.0_We7.9', substrate_type='ridge',
         R_star=2.0, We=7.9, theta_eq=162.0, amp=1.0),
    dict(label='3d_n100_flat_We7.9', substrate_type='flat',
         R_star=None, We=7.9, theta_eq=162.0, amp=0.0),
    dict(label='3d_n100_convex_R1.0_We7.9', substrate_type='convex',
         R_star=1.0, We=7.9, theta_eq=162.0, amp=1.5),
]

CASES_WE80 = [
    dict(label='3d_ridge_R1.0_We5.0', substrate_type='ridge',
         R_star=1.0, We=5.0, theta_eq=162.0, amp=1.5),
    dict(label='3d_ridge_R1.0_We10.0', substrate_type='ridge',
         R_star=1.0, We=10.0, theta_eq=162.0, amp=1.5),
    dict(label='3d_ridge_R1.0_We20.0', substrate_type='ridge',
         R_star=1.0, We=20.0, theta_eq=162.0, amp=1.5),
]

if __name__ == '__main__':
    results = []
    for c in CASES_N100:
        t0 = time.time()
        r, h = run_3d_case(c['label'], c['substrate_type'], c['R_star'],
                           c['We'], c['theta_eq'], c['amp'],
                           n_base=100, n_steps=1500)
        r['elapsed_s'] = round(time.time() - t0, 1)
        results.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != 'history'}),
              flush=True)
    for c in CASES_WE80:
        t0 = time.time()
        r, h = run_3d_case(c['label'], c['substrate_type'], c['R_star'],
                           c['We'], c['theta_eq'], c['amp'],
                           n_base=80, n_steps=1500)
        r['elapsed_s'] = round(time.time() - t0, 1)
        results.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != 'history'}),
              flush=True)

    json.dump(results, open('results/3d_followup_results.json', 'w'), indent=1)
    print('saved results/3d_followup_results.json')
