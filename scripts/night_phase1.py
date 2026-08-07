#!/usr/bin/env python3
"""Phase-1 queue: M0 static attribution + M1 constant-alpha sweep.

M0 (B3): static elliptical-ridge contact angle, modes off/const/adaptive
         -> 135 deg vs 162 deg attribution
M1 (B2): constant-alpha sweep on the elliptical ridge (impact),
         alpha = 0.5 / 1.0 / 2.0  (0 and 1.5 already available)

Usage: nohup venv/bin/python scripts/night_phase1.py > results/night_phase1.log 2>&1 &
"""
import os
import subprocess
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, 'venv', 'bin', 'python3')
LOG = os.path.join(ROOT, 'results', 'night_phase1.log')


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')


def run(script, args, tag):
    log(f"START {tag}")
    t0 = datetime.now()
    try:
        proc = subprocess.run([PY, os.path.join(ROOT, 'scripts', script)] + args,
                              cwd=ROOT, timeout=7200)
        ok = proc.returncode == 0
    except Exception as e:  # noqa: BLE001
        ok = False
        log(f"EXCEPTION {tag}: {e}")
    dt = (datetime.now() - t0).total_seconds() / 60
    log(f"END   {tag}: {'OK' if ok else 'FAIL'} ({dt:.1f}min)")
    return ok


def main():
    log("=" * 60)
    log("PHASE-1 QUEUE STARTED (M0 static + M1 const sweep)")

    # M0: static ellipse, three modes
    for mode in ['off', 'const', 'adaptive']:
        run('static_ellipse.py', ['--mode', mode], f'M0_static_{mode}')

    # M1: constant-alpha sweep on ellipse (impact)
    for alpha in [0.5, 1.0, 2.0]:
        run('adaptive_alpha_study.py',
            ['--mode=const', f'--alpha={alpha}', '--n-base=150', '--steps=2000',
             '--theta=162.0', '--tag', f'const_a{alpha}'],
            f'M1_const_a{alpha}')

    log("PHASE-1 QUEUE COMPLETE")


if __name__ == '__main__':
    main()
