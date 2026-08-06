#!/usr/bin/env python3
"""Overnight GPU task queue (12h window).

Waits for the running R*=2.76 decoupling experiment to finish, then runs
in sequence (each failure is logged but does not stop the queue):

  1. resolve_r276.py              (running since 23:22, 5 cases, ~2.5h)
  2. run_static_contact_production.py  (2 cases, ~1.2h)
  3. fill_phase_gaps.py           (21 phase-diagram cells, ~3h)
  4. run_3d_followup.py           (3D N=100 + We sweep, ~0.5h)
  5. analysis: analyze_r276.py + NIGHT_SUMMARY.md

Usage: nohup venv/bin/python scripts/night_queue.py > results/night_queue.log 2>&1 &
"""
import os
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, 'venv', 'bin', 'python3')
LOG = os.path.join(ROOT, 'results', 'night_queue.log')

# PID of the already-running resolve_r276.py job (spawned earlier)
R276_PID = int(os.environ.get('R276_PID', '13241'))

JOBS = [
    ('static_contact_production', ['scripts/run_static_contact_production.py']),
    ('fill_phase_gaps', ['scripts/fill_phase_gaps.py']),
    ('3d_followup', ['scripts/run_3d_followup.py']),
]


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')


def run_job(name, args):
    log(f"START {name}: {' '.join(args)}")
    t0 = time.time()
    cmd = [PY, os.path.join(ROOT, args[0])]
    proc = subprocess.run(cmd, cwd=ROOT)
    elapsed = (time.time() - t0) / 60
    status = 'OK' if proc.returncode == 0 else f'FAIL(rc={proc.returncode})'
    log(f"END   {name}: {status} ({elapsed:.1f} min)")
    return proc.returncode == 0


def main():
    log("=" * 70)
    log("NIGHT QUEUE STARTED")

    # 1. Wait for the running R*=2.76 experiment
    log(f"Waiting for R*=2.76 job (PID {R276_PID}) ...")
    while True:
        try:
            os.kill(R276_PID, 0)
        except (OSError, ProcessLookupError):
            break
        time.sleep(30)
    log("R*=2.76 job finished (results in results/r276_resolution.json)")

    # 2-4. Run the queued jobs
    ok = 0
    for name, args in JOBS:
        if run_job(name, args):
            ok += 1

    # 5. Analysis
    log(f"Queue finished: {ok}/{len(JOBS)} jobs OK. Running analysis ...")
    try:
        subprocess.run([PY, os.path.join(ROOT, 'scripts', 'analyze_r276.py')],
                       cwd=ROOT, timeout=300)
    except Exception as e:  # noqa: BLE001
        log(f"analyze_r276.py failed: {e}")

    # Night summary
    summary_path = os.path.join(ROOT, 'results', 'NIGHT_SUMMARY.md')
    with open(summary_path, 'w') as f:
        f.write(f"# Overnight GPU Run Summary\n\n")
        f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
        f.write("## Jobs\n")
        f.write(f"- resolve_r276 (5 cases): results/r276_resolution.json\n")
        f.write(f"- static_contact_production (2 cases): "
                f"results/static_contact_production.json\n")
        f.write(f"- fill_phase_gaps (21 cells): results/phase_gaps_fill.json\n")
        f.write(f"- 3d_followup (8 cases): results/3d_followup_results.json\n\n")
        f.write("## Key questions answered\n")
        f.write("1. R*=2.76: is amp=0 the correct operating point "
                "(H1)? any-amp>0 == wetting-ON switch (H2)?\n")
        f.write("2. Static contact angle at N=150: amp=0 vs 1.5 "
                "(reviewer suggestion)\n")
        f.write("3. Complete phase diagram: 21 cells filled\n")
        f.write("4. 3D resolution refinement N=80 -> N=100 + We sweep\n")
    log(f"Summary written to {summary_path}")
    log("NIGHT QUEUE COMPLETE")


if __name__ == '__main__':
    main()
