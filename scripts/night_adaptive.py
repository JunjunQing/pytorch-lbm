#!/usr/bin/env python3
"""Overnight adaptive-alpha feasibility pipeline (self-iterating).

Phase 0: baseline trio on the elliptical ridge (N=150, 2000 steps)
         - off      (alpha=0,        wetting BC disabled)
         - const    (alpha=1.5,      current recommended default)
         - adaptive (alpha=field,    sigma=2, alpha_max=2.0)
Phase 1: automated analysis + branch decision:
         - unstable (NaN / mass blowup)  -> branch A: alpha_max=1.8, sigma=3
         - adaptive ~ const (diff<3%)    -> branch B: steeper ellipse (a=28)
         - otherwise                      -> branch C: sigma sensitivity (1, 3)
Phase 2: branch runs (2-4 cases)
Phase 3: extension — adaptive mode We sweep (We=5, 15) + resolution 120
Phase 4: final analysis -> results/ADAPTIVE_ALPHA_REPORT.md

Every case logs to results/adaptive_alpha_<tag>.json; failures do not stop
the pipeline. Estimated runtime ~2-3h on the local GTX 1080.

Usage: nohup venv/bin/python scripts/night_adaptive.py > results/night_adaptive.log 2>&1 &
"""
import os
import subprocess
import sys
import json
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, 'venv', 'bin', 'python3')
STUDY = os.path.join(ROOT, 'scripts', 'adaptive_alpha_study.py')
LOG = os.path.join(ROOT, 'results', 'night_adaptive.log')

A, B = 35.0, 54.4   # baseline ellipse (apex R_eff ~ 22.5 lu)
A2, B2 = 28.0, 66.7                # steeper ellipse (branch B)
THETA = 162.0
N_BASE, STEPS = 150, 2000


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')


def run_case(args, tag):
    """Run one study case; returns (ok, result_dict)."""
    log(f"START {tag}: {' '.join(args)}")
    t0 = time.time()
    try:
        proc = subprocess.run([PY, STUDY] + args, cwd=ROOT,
                              capture_output=True, text=True, timeout=3600)
        elapsed = (time.time() - t0) / 60
        if proc.returncode != 0:
            log(f"FAIL {tag} rc={proc.returncode} ({elapsed:.1f}min) "
                f"stderr={proc.stderr[-300:]}")
            return False, None
        # parse last JSON line printed by the study script
        result = None
        for line in proc.stdout.strip().splitlines():
            if line.startswith('{'):
                try:
                    result = json.loads(line)
                except json.JSONDecodeError:
                    pass
        if result is None:
            # fall back to reading the saved file
            with open(os.path.join(ROOT, 'results', f'adaptive_alpha_{tag}.json')) as f:
                result = json.load(f)['result']
        log(f"END   {tag}: k_max={result.get('max_k')} stable={result.get('stable')} "
            f"({elapsed:.1f}min)")
        return True, result
    except Exception as e:  # noqa: BLE001
        log(f"EXCEPTION {tag}: {e}")
        return False, None


def load(tag):
    path = os.path.join(ROOT, 'results', f'adaptive_alpha_{tag}.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def decide(off, const, adap):
    """Branch decision from phase-0 results. Returns branch name + args."""
    if not all([off, const, adap]):
        return 'B', ['--a', str(A2), '--b', str(B2)]  # retry baseline
    if not adap.get('stable'):
        log("DECISION: adaptive unstable -> branch A (alpha_max=1.8, sigma=3)")
        return 'A', ['--alpha-max', '1.8', '--sigma', '3.0']
    diff = abs(adap['max_k'] - const['max_k']) / max(const['max_k'], 1e-9) * 100
    log(f"DECISION: adaptive-vs-const diff = {diff:.1f}%")
    if diff < 3.0:
        log("DECISION: too close to const -> branch B (steeper ellipse)")
        return 'B', ['--a', str(A2), '--b', str(B2)]
    log("DECISION: adaptive differs from const -> branch C (sigma sensitivity)")
    return 'C', ['--sigma', '1.0']


def main():
    log("=" * 70)
    log("ADAPTIVE ALPHA NIGHT PIPELINE STARTED")
    log(f"Ellipse a={A} b={B} | theta={THETA} | N={N_BASE} | steps={STEPS}")

    # ---- Phase 0: baseline trio ----
    base_args = [f'--n-base={N_BASE}', f'--steps={STEPS}', f'--theta={THETA}']
    tags = {}
    ok_off, off = run_case(base_args + ['--mode=off', '--tag=off'], 'off')
    ok_c, const = run_case(base_args + ['--mode=const', '--tag=const'], 'const')
    ok_a, adap = run_case(base_args + ['--mode=adaptive', '--sigma=2.0',
                                       '--alpha-max=2.0', '--tag=adaptive_s2'],
                          'adaptive_s2')

    # ---- Phase 1: branch decision ----
    branch, branch_args = decide(off, const, adap)
    log(f"BRANCH: {branch}")

    # ---- Phase 2: branch runs ----
    if branch == 'A':
        _, adap_a = run_case(base_args + ['--mode=adaptive', '--sigma=3.0',
                                          '--alpha-max=1.8', '--tag=adaptive_A'],
                             'adaptive_A')
    elif branch == 'B':
        # steeper ellipse: rerun all three modes for consistency
        steep = [f'--n-base={N_BASE}', f'--steps={STEPS}', f'--theta={THETA}',
                 '--a', str(A2), '--b', str(B2)]
        _, _ = run_case(steep + ['--mode=off', '--tag=steep_off'], 'steep_off')
        _, _ = run_case(steep + ['--mode=const', '--tag=steep_const'], 'steep_const')
        _, adap_b = run_case(steep + ['--mode=adaptive', '--sigma=2.0',
                                      '--alpha-max=2.0', '--tag=steep_adaptive'],
                             'steep_adaptive')
    else:  # C
        _, adap_c1 = run_case(base_args + ['--mode=adaptive', '--sigma=1.0',
                                           '--alpha-max=2.0', '--tag=adaptive_s1'],
                              'adaptive_s1')
        _, adap_c3 = run_case(base_args + ['--mode=adaptive', '--sigma=3.0',
                                           '--alpha-max=2.0', '--tag=adaptive_s3'],
                              'adaptive_s3')

    # ---- Phase 3: extension - We sweep in adaptive mode ----
    for we, tag in [(5.0, 'adaptive_we5'), (15.0, 'adaptive_we15')]:
        run_case(['--mode=adaptive', '--sigma=2.0', '--alpha-max=2.0',
                  f'--n-base={N_BASE}', f'--steps={STEPS}', '--theta', str(THETA),
                  '--we', str(we), '--tag', tag], tag)

    # ---- Phase 4: final report ----
    write_report(branch)

    log("NIGHT PIPELINE COMPLETE")


def write_report(branch):
    report = os.path.join(ROOT, 'results', 'ADAPTIVE_ALPHA_REPORT.md')
    rows = []
    for tag in ['off', 'const', 'adaptive_s2', 'adaptive_s1', 'adaptive_s3',
                'adaptive_A', 'steep_off', 'steep_const', 'steep_adaptive']:
        d = load(tag)
        if d:
            r = d['result']
            rows.append((tag, r))
    lines = [
        f"# Adaptive alpha_geo Feasibility Report",
        f"",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S} | branch: {branch}",
        f"Ellipse a=35 b=54.4 (apex R_eff ~ 22.5 lu), theta=162, N=150, 2000 steps",
        f"",
        f"| tag | mode | max_k | Dx | Dy | stable | kappa err% | alpha_max_eff |",
        f"|-----|------|-------|----|----|--------|-------------|----------------|",
    ]
    for tag, r in rows:
        lines.append(
            f"| {tag} | {r['mode']} | {r['max_k']} | {r['Dx']} | {r['Dy']} | "
            f"{r['stable']} | {r['kappa_rel_err_pct']} | "
            f"{r.get('alpha_max', '-')} |")
    lines += [
        "",
        "## Interpretation",
        "- off vs const: amplification effect on the elliptical ridge",
        "- adaptive vs const: does the alpha field track local curvature?",
        "  (adaptive should sit between off and const, or beat const where",
        "   curvature varies: strong at apex, weak at flanks)",
        "",
        "## Curvature estimator",
        "- kappa = div(grad(f)/|grad(f)|) on the solid fraction field,",
        "  Gaussian-smoothed (sigma=1/2/3), apex error reported above.",
        "",
        "## Next steps (if feasible)",
        "- verify adaptive alpha on mixed-curvature + compare contact-line",
        "  local angles; scale to 3D (principal curvatures kappa1, kappa2)",
    ]
    with open(report, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    log(f"Report written to {report}")


if __name__ == '__main__':
    main()
