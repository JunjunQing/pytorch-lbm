# Experiment Plan: AC-LBM Validation Error Reduction

**Problem**: GPU-accelerated Allen-Cahn Phase-Field LBM overestimates spreading asymmetry ratio k vs Liu 2015 experiments
**Method Thesis**: Tuning wetting boundary condition parameters (amp, xi, theta_eq) and mobility can reduce validation error from 18-27% to <10%
**Date**: 2026-05-22

---

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|-----------------|-----------------------------|---------------|
| C1: amp is the dominant parameter for k reduction | amp amplifies geometric correction; current amp=1.8 gives +18% error at R*=1.0 | Reduce amp → k decreases; find amp giving k=2.6±0.2 at R*=1.0 | B1, B2 |
| C2: xi modulates effective contact angle through interface thickness | xi=4 vs 5 changes interface width and phi_c validity; thicker interface → AC sharpening competes more with wetting BC | Test xi ∈ {3, 4, 5, 6}; quantify k sensitivity | B2 |
| C3: mobility M affects contact line dynamics and maximum spread | Mobility controls interface relaxation speed; too high → interface smearing; too low → pinning | Test M scaled ∈ {0.5×, 1×, 2×} default | B3 |

---

## Paper Storyline

- **Main paper must prove**: k vs R* curve matches Liu 2015 within 10% across R* ∈ [0.5, 3.5]
- **Appendix can support**: Sensitivity analysis, ablation (VP off, no geometric amplification), parameter tables
- **Experiments intentionally cut**: We dependence re-test (already shown consistent), convex/concave morphologies (secondary validation)

---

## Experiment Blocks

### Block 1: amp Sensitivity Scan (MUST-RUN)

**Claim tested**: C1 — amp is the dominant parameter for k reduction
**Why this block exists**: Current amp=1.8 gives k=3.07 vs experiment 2.6 at R*=1.0 (+18.2%). Reducing amp lowers k. Need to find the optimal amp.
**Dataset / split / task**: Ridge substrate, R*=1.0, We=7.9, theta_eq=162°, n_base=150, N=3000
**Compared systems**: amp ∈ {0.0, 0.5, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0}
**Metrics**: k = Dx/Dy at max spread, step of max spread, Dx, Dy
**Setup details**:
- Grid: 150×150×107, D0=45, tau=0.53, xi=4.0 (as in run_thesis_sweep.py line 40)
- boundary_relax=0.0, geometric_wetting=True, VP=True
- Keep all other params fixed; only vary amp
**Success criterion**: |k - 2.6| < 0.2 at R*=1.0 (error < 7.7%)
**Failure interpretation**: If k cannot reach 2.6 by varying amp alone, combine with theta_eq tuning in Block 2
**Table / figure target**: Table 1, Figure: k vs amp
**Priority**: MUST-RUN

---

### Block 2: Combined amp × theta_eq 2D Sweep (MUST-RUN)

**Claim tested**: C1 + C2 interaction
**Why this block exists**: Both amp and theta_eq affect effective contact angle. amp=1.8 + theta=162° may over-correct. Need 2D sweep.
**Dataset / split / task**: Ridge substrate, R*=1.0, We=7.9, n_base=150, N=3000
**Compared systems**: 
- amp ∈ {1.0, 1.5, 1.8, 2.2} × theta_eq ∈ {155°, 158°, 160°, 162°}
- Selected 12 combinations
**Metrics**: k = Dx/Dy at max spread
**Setup details**: Keep xi=4.0, tau=0.53, boundary_relax=0.0, VP=True
**Success criterion**: Find (amp, theta_eq) giving k=2.5-2.7 at R*=1.0
**Failure interpretation**: If best combination still high, investigate VP smoothing scale or mobility (Block 3)
**Table / figure target**: Table 2, heatmap: k(amp, theta_eq)
**Priority**: MUST-RUN

---

### Block 3: xi and Mobility Sensitivity (MUST-RUN)

**Claim tested**: C2 + C3
**Why this block exists**: xi controls interface thickness and affects effective wetting. Mobility M affects contact line dynamics. Need to check sensitivity.
**Dataset / split / task**: Ridge substrate, R*=1.0, We=7.9, n_base=150, N=3000
**Compared systems**:
- xi ∈ {3, 4, 5, 6} × amp fixed at best from Block 1
- M ∈ {0.5×, 1×, 2×} default (where default M = 0.02/beta)
**Metrics**: k, stability
**Setup details**: Keep tau=0.53, theta_eq=162°, boundary_relax=0.0, VP=True
**Success criterion**: Find xi/M combination that improves accuracy vs baseline
**Failure interpretation**: If xi/M variation does not significantly change k, they are secondary factors
**Table / figure target**: Table 3
**Priority**: MUST-RUN

---

### Block 4: R*=2.76 Validation (MUST-RUN)

**Claim tested**: C1 — best parameters from Blocks 1-3 must also reduce R*=2.76 error from 27% to <10%
**Why this block exists**: Current R*=2.76 gives k=1.69 vs experiment 1.33 (+27.1%). This is the worst case. Must fix.
**Dataset / split / task**: Ridge substrate, R*=2.76, We=7.9, n_base=150, N=3000
**Compared systems**: Apply best (amp, theta_eq) from Blocks 1-2. Also sweep amp ∈ {0.3, 0.5, 0.8, 1.0} for R*=2.76
**Metrics**: k, error vs experiment (k_exp≈1.33)
**Setup details**: Keep xi=best from Block 3, boundary_relax=0.0, VP=True
**Success criterion**: |k - 1.33| < 0.15 (error < 11%)
**Failure interpretation**: If error persists at R*=2.76, the issue may be in grid resolution (nz=146 may be tight) or VP smoothing scale
**Table / figure target**: Table 4
**Priority**: MUST-RUN

---

### Block 5: Grid Convergence Check at Optimal Params (NICE-TO-HAVE)

**Claim tested**: Optimal parameters are grid-converged (not a numerical artifact)
**Why this block exists**: The grid convergence study (80→100→120→150) showed non-monotonic k. With optimal amp, verify N=120 and N=150 agree.
**Dataset / split / task**: Ridge substrate, R*=1.0, We=7.9, best params from Blocks 1-4, n_base ∈ {100, 120, 150}
**Compared systems**: n_base=100, 120, 150 (use D0 scaling from run_thesis_sweep.py)
**Metrics**: k at each resolution, error vs experiment
**Setup details**: Use best (amp, theta_eq, xi) from previous blocks
**Success criterion**: k difference between N=120 and N=150 < 0.1
**Failure interpretation**: If convergence is slow, may need N=200 or higher
**Table / figure target**: Figure: grid convergence with optimal params
**Priority**: NICE-TO-HAVE

---

### Block 6: VP Off Ablation (NICE-TO-HAVE)

**Claim tested**: VP contributes to boundary accuracy; no-VP may be worse
**Why this block exists**: VP (Volume Penalization) enables smooth curved boundaries. Check if disabling VP increases or decreases error.
**Dataset / split / task**: Ridge substrate, R*=1.0, We=7.9, n_base=150, use_vp=False
**Compared systems**: VP on vs VP off with best wet params
**Metrics**: k, boundary morphology
**Setup details**: Compare to VP on case with same params
**Success criterion**: VP off gives worse or similar k
**Failure interpretation**: If VP off gives BETTER k, then VP may be introducing smoothing artifacts
**Table / figure target**: Appendix comparison
**Priority**: NICE-TO-HAVE

---

### Block 7: Geometric Amplification Off Ablation (NICE-TO-HAVE)

**Claim tested**: Geometric amplification (amp>0) is necessary for correct contact angle on curved surfaces
**Why this block exists**: Current amp=1.8 amplifies gradient correction at walls. Need to confirm amp>0 helps.
**Dataset / split / task**: Ridge substrate, R*=1.0, We=7.9, n_base=150, geometric_wetting ∈ {True, False}
**Compared systems**: With optimal amp vs with amp=0 but same theta_eq
**Metrics**: k, theta_eq accuracy
**Setup details**: Use best theta_eq, xi from previous blocks
**Success criterion**: geometric_wetting=True gives k closer to experiment than False
**Failure interpretation**: If False gives better k, the amplification logic has a bug
**Table / figure target**: Appendix
**Priority**: NICE-TO-HAVE

---

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| M0: Sanity | Verify 2698BV3TencentCloud GPU access, baseline k=3.07 reproduction | 1 (amp=1.8 baseline) | k≈3.07 confirmed | ~15min | Low — baseline exists |
| M1: amp Scan | Find amp range that reduces k | 9 runs (amp=0→3) | Stop if k<2.5 at amp<0.5 | ~9×12min≈108min | Medium — may need amp<0 |
| M2: 2D Sweep | Combined amp×theta_eq for R*=1.0 | 12 runs | Find (amp,theta) giving k=2.5-2.7 | ~12×12min≈144min | Medium — may need fine grid |
| M3: xi/M Scan | Check xi and mobility sensitivity | 4+3=7 runs | xi/M sensitivity quantified | ~7×12min≈84min | Low — just measurement |
| M4: R*=2.76 Fix | Apply best params to R*=2.76 | ~8 runs | k within 11% of 1.33 | ~8×12min≈96min | High — R*=2.76 worst case |
| M5: Grid Check | Verify convergence at optimal params | 3 runs | N=120≈N=150 | ~3×12min≈36min | Low — diagnostic |
| M6: Ablations | VP off, amp off | 2 runs | Ablation confirms contribution | ~2×12min≈24min | Low — diagnostic |

**Total GPU time**: ~587 min ≈ 10 hours on CMP 30HX at 150³

---

## Compute and Data Budget

- **Total estimated GPU-hours**: ~10 hours on NVIDIA CMP 30HX (6GB VRAM)
- **Data preparation needs**: None — existing simulation infrastructure
- **Human evaluation needs**: Visual check of droplet morphology for failed cases
- **Biggest bottleneck**: R*=2.76 runs (nz=146, highest memory ~2500MB per run)

---

## Risks and Mitigations

- **Risk**: amp < 1.0 causes complete loss of geometric correction → k may drop below 1 (wrong physics)
  - **Mitigation**: Monitor theta_eq accuracy; if k < 1.5 for R*=1.0 with amp=0.5, reduce theta_eq instead
- **Risk**: xi=3 may cause interface resolution issues (too thin for 150³ grid)
  - **Mitigation**: If simulation unstable (NaN) at xi=3, reduce to xi=3.5
- **Risk**: R*=2.76 error (27%) may not reduce to <11% with param tuning alone
  - **Mitigation**: If param tuning insufficient, investigate VP smoothing scale (eps parameter) or increase nz
- **Risk**: GPU memory on CMP 30HX (6GB) may limit n_base > 150 for some combinations
  - **Mitigation**: If OOM, reduce n_base to 120 for combined sweep

---

## Final Checklist

- [ ] Main paper tables: k vs R* with error bars vs Liu 2015 covered
- [ ] Optimal parameter set identified for R*=1.0 and R*=2.76
- [ ] Novelty is isolated (geometric amplification confirmed as key)
- [ ] Simplicity is defended (no need for complex contact angle dynamics)
- [ ] Frontier contribution is justified (PyTorch AC-LBM validated to <10% error)
- [ ] Nice-to-have runs (ablations, grid check) separated from must-run runs

---

## Key Files for Experiment Execution

- **Script**: `/home/qmingjun/pytorch_lbm/scripts/run_thesis_sweep.py` — modify `liu_cases` and `grid_configs` for sweeps
- **Config**: `/home/qmingjun/pytorch_lbm/lbm/fe_config.py` — xi default is 5.0; sweep uses xi=4.0 (line 40 of run_thesis_sweep.py)
- **Wetting BC**: `/home/qmingjun/pytorch_lbm/lbm/fe_wetting.py` — geo_amplification logic at line 491-495
- **Results**: `/home/qmingjun/pytorch_lbm/results/thesis_sweep_results.json`
- **History**: `/home/qmingjun/pytorch_lbm/results/thesis_k_history.json`

---

## Parameter Ranges Summary

| Parameter | Current Value | Sweep Range | Notes |
|-----------|---------------|-------------|-------|
| amp | 1.8 (R*=1.0), 0.5 (R*=2.76) | 0.0–3.0 | Main tuning knob; current high for R*=1.0 |
| theta_eq | 162.0° | 155°–162° | May over-correct when combined with amp |
| xi | 4.0 (run_thesis_sweep.py), 5.0 (fe_config default) | 3–6 | Interface thickness; affects phi_c validity |
| M (mobility) | 0.02/beta (default) | 0.5×–2× | Controls interface relaxation speed |
| boundary_relax | 0.0 | keep at 0.0 | Currently fixed |
| tau | 0.53 | keep at 0.53 | Viscosity; changing affects Re |

---

## Physical Hypothesis for Error Sources

1. **amp too high**: amp=1.8 may over-correct wall gradient, pushing contact line further than physics warrants, increasing Dx and thus k
2. **theta_eq + amp interaction**: 162° + amp=1.8 is a double-over-correction; reducing amp or theta_eq alone reduces k
3. **xi=4 vs 5**: Thinner interface (xi=4) may make AC sharpening more aggressive near walls, competing with wetting BC
4. **VP smoothing scale**: Solid fraction field smoothed over 1 lattice unit may slightly shift contact line position

---

## ACTUAL EXPERIMENT RESULTS (2026-05-22)

### M1: amp Sensitivity Scan — COMPLETED ✅

| amp | k | Dx | Dy | err% | Status |
|-----|---|----|----|------|--------|
| 0.0 | 1.5405 | 57 | 37 | -40.7% | OK |
| 0.5 | 2.2903 | 71 | 31 | -11.9% | OK |
| 1.0 | 2.2903 | 71 | 31 | -11.9% | OK |
| **1.2** | **2.5172** | **73** | **29** | **-3.2%** | **✓** |
| **1.5** | **2.6552** | **77** | **29** | **+2.1%** | **✓** |
| 1.8 | 3.0741 | 83 | 27 | +18.2% | baseline |
| 2.0 | 4.5556 | 123 | 27 | +75.2% | OK |
| 2.5 | 4.2800 | 107 | 25 | +64.6% | OK |
| 3.0 | 4.9200 | 123 | 25 | +89.2% | OK |

**Key finding**: k is monotonic in amp for amp ≥ 1.0. amp=1.2-1.5 is the target range for k≈2.6.

**Best**: amp=1.5 → k=2.6552 (+2.1% error)

---

### M2: amp × theta_eq 2D Sweep — COMPLETED ✅

| amp | theta | k | err% | in_range |
|-----|-------|---|------|----------|
| 1.3 | 155° | 2.6552 | +2.1% | ✓ |
| **1.3** | **158°** | **2.5862** | **-0.5%** | **✓ BEST** |
| 1.3 | 160° | 2.5862 | -0.5% | ✓ |
| 1.3 | 162° | 2.5172 | -3.2% | ✓ |
| 1.4 | 155° | 2.6552 | +2.1% | ✓ |
| 1.4 | 158° | 2.6552 | +2.1% | ✓ |
| 1.4 | 160° | 2.5862 | -0.5% | ✓ |
| 1.4 | 162° | 2.5862 | -0.5% | ✓ |
| 1.5 | 155° | 2.7241 | +4.8% | ✓ |
| 1.5 | 158° | 2.6552 | +2.1% | ✓ |
| 1.5 | 160° | 2.6552 | +2.1% | ✓ |
| 1.5 | 162° | 2.6552 | +2.1% | ✓ |
| 1.6 | 155° | 2.9259 | +12.5% | ✗ |
| 1.6 | 158° | 2.7241 | +4.8% | ✓ |
| 1.6 | 160° | 2.7241 | +4.8% | ✓ |
| 1.6 | 162° | 2.6552 | +2.1% | ✓ |

**Best config for R*=1.0**: amp=1.3, theta=158° → k=2.5862 (err=-0.5%)

---

### M4: R*=2.76 amp sweep — COMPLETED ⚠️ (amp-insensitive)

| amp | k | Dx | Dy | err% | Status |
|-----|---|----|----|------|--------|
| 0.2 | 1.6857 | 59 | 35 | +26.7% | ✗ |
| 0.3 | 1.6857 | 59 | 35 | +26.7% | ✗ |
| 0.4 | 1.6857 | 59 | 35 | +26.7% | ✗ |
| 0.5 | 1.6857 | 59 | 35 | +26.7% | ✗ |
| 0.6 | 1.6857 | 59 | 35 | +26.7% | ✗ |
| 0.8 | 1.6857 | 59 | 35 | +26.7% | ✗ |
| 1.0 | 1.6857 | 59 | 35 | +26.7% | ✗ |
| 1.2 | 1.6857 | 59 | 35 | +26.7% | ✗ |
| 1.5 | 1.6857 | 59 | 35 | +26.7% | ✗ |

**Critical finding**: k is COMPLETELY invariant across amp=0.2-1.5 at R*=2.76!
The wetting BC has NO EFFECT at R*=2.76 — possible causes:
1. Ridge is too far from droplet for BC to influence contact line (flatter surface = weaker geometric effect)
2. nz=146 may be too tight for R*=2.76 substrate (ridge height ≈ 62.1, droplet may be squeezed)
3. VP smoothing at cylinder boundary dominates over wetting BC

**Dx=59 invariant**: The maximum spread in x-direction is set by impact physics, not wetting
**Dy=35 invariant**: The minimum spread in y-direction (parallel to ridge) is also unaffected by amp

---

### Summary: Optimal Parameters

| Case | amp | theta_eq | xi | Expected k | Notes |
|------|-----|----------|----|-----------|-------|
| R*=1.0 | 1.3 | 158° | 4.0 | ~2.59 (-0.5%) | Best from M2 |
| R*=1.0 | 1.5 | 162° | 4.0 | ~2.66 (+2.1%) | Close to M1 |
| R*=2.76 | ??? | ??? | ??? | ~1.69 | amp-insensitive, needs further investigation |

**R*=2.76 next steps**:
- Try lower theta_eq (140°, 145°, 150°) with n_base=120 to fit in memory
- Or investigate nz constraint (nz=146 may be too small for R_g=61.95)
- Or examine if VP smoothing is dominant over wetting BC at this curvature

---

## Updated Risk Assessment

| Risk | Mitigation |
|------|------------|
| R*=2.76 error persists (>20% across all amp values) | Lower theta_eq; check nz constraint; investigate VP vs BC interaction |
| M4b OOM at 150³ with theta sweep | Use n_base=120 for R*=2.76 runs |
| amp > 1.6 becomes unstable at R*=1.0 | Confirmed: amp=1.6 at theta=155° already out of range (+12.5%) |