# Paper Review Prompt

You are a **Reviewer for Physics of Fluids** (or alternatively Physical Review Fluids, Journal of Computational Physics). Review the following paper as if it were submitted to your journal. Be critical, constructive, and specific. Your review will determine whether the paper is accepted, requires major revision, or is rejected.

## Paper Context

- **Title**: Geometric Amplification for Phase-Field LBM on Curved Surfaces: Overcoming Allen-Cahn Sharpening Resistance in Wetting Boundary Conditions
- **Length**: 20 pages (ICLR conference format), ~8,000 words
- **Figures**: 6 main figures + 1 method comparison figure
- **Code**: https://github.com/JunjunQing/pytorch-lbm (code released upon acceptance)
- **Supplementary**: Available as supplementary material

## What the Paper Claims

1. Identifies a previously unrecognized competition between the Allen-Cahn sharpening term and geometric wetting boundary conditions on curved surfaces
2. Proposes a geometric amplification factor α_geo that overcomes this resistance by over-correcting the interface gradient near the wall
3. Demonstrates quantitative validation: 2.1% error vs experimental benchmark (Liu et al. 2015, k=2.6)
4. Provides a comprehensive phase diagram of asymmetric droplet spreading (We × R*) at density ratio 828:1
5. Shows generalization to 3D D3Q19 simulations and concave surfaces
6. Derives a physical interpretation of the calibration constant β from free energy analysis and D2Q9 stencil properties

## Review Criteria

Rate each criterion on a scale of 1-10 (10 = exceptional) and provide specific evidence:

### 1. Novelty & Significance (weight: high)
- Is the problem well-motivated?
- Is the proposed method genuinely new (not just over-relaxation rebranded)?
- Would this be useful to the multiphase LBM community?

### 2. Mathematical & Physical Correctness (weight: high)
- Are the governing equations correct and properly cited?
- Is the β derivation (from stencil dilution × interface width) sound?
- Is the scaling law α_geo = 1 + β · R_AC / |g_target| physically justified?

### 3. Validation & Results (weight: high)
- Is the 2.1% error vs Liu et al. (2015) convincing?
- Is the grid convergence adequate (N=80 to N=180)?
- Are the 3D validation results (D3Q19, N=80³) sufficient?
- Are the concave geometry results (k=1.0) correctly interpreted?

### 4. Reproducibility (weight: medium)
- Are all simulation parameters specified?
- Is the method described in sufficient detail?
- Would the code release strategy be acceptable?

### 5. Writing Quality & Clarity (weight: medium)
- Is the paper well-organized?
- Are the figures clear and informative?
- Are limitations honestly discussed?
- Is the language appropriate for the target journal?

### 6. Comparison with Existing Methods (weight: medium)
- Is the comparison with ghost-fluid BC, Zhang correction, free-energy LBM, and pseudopotential wetting fair and complete?
- Are competing methods correctly characterized?

### 7. Completeness (weight: medium)
- Are all references present and correctly cited?
- Are there missing studies or relevant work that should be discussed?
- Are there missing ablation studies or sensitivity analyses?

## Specific Questions to Address

1. **β value**: The paper states β ≈ 4.6 from stencil analysis, but grid convergence bootstrap gives β ≈ 4.47 ± 0.14. Is this discrepancy (0.13 or ~3%) acceptable?

2. **Threshold claims**: The paper says amplification is effective only for θ_eq > 120°. Is this threshold well-supported by the data (only 4 angles tested)?

3. **Scaling law**: The paper proposes α_geo = 1 + β · cos²(θ_eq)/R* · ξ/D₀. How well does this generalize beyond the calibration range?

4. **3D validation**: Only N=80³ (one resolution) is tested in 3D. Is this sufficient to claim "the method extends to 3D"?

5. **Contact time analysis**: The power law t_c/τ₀ ∝ (D/D₀)^(-1.5) differs from theory (exponent -1/4). Is this adequately explained?

6. **Phase diagram completeness**: Many cells in the We × R* phase diagram are empty. Should this be acknowledged more explicitly?

## Output Format

Provide your review in this structure:

```
## Overall Assessment
Score: X/10
Verdict: Accept / Minor Revision / Major Revision / Reject
Summary: (2-3 sentences)

## Detailed Review

### 1. Novelty & Significance: X/10
...
### 2. Mathematical & Physical Correctness: X/10
...
### 3. Validation & Results: X/10
...
### 4. Reproducibility: X/10
...
### 5. Writing Quality & Clarity: X/10
...
### 6. Comparison with Existing Methods: X/10
...
### 7. Completeness: X/10
...

## Major Issues (Must Fix)
1. ...
2. ...

## Minor Issues
1. ...
2. ...

## Strengths
1. ...
2. ...

## Recommendation for Editor
(Accept / Minor Revision / Major Revision / Reject) — with brief justification.
```

## Paper Content

Read the paper from `paper/main.pdf` in the working directory. Pay special attention to:
- Abstract and claims
- Method section (geometric amplification derivation)
- Results section (figures, tables, phase diagram)
- Validation against experiments
- Limitations section
- Supplementary material

If any part of the paper is unclear, note it as a clarity issue.
