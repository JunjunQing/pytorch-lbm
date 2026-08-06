#!/usr/bin/env python3
"""Generate REVIEW_PROMPT_FULL.md — preserves equations and citations."""
import os, re

PROJECT = '/home/qmingjun/projects/pytorch_lbm-main'
PAPER = os.path.join(PROJECT, 'paper')
SECTIONS = os.path.join(PAPER, 'sections')
OUTPUT = os.path.join(PROJECT, 'REVIEW_PROMPT_FULL.md')

section_files = [
    ('Abstract', '0_abstract.tex'),
    ('Introduction', '1_introduction.tex'),
    ('Related Work', '2_related_work.tex'),
    ('Geometric Amplification Method', 'method_geo_amp.tex'),
    ('Experiments and Phase Diagram', '4_experiments.tex'),
    ('Conclusion', '5_conclusion.tex'),
]

def read_section(name, filename):
    path = os.path.join(SECTIONS, filename)
    if not os.path.exists(path):
        return f"(Section {name} not found)\n"
    with open(path) as f:
        return f.read()

def minimal_clean(text):
    """Minimal LaTeX cleanup: keep equations, citations, labels but remove cruft."""
    # Section headings
    text = re.sub(r'\\section\{(.*?)\}', r'\n\n=== \1 ===', text)
    text = re.sub(r'\\subsection\{(.*?)\}', r'\n--- \1 ---', text)
    text = re.sub(r'\\subsubsection\{(.*?)\}', r'\n-- \1 --', text)
    text = re.sub(r'\\paragraph\{(.*?)\}', r'\n[ \1 ]', text)
    
    # Strip label, ref, clevref markers only (keep the text after \section{...})
    text = re.sub(r'\\label\{.*?\}', '', text)
    
    # Keep \ref, \cref, \citet, \citep as-is (GPT can parse these)
    # Keep \textbf, \emph markers
    text = re.sub(r'\\textbf\{(.*?)\}', r'**\1**', text)
    text = re.sub(r'\\emph\{(.*?)\}', r'*\1*', text)
    
    # Remove figure environments (GPT can't see images)
    text = re.sub(r'\\begin\{figure\}.*?\\end\{figure\}', '[FIGURE: omitted]', text, flags=re.DOTALL)
    
    # Clean tables to readable form
    text = clean_tables(text)
    
    # Remove comments
    text = re.sub(r'(?<!\\)%.*', '', text)
    
    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text

def clean_tables(text):
    """Convert tabular to readable table, keep caption."""
    tables = re.findall(r'\\begin\{table\}.*?\\end\{table\}', text, re.DOTALL)
    for t in tables:
        simple = '\n[TABLE] '
        cap = re.search(r'\\caption\{(.*?)\}', t)
        if cap:
            simple += f'┌ {cap.group(1)}\n'
        rows = re.findall(r'([^\\\\]+)\\\\', t)
        for row in rows:
            row_clean = re.sub(r'\\hline', '', row)
            row_clean = re.sub(r'\\textbf\{(.*?)\}', r'**\1**', row_clean)
            row_clean = row_clean.replace('&', ' |')
            row_clean = ' '.join(row_clean.split())
            if row_clean.strip():
                simple += f'  {row_clean.strip()}\n'
        simple += '[/TABLE]'
        text = text.replace(t, simple)
    return text

def main():
    sections = []
    for name, filename in section_files:
        raw = read_section(name, filename)
        clean = minimal_clean(raw)
        sections.append((name, clean))

    prompt = """# Peer Review Request

## Paper Title
"Geometric Amplification for Phase-Field LBM on Curved Surfaces: Overcoming Allen-Cahn Sharpening Resistance in Wetting Boundary Conditions"

## Target Venues
Physics of Fluids / Physical Review Fluids / Journal of Computational Physics

## Review Instructions
Please act as a **senior reviewer** for a top-tier computational physics journal.
Score this work from 1 to 10, and provide a structured review with the following sections:

1. **Overall Assessment** (1 paragraph summary)
2. **Major Strengths** (numbered, with reasoning)
3. **Major Weaknesses** (numbered by severity, with reasoning)
4. **Minor Issues** (numbered)
5. **Specific Questions** (numbered)
6. **Verdict**: Ready / Minor revisions / Major revisions / Reject
7. **Target Journal Fit**

## Key Context

This paper identifies a specific numerical deficiency: the Allen-Cahn sharpening term in
phase-field LBM resists geometric wetting boundary condition corrections on curved surfaces.
The authors propose an amplification factor alpha_geo, prove analytically that it is necessary
(linearized AC equation near curved walls yields a compatibility condition that fails for
theta > 90°), and calibrate optimal alpha_geo = 1.5 across three parameter dimensions.

## Key Experimental Results (for reference)

1. AC ablation: s=0 (no AC) -> k=3.32, mass -76%; s=1 -> k=2.66, mass -0.9%; s=2 -> k=1.43, mass -26%
2. Grid convergence: N=80 (k=2.38), N=100 (2.63), N=120 (2.67), N=150 (2.66), N=180 (2.70) -> k_inf ~ 2.66 +/- 0.04
3. Thresholds: alpha effective only for theta > 120°, R* < 3.0, D0/xi > 3.0
4. alpha=1.5 recommended for We=3-15, R*=0.5-3.0, theta=120-162°
5. We x R* phase diagram: 90+ simulations, three regimes (symmetric k<1.2, moderate 1.2-2.0, highly asymmetric k>=2.0)
6. Contact angle error: 13° -> 3° at theta=162° with alpha=1.5
7. Analytical proof: alpha_min = 1 + (xi/R_eff)*cot(theta) -> predicts 1.55 for R*=1.0
8. 3D validation: D3Q19 at 80^3, k=2.47; correct symmetric spreading on flat/hemisphere
9. Mass conservation: all cases < 1.3% drift
10. Literature gap: no prior work validates geometric WBC at theta > 140° on curved surfaces

## Known Limitations
- Only validated for cylindrical ridge geometry of Liu et al.
- 3D study is preliminary (single resolution at N=80)
- Contact time measurements are approximate
- alpha is constant rather than adaptive to local curvature
- Metrics differ from Liu (spreading ratio vs momentum ratio)

---

# FULL PAPER TEXT (equations preserved)

"""

    for name, text in sections:
        prompt += f"\n\n{'=' * 70}"
        prompt += f"\nSECTION: {name}"
        prompt += f"\n{'=' * 70}\n\n"
        prompt += text.strip()

    prompt += f"""

{'=' * 70}
END OF PAPER
{'=' * 70}

---

## Additional Information

### Bibliography (29 entries)
Key references include:
- Liu et al. (2015) Nature Communications — experimental benchmark
- Fakhari & Bolster (2017) JCP — geometric wetting on curved boundaries
- Connington & Lee (2015) JCP — wetting model for curved surfaces
- Fakhari et al. (2017) PRE — improved locality AC-LBM
- Huang & Zhang (2022) PoF — simplified curved wetting BC (CACE-LBM)
- Zhang, Tang & Wu (2022) CAMWA — wetting boundary schemes (phase-field LBM)
- Wang et al. (2024) PoF — 3D curved wetting BC (color-gradient LBM)
- Ezzatneshan (2021) — droplet impact on curved surfaces (AC-LBM)

### Reviewer Guidance (please address these in your review)

1. The paper provides an analytic proof (linearized AC equation near curved walls) that
   the geometric wetting BC and AC equation are incompatible for theta > 90°, yielding
   alpha_min = 1 + (xi/R_eff) * cot(theta). The predicted alpha_min ~ 1.55 for R*=1.0
   is consistent with the empirically optimal alpha=1.5. Is this derivation convincing
   as a rigorous justification (as opposed to heuristic argument)?

2. The paper includes literature validation coverage (Table 1) showing no prior work
   validates geometric WBC at theta > 140°. All prior validations stop at theta=135-140°.
   Does this evidence adequately support the novelty claim?

3. The contact angle error table shows improvement from 13° to 3° at theta=162° with
   alpha=1.5. Is this sufficient, or should more quantitative contact-angle validation
   (e.g., plots of theta_eff vs R*) be added?

4. The 3D study is preliminary (single 80^3 resolution). Is this acceptable for
   PoF / PRFluids, or does it need substantial expansion?

5. The Liu et al. comparison uses different We and metrics (spreading ratio vs momentum
   ratio). Does this weaken the validation sufficiently to disqualify publication?

6. Is the paper's contribution best described as: (a) a numerical correction method,
   (b) a physical insight about AC/wetting interaction, or (c) an engineering tool?
"""

    with open(OUTPUT, 'w') as f:
        f.write(prompt)

    print(f"Generated {OUTPUT}")
    print(f"  Lines: {prompt.count(chr(10))}")

if __name__ == '__main__':
    main()
