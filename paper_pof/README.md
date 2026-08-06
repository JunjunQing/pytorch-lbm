# PoF Submission Format (paper_pof/)

Physics of Fluids (AIP Publishing) submission version of the paper,
converted from the ICLR-format source in `../paper/`.

## Status

- **Compiles clean**: 11 pages, two-column AIP layout, 0 errors, 0 undefined refs.
- All 10 tables, 5 figures, and 24 AIP-style numbered references render correctly.
- Content is shared with the ICLR version via `\input{../paper/sections/...}`
  — single source of truth; edits to sections apply to both formats.

## Build

```bash
cd paper_pof
export TEXINPUTS="$PWD/texmf/tex//:/usr/share/texlive/texmf-dist/tex//:"
export BSTINPUTS="$PWD/texmf/bibtex/bst//:"
export BIBINPUTS=".:$PWD:"
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## Format notes

- Class: `\documentclass[aip,reprint,numerical,floatfix]{revtex4-2}`
  with the AIP substyle `aip4-2.rtx` (from the official REVTeX 4.2 TDS
  package, vendored under `texmf/`).
- Bibliography style: `aipnum4-2` (AIP numbered citations).
- REVTeX 4.2f quirks handled in `main.tex`:
  - `\title/\author/\affiliation` must appear **inside** `\begin{document}`
    (unlike standard classes).
  - `graphicx`/`hyperref` are loaded with `\usepackage` (the class-option
    form was removed in 4.2f).
  - `\Cref` (cleveref) is emulated via `\autoref`.

## Before final submission

AIP's current official template is the **aip-journals** class, available on
Overleaf: `template-for-submission-to-aip-journals` (id `wdmsvzfjgvyj`).
Swap the class at the top of `main.tex`:

```latex
\documentclass[aip,reprint,numerical,floatfix]{revtex4-2}
% -> \documentclass[aip,poa]{aip-journals}   (check current AIP instructions)
```

and remove the vendored `texmf/` fallback. All content macros used here
(`\Cref`, `\citet`, `\citep`) are compatible with aip-journals.
