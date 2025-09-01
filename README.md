# Hybrid Stars — Master Thesis

LaTeX project for a master’s thesis on hybrid stars (compact objects with possible exotic cores). The repo includes the thesis source, CI to build the PDF, and a placeholder for future numerical work.

## Quick Start
- Build the PDF: `make pdf`
- Output: `thesis/main.pdf`

## Requirements
- A TeX distribution with `latexmk` and `biber` (e.g., TeX Live/MacTeX).
- Common packages used: `biblatex` (backend=biber, style=phys), `siunitx`, `amsmath`, `amssymb`, `hyperref`, `geometry`.

## Repository Layout
- `thesis/` — LaTeX source; entry point: `main.tex` (includes chapters and appendix, and `bib/refs.bib`).
- `numerics/` — Placeholder Python package for TOV/EoS work (to be filled in later).
- `docs/` — Notes and summaries.
- `.github/workflows/latex.yml` — GitHub Actions workflow to build and upload the PDF artifact.
- `Makefile` — Convenience `pdf` target that runs `latexmk` in `thesis/`.

## Build Locally
From the repo root:

```bash
make pdf
```

Equivalent manual command:

```bash
cd thesis && latexmk -pdf -interaction=nonstopmode main.tex
```

Clean build artifacts (run inside `thesis/`):

```bash
latexmk -C
```

## Continuous Integration
On each push/PR, GitHub Actions compiles `thesis/main.tex` and uploads `thesis/main.pdf` as an artifact (`thesis-pdf`).

## Editing Content
- Chapters: `thesis/chapters/`
- Appendix: `thesis/appendix/`
- Bibliography: `thesis/bib/refs.bib` (managed with `biblatex`/`biber`).
