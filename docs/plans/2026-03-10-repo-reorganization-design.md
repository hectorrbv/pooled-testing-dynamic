# Repo Reorganization Design

**Date:** 2026-03-10
**Goal:** Reorganize the repository with best practices and a professional README.

## Approach: Pragmatic

Keep all Python module paths unchanged (no broken imports). Focus on file organization, documentation, and developer experience.

## Changes

### 1. New subdirectories in `augmented/`
- `augmented/figures/` — move all 14 result PNGs here
- `augmented/notebooks/` — move `examples_notebook.ipynb` here
- `augmented/paper/` — move `results.tex` and `findings_report.tex` here

### 2. `.gitignore` updates
Add: `_tmp_*.png`, `*.egg-info/`, `.env`, `venv/`, `*.log`

### 3. New files
- `requirements.txt` — numpy, matplotlib, pandas, graphviz
- `README.md` — complete rewrite with: features, structure, installation, quick start, module table, paper reference, authors

### 4. Root cleanup
- Move `Action_Plan.md` to `docs/`

### 5. Git cleanup
- Remove `_tmp_*.png` from tracking
