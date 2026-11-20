# Issue 01: LaTeX Compilation Bugs, Code Flaws, and Typography

Status: resolved

## Description

Address all LaTeX source syntax errors and typography defects:
1. Fix undefined `\Comment` macro in Algorithm 3: replaced with `\hfill \textit{// ...}`.
2. Fix column mismatch in Table 3 (`tab:mcu_perf`): changed to `{llrrr}` and wrapped in `\resizebox{\columnwidth}{!}`.
3. Rename legacy label `\label{alg:option_a_distill}` to `\label{alg:dssr_distill}` in Algorithm 2.
4. Add `\usepackage[T1]{fontenc}` and `\usepackage{lmodern}` to ensure proper glyph rendering (accented characters like Ré, Kajić, quotes, underscores, hyphenation).
5. Replace raw Unicode em-dashes (`—`) in Section 1 with LaTeX standard `---`.
6. Add `\usepackage{lmodern}` and `\usepackage[T1]{fontenc}` for publication-grade vector typography.
7. Change theorem environments `theorem`, `lemma`, and `proposition` to `\theoremstyle{plain}`.

## Acceptance Criteria
- [x] No undefined control sequences or compilation warnings in `main.log`.
- [x] All table column counts match data columns.
- [x] Clean font encoding with `lmodern` and T1.
- [x] No raw Unicode punctuation or legacy labels.
