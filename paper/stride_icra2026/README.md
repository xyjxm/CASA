# STRIDE ICRA 2026 Draft

This directory contains the ICRA-style STRIDE paper draft and lightweight paper artifacts.

## Contents

- `main.tex`: paper source using the PaperPlaza/IEEE RAS `ieeeconf` class.
- `refs.bib`: BibTeX references used by the draft.
- `figures/*.pdf`: generated vector figures on white backgrounds.
- `tables/*.tex`: generated LaTeX tables.
- `scripts/generate_paper_artifacts.py`: regenerates tables, figures, and consistency checks from the manifest.
- `source_manifest.json`: source provenance, official template records, row mappings, and figure-source notes.
- `table_consistency_check.json`: machine-readable checks that table values match the source CSV/JSON metrics.
- `review/*.md`: two rounds of reviewer critique and revision notes.

## Build

Run from this directory:

```bash
python3 scripts/generate_paper_artifacts.py
latexmk -pdf -interaction=nonstopmode main.tex
```

or:

```bash
make
```

The committed source excludes LaTeX build products. Final draft PDFs are copied to `/mnt/data/students/lph/recording/stride_icra2026_draft_YYYYMMDD_HHMMSS/`.

## Provenance Notes

The paper uses the official PaperPlaza LaTeX support bundle containing `ieeeconf.cls`, plus the PaperPlaza IEEEtran BibTeX style bundle. No real video frames or screenshots are used in the figures; all figures are generated schematics or plots from CSV metrics.

External comparison rows are adapted baselines for this local evaluation, not official reproductions. Numeric claims in the tables are generated from local CSV/JSON artifacts and checked by `table_consistency_check.json`.
