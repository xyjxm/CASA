# STRIDE ICRA 2026 Draft

This directory contains the ICRA-style STRIDE paper draft and lightweight paper artifacts.

## Contents

- `main.tex`: paper source using the PaperPlaza/IEEE RAS `ieeeconf` class.
- `refs.bib`: BibTeX references used by the draft.
- `figures/fig_stride_framework_image2_icra20.png`: IMAGE2-generated framework figure informed by a 20-paper ICRA figure-style study.
- `figures/fig_replan_video_stream.png`: lightweight real-frame montage from a verified same-episode replan comparison.
- `review/icra20_figure_style_study.md`: list of the 20 accepted ICRA reference papers and the resulting style rules.
- `figures/*.pdf`: generated recovery/results/ablation figures on white backgrounds.
- `tables/*.tex`: generated LaTeX tables.
- `scripts/generate_paper_artifacts.py`: regenerates tables, figures, and consistency checks from the manifest.
- `source_manifest.json`: source provenance, official template records, row mappings, and figure-source notes.
- `video_stream_manifest.json`: source video/log/frame provenance and sha256 records for the replan montage.
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

The paper uses the official PaperPlaza LaTeX support bundle containing `ieeeconf.cls`, plus the PaperPlaza IEEEtran BibTeX style bundle. The framework is generated with IMAGE2 after a 20-paper ICRA figure-style study. The replan video-stream figure is a downsampled montage of real simulation-rendered frames from local videos, with source paths and sha256 records in `video_stream_manifest.json`. The remaining plots are generated from CSV metrics.

External comparison rows are adapted baselines for this local evaluation, not official reproductions. Numeric claims in the tables are generated from local CSV/JSON artifacts and checked by `table_consistency_check.json`.
