see pipeline_package_v2.zip to get the whole package
# Computational Ad-Attention Screening Pipeline

A lightweight, fully-local pipeline that predicts attention distribution on ad creative using published computational saliency algorithms + face detection + OCR-based AOI (Area-of-Interest) mapping — built to screen for competing focal points before creative goes to spend.

**Start here:** read `NOTES.md` first — it has the full project story, methodology, results, and honest limitations, written for both technical and non-technical readers.

## Quick start

```bash
pip install -r requirements.txt
# also needs the Tesseract OCR binary installed at the system level:
#   apt-get install tesseract-ocr   (Ubuntu/Debian)
#   brew install tesseract          (Mac)

python3 download_models.py          # fetches the face detector (~1.2MB)
python3 run_analysis.py your_ad.png # edit the AOI list inside the script for your creative's layout
```

## What it outputs

- A per-AOI attention share breakdown (% of predicted attention landing in each named region: logo, headline, offer, CTA, etc.)
- A **Focus score** (0–1): how concentrated vs. fragmented attention is across the creative
- **Competition flags**: pairs of AOIs fighting for the same glance (near-equal attention share, spatially close)
- A **gaze path**: the predicted fixation order, drawn as numbered arrows over a heatmap, so you can see whether the creative tells a coherent visual story or jumps around incoherently

## Project files

| File | Purpose |
|---|---|
| `NOTES.md` | **Read this first.** Full chain-of-thought, methodology, results, limitations, interview talking points |
| `saliency_pipeline.py` | Core module — v1 algorithms (still used for OCR, AOI shares, gaze path) |
| `saliency_v2.py` | v2 saliency: 8 channels, multi-face + centre prior, learned fusion (default in `run_analysis.py`; `--v1` for the original) |
| `validate_osie.py`, `metrics.py` | Scores v1 vs v2 vs baselines vs human agreement on OSIE human eye-tracking (CC, SIM, KLD, NSS, AUC) |
| `motion_v2.py` | v2 motion module: ego-motion compensation, bidirectional multi-scale correlators, contrast normalisation, top-K regions |
| `compare_motion.py`, `real_video_test.py` | Before/after stress tests (synthetic probes, real open footage) |
| `button_test.py` | Flat vs bordered CTA and colour pop-out probe |
| `motion_emd.py`, `game_motion_test.py`, `followups.py` | Original v1 motion module and the first stress tests (kept for comparison) |
| `run_analysis.py` | Runnable example — start here to run it yourself |
| `download_models.py` | Fetches the required face-detection model |
| `requirements.txt` | Dependencies |

## Honesty note

This is a **heuristic screening tool**. v2 saliency is now scored against real human eye-tracking on natural-scene photos (OSIE; CC 0.45 vs 0.88 human-to-human) but not on ads, and motion results are not validated against human gaze on video. See `NOTES.md` Sections 9, 15-17 for the full limitations list — read that before presenting this as more than it is.
