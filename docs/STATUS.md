# Status — paused 2026-10-17

Data pipeline is **done and verified**. Training has **not been run**. No results
exist yet, and nothing in this repo reports a performance number.

Resume at step 1 below. Full instructions for each step are in
[RUNBOOK.md](RUNBOOK.md); this file is just the bookmark.

---

## Done

| | |
|---|---|
| Code | 12 modules, 7 scripts, 1 notebook, **33 tests passing** |
| Charades | 9,848 videos downloaded (16,339,546,533 bytes, size-exact) and unzipped |
| Manifest | 16,146 clips — 13,994 train / 601 val / 1,551 test, all 157 classes |
| Frames | **141,576 decoded, 0 failures** (18.6 min @ 14.5 clips/s) |
| Leakage | 0 subject and 0 video overlap across all three split pairs |
| Ablation control | Verified byte-identical: 1/4/8-frame views == strided 16-frame cache |
| Kaggle payload | 3.32 GB packed and entry-count verified |
| Published | <https://github.com/ksujith/descriptive-alerts-vlm> (public) |

## Not done

- No training run — no `runs/*/adapter` anywhere
- No predictions — `reports/preds/` is empty
- HF not logged in; no Kaggle API credentials on this machine
- Nothing pushed to the Hub

---

## Local artifacts (on this machine only, NOT in git)

These took roughly 2.5 h to produce, mostly download time. **Do not delete
`data/`** unless you want to redo that.

| Path | Size | Regenerate with |
|---|---|---|
| `data/raw/Charades_v1_480.zip` | 15 G | `scripts/00_download_videos.sh` (resumable) |
| `data/raw/Charades_v1_480/` | 15 G | `unzip` of the above |
| `data/raw/Charades/` | 9.2 M | annotations zip (3.5 MB download) |
| `data/interim/clips.csv` | 4.4 M | `scripts/01_build_manifest.py` (seeded, deterministic) |
| `data/frames/` | 3.1 G | `scripts/02_extract_frames.py` (resumable, ~19 min) |
| `data/kaggle_upload/` | 3.1 G | `scripts/05_pack_for_kaggle.py` (~2 min) |

The zip and the unzipped videos are both 15 G; once `data/frames/` exists, the
zip can be deleted to reclaim space without blocking training.

Everything is deterministic — `01_build_manifest.py` uses a fixed seed — so a
rebuild reproduces the identical manifest and therefore the identical splits.

---

## Resume here

1. **Kaggle credentials + upload** (~3.3 GB, must stay private)
   ```bash
   mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json && chmod 600 ~/.kaggle/kaggle.json
   .venv/bin/kaggle datasets create -p data/kaggle_upload --dir-mode skip
   ```
2. **Run `notebooks/train_kaggle.ipynb`** on Kaggle — GPU T4 ×2, attach the
   dataset, add an `HF_TOKEN` secret, Run All. Budget 4–8 h training + 1.5–3 h
   eval. Both resume after a session timeout.
3. **Push** — notebook step 7, then commit `reports/RESULTS.md` and
   `reports/results.json` back with `git add -f` (reports/ is gitignored so
   partial runs don't get committed).

## Open decision

The eval matrix is 8 cells × 1,551 clips ≈ 1.5–3 h. Adding `--limit 600` to the
`03_generate.py` calls cuts that to well under an hour for a first look, at the
cost of wider confidence intervals. Re-run later without the limit (plus
`--overwrite`) for final numbers. Not yet wired in — currently the notebook runs
the full test set.

## Known-good versions

Validated against `transformers==5.19.0`, `peft==0.21.2`, `torch==2.14.1`,
`av==19.0.1`. The notebook pins transformers and peft with `--no-deps`; never
reinstall `torch` on Kaggle, it breaks the CUDA build.
