# Runbook — from a prepared laptop to a published model

Everything up to this point is reproducible by `scripts/00`–`02`. This document
covers the three steps that need **your credentials**, so they cannot be
automated here.

Current local state after data prep:

| Artifact | Value |
|---|---|
| Charades videos | 9,848 (15 GB) at `data/raw/Charades_v1_480/` |
| Clip manifest | 16,146 clips — 13,994 train / 601 val / 1,551 test |
| Frames | **141,576** JPEGs, 3.1 GB, 0 extraction failures |
| Kaggle payload | 3.32 GB in `data/kaggle_upload/` |

---

## 1. Upload the frames as a PRIVATE Kaggle dataset

Get an API token at <https://www.kaggle.com/settings> → *Create New Token*,
then:

```bash
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# dataset-metadata.json already pins isPrivate: true
.venv/bin/kaggle datasets create -p data/kaggle_upload --dir-mode skip
```

Upload is ~3.3 GB. Verify it shows as **Private** on Kaggle before continuing —
Charades may not be redistributed (see [DATA_LICENSE.md](../DATA_LICENSE.md)).

To refresh later: `kaggle datasets version -p data/kaggle_upload -m 'refresh' --dir-mode skip`

## 2. Run the notebook on Kaggle

1. New Notebook → **File → Import Notebook** → `notebooks/train_kaggle.ipynb`
2. **Settings → Accelerator → GPU T4 ×2**
3. **Add Input** → your `charades-alert-frames` dataset
4. **Add-ons → Secrets** → add `HF_TOKEN` (write-scoped) if you want step 7 to push
5. Edit the `REPO_ID` in the last cell to your HF username
6. Run All

Expected wall clock on one T4:

| Stage | Time |
|---|---|
| Extract tars + install | ~5 min |
| Smoke test (16 samples) | ~3 min |
| Train, 2 epochs × 13,994 clips | **4–8 h** |
| 8 eval cells × 1,551 clips | ~1.5–3 h |

That exceeds a single 12 h session only if training runs long. Both training and
evaluation are **resumable**: training picks up the newest checkpoint
automatically, and each eval cell skips if its JSONL already exists. If the
session dies, re-run the notebook from the top.

GPU quota is 30 h/week, so the full matrix fits in one week with room to retry.

**If you want results sooner**, add `--limit 600` to the `03_generate.py` calls
for a first pass, then re-run without it later (use `--overwrite`).

## 3. Publish

Step 7 of the notebook runs:

```bash
python scripts/06_push_to_hf.py \
  --adapter runs/t4_8frame/adapter \
  --repo-id YOUR_HF_USERNAME/qwen2vl-2b-charades-alerts-lora \
  --results reports/results.json \
  --run-config runs/t4_8frame/config.json
```

Add `--dry-run` first to inspect the generated card without uploading. The card
is built **from `results.json`**, so its numbers always match the run. Uploads
are allowlisted to the adapter, processor config, card and result tables — no
frames.

Then commit the results back:

```bash
# locally, after downloading results.tar from the Kaggle session
tar xf results.tar
git add -f reports/RESULTS.md reports/results.json
git commit -m "Add baseline vs fine-tuned results, ablation and failure taxonomy"
git push
```

`reports/` is gitignored by default so partial runs do not get committed; the
`-f` is deliberate and only for the final tables.

---

## Troubleshooting

**`ValueError: ... mm_token_type_ids is missing`** — you are on a transformers
build where the collator must forward it. Already handled; make sure you are on
the pinned `transformers==5.19.0` from the notebook's install cell.

**OOM on the 16-frame eval arm** — drop `--batch-size` to 2, or lower
`--max-tokens-per-frame` to 160. A 16-frame clip is ~1,450 tokens, so this
should not happen on a 16 GB T4 at batch 4.

**`Attempting to unscale FP16 gradients`** — LoRA parameters are not in fp32.
`attach_lora` does this cast; do not bypass it.

**Training loss is NaN early** — fp16 instability. Lower `lr` to 5e-5 or set
`precision: 4bit` in the config (QLoRA is more numerically forgiving on T4).

**Kaggle `pip install` breaks CUDA** — never reinstall `torch` on Kaggle. The
install cell deliberately uses `--no-deps` for transformers/peft.
