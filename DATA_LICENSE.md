# Data licensing and what this repository does about it

## The constraint

This project trains on [Charades](https://prior.allenai.org/projects/charades)
(Sigurdsson et al., 2016), distributed by the Allen Institute for AI. Its
licence states, in part:

> This software and data is available for general use by academic or non-profit,
> or government-sponsored researchers. It may also be used for evaluation
> purposes elsewhere. This license does not grant the right to use this software
> or any derivation of it in a for-profit enterprise.
>
> **This license does not grant the right to modify and publicly release the
> data in any form.**
>
> **This license does not grant the right to distribute the data to a third
> party in any form.**
>
> The subjects in this data should be treated with respect and dignity. This
> license only grants the right to publish short segments or still images in an
> academic publication where necessary to present examples, experimental
> results, or observations.

The full text ships with the annotation release as `license.txt`.

## What that rules out, and what it does not

| Action | Allowed here? | Reasoning |
|---|---|---|
| Train a model on Charades locally | Yes | Research/evaluation use. |
| Publish the trained **LoRA adapter** | Yes | Model weights are not the data. The model card states the non-commercial restriction so it propagates. |
| Publish **extracted frames** as an HF or Kaggle dataset | **No** | Public release of a modified form of the data. Explicitly withheld. |
| Publish the **clip manifest** (ids, timestamps, labels) | **No** | It is derived from the annotations. The repo ships the *script* that regenerates it instead. |
| Put frames on a rented GPU to train | Yes, kept private | Any cloud training places data on third-party hardware; that is inherent to Colab/Kaggle/RunPod alike. What we avoid is *public* redistribution. |
| Commercial use of the adapter | **No** | The licence excludes for-profit use of derivations. |

## How the repository is arranged around this

- **No data is committed.** `.gitignore` excludes `data/`, including the
  manifest CSV, frames and video archives.
- **Regeneration, not redistribution.** `scripts/00_download_videos.sh` fetches
  Charades from AI2 under your own acceptance of its licence, and
  `scripts/01_build_manifest.py` rebuilds the exact clip manifest
  deterministically (fixed seed), so results are reproducible without shipping
  the data.
- **The Kaggle packer forces privacy.** `scripts/05_pack_for_kaggle.py` writes a
  `dataset-metadata.json` with `isPrivate: true` and uploads nothing itself.
- **The Hub push is allowlisted.** `scripts/06_push_to_hf.py` uploads only the
  adapter, tokenizer/processor config, model card and results tables — by
  explicit `allow_patterns`, not by exclusion.

## On the people in the videos

Charades subjects consented to crowd-sourced data collection for activity
research. The licence asks that they be treated with respect and dignity. This
repository publishes no frames, no crops and no subject identifiers, and the
model card forbids use for surveillance of people without consent.

## If you need commercial rights

Contact the Allen Institute for AI about Charades licensing, or retrain the same
pipeline on a permissively licensed corpus. The code here is dataset-agnostic:
`src/dalerts/charades.py` is the only module that knows the Charades schema.
