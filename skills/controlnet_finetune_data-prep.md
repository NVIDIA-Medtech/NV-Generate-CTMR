---
name: controlnet_finetune_data-prep
description: How to turn your own original images + original label masks into the preprocessed files (VAE embeddings, VISTA-3D pseudo labels, combined labels) needed to finetune the CT ControlNet on a NEW class. Covers remapping an unseen user label onto any unclaimed integer index (0–255; the "dummy" placeholders are just one option) in label_dict.json, building the JSON data list, and the fold/weighted_loss settings. Trigger when the user says "I only have images and masks", "how do I prepare data to finetune the ControlNet", "how do I add my own class/tumor/lesion", "what index do I give my new label", "how do I make the *_emb / pseudo_label / combined_label files", or wants to reproduce the C4KC-KiTS example on their own data. CT-only.
---

# Preparing your own data for ControlNet finetuning

This skill is for the case where **you only have two things per case**:

- an **original image** (`*.nii.gz`), and
- an **original label mask** (`*.nii.gz`) that contains one or more classes — including at least one **new class** the released model has never seen (a tumor, lesion, device, etc.).

It explains how to produce the **three derived files** the ControlNet training loop actually consumes, and — the part that trips people up — **how to remap your new label onto a `dummy` placeholder index** so the MAISI label vocabulary stays consistent.

> **CT-only.** The released ControlNet checkpoints were trained on CT masks; there is no MR ControlNet in this repo. The reference walkthrough is [data/README.md §4.3](../data/README.md#43-example-finetuning-on-a-new-dataset) (the C4KC-KiTS Kidney-Tumor example, which maps its new class to index `129`).

## What you have → what you need

```text
            |-image.nii.gz                # original image          ← you have this
KiTS-000* --|-mask.nii.gz                 # original mask           ← you have this
            |-image_emb.nii.gz            # VAE-encoded embedding         (Step 1)
            |-mask_pseudo_label.nii.gz    # VISTA-3D whole-body labels    (Step 2)
            |-mask_combined_label.nii.gz  # pseudo labels + your remapped mask  (Step 3)
```

The training loop reads only `*_emb.nii.gz` (as `image`) and `*_combined_label.nii.gz` (as `label`). Steps 1–3 below produce them.

---

## Step 1 — Image embedding (`*_emb.nii.gz`)

VAE-encode each original image with `scripts/diff_model_create_training_data.py`. It resamples every image to the nearest multiple of 128 per axis, runs the autoencoder encoder (sliding-window, AMP), and writes `<image>_emb.nii.gz`.

Point an `environment_*` config at your data and use **`autoencoder_v1.pt`** (the CT ControlNet's autoencoder):

```json
{
    "trained_autoencoder_path": "./models/autoencoder_v1.pt",
    "data_base_dir": "./datasets/my_dataset",
    "embedding_base_dir": "./datasets/my_dataset",
    "json_data_list": "./datasets/my_dataset.json"
}
```

```bash
python -m scripts.diff_model_create_training_data \
    -t ./configs/config_network_rflow.json \
    -c ./configs/config_maisi_diff_model_rflow-ct.json \
    -e ./configs/<your_env>.json -g 1
```

> The data list for **this step** must carry a `modality` field per entry (e.g. `"ct"`) — it drives intensity normalization. See [modality_mapping.json](../configs/modality_mapping.json) for valid values. Encoding up front (instead of inside the training loop) is what keeps GPU memory low during finetuning.

## Step 2 — Pseudo labels (`mask_pseudo_label*.nii.gz`)

Run [VISTA-3D](https://github.com/Project-MONAI/VISTA) on each **original image** to produce a whole-body segmentation already expressed in the MAISI 132-class vocabulary. This gives the ControlNet anatomical context your ROI-only mask lacks.

**VISTA-3D is a separate tool — not part of this repo.** Install/run it independently and save its output next to each case as `mask_pseudo_label*.nii.gz`.

## Step 3 — Remap your mask, then combine (`mask_combined_label*.nii.gz`)

This is the key step. The combined mask = the VISTA pseudo label, with **your mask written on top** in MAISI indices. Two sub-steps:

### 3a. Remap your label values to MAISI indices

For every class in your original mask, decide its MAISI index:

- **Class already exists in MAISI** (liver, kidney, spleen, …) → use its existing index from [label_dict.json](../configs/label_dict.json) (e.g. `liver=1`, `left kidney=14`, `right kidney=5`).
- **Class is new / unseen** (your tumor, lesion, device — not in the vocabulary) → assign it to **any unclaimed integer below 256**.

**The rule for a new class: pick any integer in `0–255` that isn't already used and isn't reserved.** ControlNet supports up to 256 labels (`0–255`). Don't reuse an index a real anatomical class already owns, don't use `0` (background), and don't use `200` (body envelope). Anything else is fair game — there is nothing special about the `dummy` names below; they're just **pre-named convenience slots**.

**Free indices that collide with nothing** (no existing label uses them): **`133–199`** and **`201–255`** — 123 values, wide open. Pick from here if you want zero risk of clobbering an existing class.

**Pre-named `dummy` placeholders** in `label_dict.json` (handy because they already have an entry you can rename — see [§ below](#tell-the-training-config-about-your-new-class)):

| Placeholder | Index | |
|---|---|---|
| `dummy6` | **129** | the slot the C4KC Kidney-Tumor example uses |
| `dummy7` | 130 | |
| `dummy8` | 131 | |
| `dummy1`–`dummy5` | 2, 16, 18, 20, 21 | lower indices, interspersed among real organs |

> Whatever index you pick — a `dummy` slot or a fresh integer like `150` — **add a named entry for it in `label_dict.json`** and set it as `weighted_loss_label` (see below). The only hard constraints are: integer, `0–255`, not already claimed, not `0`, not `200`.

### 3b. Combine pseudo label + remapped mask

Overlay your remapped mask onto the VISTA pseudo label and keep the **body envelope (`200`)** present. The repo provides the remap building block; the overlay is a small step you assemble:

```python
import torch
from scripts.augmentation import remap_labels   # remap_labels(tensor, {old_value: new_index})

# your_mask: integer label tensor from mask.nii.gz
# pseudo:    integer label tensor from mask_pseudo_label.nii.gz (already MAISI vocab, includes body=200)

# 3a: liver(1) and right-kidney(2) exist in MAISI; my new lesion(3) is unseen -> dummy6 (129)
remapped = remap_labels(your_mask, {1: 1, 2: 5, 3: 129})

# 3b: write your foreground classes on top of the pseudo label, leave its background/body intact
combined = pseudo.clone()
combined[remapped > 0] = remapped[remapped > 0]
# save `combined` as mask_combined_label*.nii.gz (same affine/spacing as the pseudo label)
```

(`scripts/utils.py::remap_labels` does the same thing but reads a JSON of `[orig, target]` pairs — handy if you prefer a config file. Use whichever fits your pipeline.)

---

## Build the JSON data list

One JSON pairs each embedding with its combined label. Paths are **relative to `data_base_dir`**:

```python
{
    "training": [
        {
            "image": "KiTS-000/image_emb.nii.gz",        # from Step 1
            "label": "KiTS-000/mask_combined_label.nii.gz",  # from Step 3
            "dim": [512, 512, 512],                        # resampled volume size — informational
            "spacing": [1.0, 1.0, 1.0],                    # voxel spacing
            "top_region_index": [0, 1, 0, 0],              # body-region one-hot (CT)
            "bottom_region_index": [0, 0, 0, 1],           # body-region one-hot (CT)
            "modality": "ct",                              # required by Step 1's embedding script
            "fold": 0
        }
        // ...
    ]
}
```

> **Fold split (read carefully — easy to get backwards):** an item is held out for **validation** when its `"fold"` **equals** `fold` in `config_maisi_controlnet_train*.json` (default `0`), and used for **training** otherwise. So if *every* item is `fold: 0` with the default config, your **training set is empty**. Spread items across folds (`0`, `1`, `2`, …) so the held-out fold gives a non-empty validation set and the rest train.

## Tell the training config about your new class

In `configs/config_maisi_controlnet_train*.json`, set `weighted_loss_label` to the index/indices you chose in Step 3a so the loss emphasizes your new structure:

```json
"weighted_loss_label": [129],
"weighted_loss": 100
```

In `configs/label_dict.json`, rename the placeholder you used so the vocabulary is self-documenting (optional but recommended):

```diff
-    "dummy6": 129,
+    "kidney tumor": 129,
```

## Launch finetuning

Once the three files and JSON exist and the configs point at them, run the ControlNet training command from [docs/training.md → 3D ControlNet Training](../docs/training.md#3d-controlnet-training). In short:

```bash
network="rflow"; generate_version="rflow-ct"
python -m scripts.train_controlnet \
    -t ./configs/config_network_${network}.json \
    -c ./configs/config_maisi_controlnet_train_${generate_version}.json \
    -e ./configs/environment_maisi_controlnet_train_${generate_version}.json -g 1
```

## Gotchas checklist

- [ ] Embeddings made with **`autoencoder_v1.pt`** (not v2) for the CT ControlNet.
- [ ] Each combined label still contains the **body envelope (`200`)** on non-organ body voxels.
- [ ] New classes remapped to **any unclaimed integer in `0–255`** (free ranges `133–199` / `201–255`, or a `dummy` slot like `129`); existing organs remapped to their real MAISI indices. Never reuse a claimed index, `0`, or `200`.
- [ ] `weighted_loss_label` matches the index you assigned; `label_dict.json` updated if you go past the 8 dummies.
- [ ] Items spread across **multiple folds** so the held-out (validation) fold isn't the whole dataset.
- [ ] `modality` field present (needed by the Step-1 embedding script).
