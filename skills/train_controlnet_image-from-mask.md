---
name: train_controlnet_image-from-mask
description: How to train a 3D image-from-mask ControlNet with scripts.train_controlnet — for any modality (CT or MR) and any label vocabulary. Focuses on the training-data format the loop consumes (paired VAE latent embedding + combined integer label mask, the JSON data list schema, required vs conditional fields, the label→8-bit binary condition, folds, modality, multi-dataset lists) and the spatial relationship between the latent embedding and the label (the label must be on the same grid as the image that was VAE-encoded = 4× the latent per axis; the ControlNet downsamples it internally). Covers configs, single/multi-GPU launch, key knobs (n_epochs, lr, fold, weighted_loss), and outputs. Trigger when the user asks "how do I train the ControlNet", "what is the training data format / JSON schema", "what shape must the label mask be vs the embedding", "do I resample the label to match the latent and how", "how do I train a ControlNet for a new modality or vocabulary", or "which config/command trains train_controlnet".
---

# Training an image-from-mask ControlNet

This skill covers training a 3D ControlNet with `scripts.train_controlnet`, for **any modality (CT or MR) and any label vocabulary**. It sits between:

1. **Prepare data** → [`finetune_image-from-mask_data-prep`](finetune_image-from-mask_data-prep.md) produces the two per-case files (`*_emb.nii.gz`, `*_combined_label.nii.gz`) + the JSON data list.
2. **Train** ← *you are here.*
3. **Generate** → [`infer_image-from-mask`](infer_image-from-mask.md) runs the trained checkpoint on a mask.

## How training is wired

`scripts/train_controlnet.py` needs a **frozen pretrained diffusion U-Net** (`trained_diffusion_path`) — *not* the ControlNet. The training:

1. Loads the frozen DM, reads its `scale_factor`, sets every U-Net param `requires_grad = False`.
2. Builds the ControlNet and **initializes it by copying the DM's encoder/mid weights** (`copy_model_state`).
3. Optionally loads a ControlNet checkpoint from **`existing_ckpt_filepath`** to continue/finetune. Leave it `null` to start from the DM-copied init; set it to a checkpoint path to warm-start — your choice.
4. Trains **only** the ControlNet (AdamW, L1 loss, `PolynomialLR` power 2.0), saving after every epoch.

The DM you supply must match your modality (`diff_unet_3d_rflow-ct` for CT; `diff_unet_3d_rflow-mr` for MR). The **autoencoder is not loaded by this script** — it is used earlier, in data-prep, to produce the `*_emb.nii.gz` embeddings; by training time the `image` is already a latent. (`trained_autoencoder_path` may appear in the env config but the trainer ignores it.)

---

## Training data format

The loop consumes **one thing per training case: a `(image, label)` pair**, listed in a JSON data list. `scripts/utils.py::prepare_maisi_controlnet_json_dataloader` loads it; the loader does **no spatial resampling** — the files must already be on compatible grids (see below).

### Per-case files

| JSON key | File | On-disk shape | What it is |
|---|---|---|---|
| `image` | `*_emb.nii.gz` | `[X/4, Y/4, Z/4, 4]` | The **VAE latent embedding** of the real image — a **4-channel** float volume saved channel-*last* (a 4D NIfTI), produced by the AE encoder in data-prep Step 1. |
| `label` | `*_combined_label.nii.gz` | `[X, Y, Z]` | The **combined integer label mask** — a plain **3D** volume (no channel axis): MAISI-vocabulary organ labels + body envelope `200` + your ROI/class, on the **same grid as the image that was VAE-encoded** (so its spatial size is 4× the latent). |

The loader (`LoadImaged(..., ensure_channel_first=True)`) moves/adds the channel axis to the **front** in memory, so downstream the tensors are channel-first: image → `[4, X/4, Y/4, Z/4]`, label → `[1, X, Y, Z]` (`C=1`, a single integer channel). Shapes in the rest of this section use that channel-first `[C, X, Y, Z]` form.

The autoencoder downsamples by **4× per spatial axis**, so a `[512, 512, 128]` image encodes to a `[4, 128, 128, 32]` latent (channel-first), while its `label` stays on the image grid → `[1, 512, 512, 128]`, i.e. 4× the latent per spatial axis. (Inside the loop that single label channel is expanded to **8 channels** by `binarize_labels` → `[8, 512, 512, 128]` — see [the condition](#the-label-vocabulary-and-how-it-becomes-the-condition) below.)

### JSON data list schema

`data_base_dir` + `json_data_list` are **lists** (one entry each per dataset — you can mix datasets). Paths inside each JSON are **relative to** that JSON's `data_base_dir`. The `"training"` array holds the cases:

```jsonc
{
  "training": [
    {
      "image": "case_000/img_emb.nii.gz",       // REQUIRED — latent embedding, on disk [X/4, Y/4, Z/4, 4]
      "label": "case_000/combined_label.nii.gz", // REQUIRED — integer mask, on disk [X, Y, Z]
      "spacing": [1.0, 1.0, 1.0],                // REQUIRED — voxel spacing of the encoded image (mm)
      "modality": "ct",                          // REQUIRED when the network uses modality conditioning
      "fold": 0,                                 // REQUIRED — cross-val fold (see semantics below)
      "dim": [512, 512, 128],                    // informational only (not read by the loader)
      "top_region_index": [0, 1, 0, 0],          // ddpm-ct ONLY — omit for rflow-ct / rflow-mr
      "bottom_region_index": [0, 0, 0, 1]        // ddpm-ct ONLY — omit for rflow-ct / rflow-mr
    }
    // ... more cases
  ]
}
```

Field notes (from the loader transforms):

- **`image` / `label`** — required. `label` is oriented to **RAS** and cast to **`long`** (integer); `image` is used as-is.
- **`spacing`** — required; converted to a float tensor and scaled ×100 as a conditioning input.
- **`modality`** — mapped to an integer via `modality_mapping_path` ([`configs/modality_mapping.json`](../configs/modality_mapping.json), e.g. `"ct"→1`, `"mri_t2"→10`). Required whenever the diffusion U-Net has modality conditioning — i.e. `diffusion_unet_def.num_class_embeds` is non-null, which gates `include_modality` (it's `128` in the shipped rflow/ddpm nets, so required there).
- **`fold`** *(easy to get backwards)* — an item is held out for **validation** when its `"fold"` **equals** `fold` in the training config (default `0`), and used for **training** otherwise. If every item is `fold: 0` with the default config, the **training set is empty**. Spread items across folds (`0`, `1`, `2`, …).
- **`top_region_index` / `bottom_region_index`** — 4-D body-region one-hots, needed **only** for `ddpm-ct` (`config_network_ddpm.json` sets `include_body_region: true`). `rflow` nets set it `false` and ignore them.

### The label vocabulary and how it becomes the condition

The `label` is a 1-channel **integer** mask. During training it is converted to the ControlNet condition by `binarize_labels` — an **8-bit binary encoding** → 8 channels (`conditioning_embedding_in_channels: 8`). That is why the ControlNet supports **up to 256 labels (0–255)**. Use the MAISI 132-class vocabulary + body `200`, plus any new-class index you assigned in data-prep. Values are integer class IDs — **never** continuous. (Label authoring, remapping, and adding a new class: see the data-prep skill.)

### Spatial relationship — label vs. embedding (important)

The **label must be exactly 4× the latent per spatial axis** (e.g. latent `[4, 128, 128, 32]` → label `[1, 512, 512, 128]`) and share the image's FOV/affine — the ControlNet downsamples the label 4× internally to add it to the latent, and there's no auto-resampling, so a mismatch errors out. Data prep puts the label on that grid; see [data-prep Step 4](finetune_image-from-mask_data-prep.md#step-4--put-the-combined-label-on-the-encoded-image-grid).

### Orientation — label and embedding must agree

The loader reorients the **label** to canonical **RAS** (`Orientationd(keys=["label"], axcodes="RAS")`) but loads the **image** embedding **as-is**. Standard embeddings are already RAS — `diff_model_create_training_data.py` orients each image to RAS before encoding — so the label→RAS step brings the label into that same frame, and the current label-only transform is enough.

Adding `"image"` to that transform is a harmless hardening: a no-op for standard (already-RAS) embeddings, and it would realign a non-standard, non-RAS embedding too — `Orientationd` only permutes/flips axes losslessly, so it can't corrupt latent values, as long as the embedding's affine faithfully encodes its orientation. The only unrecoverable case is an embedding whose affine doesn't match its voxels, which no affine-based tool can fix. Without that guard, a non-RAS embedding misaligns with the RAS label and the ControlNet trains on a spatially scrambled mask with **no error raised**.

---

## The three config files you pass

| Flag | File | Role |
|---|---|---|
| `-t` | `configs/config_network_${network}.json` | Network architecture (`rflow` or `ddpm`). Sets `include_body_region`. |
| `-c` | `configs/config_maisi_controlnet_train_${generate_version}.json` | `controlnet_train` hyper-parameters. |
| `-e` | `configs/environment_maisi_controlnet_train_${generate_version}.json` | Paths: weights, data, outputs. |
| `-g` | — | Number of GPUs (`1` = single-GPU, no DDP). |

### Environment config (`-e`) — key fields

```json
{
    "trained_autoencoder_path": "./models/autoencoder_v1.pt",       // present but NOT read by train_controlnet.py
    "trained_diffusion_path":  "./models/diff_unet_3d_rflow-ct.pt", // frozen DM (REQUIRED, non-null)
    "existing_ckpt_filepath":  null,                                // null, or a ControlNet ckpt to warm-start
    "exp_name": "my_controlnet",                                    // names the output checkpoints
    "model_dir":  "./models/",
    "tfevent_path": "./outputs/tfevent",
    "data_base_dir": ["./datasets/my_dataset"],                     // LIST
    "json_data_list": ["./datasets/my_dataset.json"],              // LIST (paired with data_base_dir)
    "modality_mapping_path": "./configs/modality_mapping.json"      // required when net uses modality
}
```

> To warm-start from a ControlNet checkpoint, set **`existing_ckpt_filepath`** — the training loop reads that key (train_controlnet.py:309), *not* `trained_controlnet_path`. (The stale `environment_maisi_controlnet_train.json` uses the old `trained_controlnet_path` key, which the trainer ignores — prefer the versioned `_rflow-ct` / `_rflow-mr` / `_ddpm-ct` env configs.)

### Training config (`-c`) — the `controlnet_train` knobs

The shipped `config_maisi_controlnet_train_rflow-ct.json` (values as delivered — this is the real default, **not** a disabled template):

```json
"controlnet_train": {
    "batch_size": 1,          // images trained whole (not patched); keep at 1
    "cache_rate": 0.0,        // MONAI CacheDataset ratio; raise if RAM allows
    "fold": 0,                // items whose "fold" == this validate; the rest train
    "lr": 1e-5,               // AdamW LR, polynomial-decayed to 0 over all steps
    "n_epochs": 100,          // total training epochs
    "weighted_loss_label": [23],  // ROI/class index(es) up-weighted in L1 (23 = lung tumor)
    "weighted_loss": 100,         // multiplier on those voxels; set to 1 to weight all voxels equally
    "use_region_contrasive_loss": true,   // rflow-ct default (see caveat below)
    "region_contrasive_loss_delta": 2,
    "region_contrasive_loss_weight": 0.01
}
```

- **`weighted_loss` / `weighted_loss_label`** — up-weight the L1 loss on a small/hard ROI (e.g. a tumor). Set `weighted_loss_label` to **your** ROI/class index (the CT config ships `[23]`; the MR/ddpm configs ship `[129]`). To disable emphasis entirely, use `"weighted_loss": 1` with `"weighted_loss_label": [null]`.
- **Region-contrastive loss** — on by default in `rflow-ct` (the `ddpm-ct` / `rflow-mr` configs omit these three keys, i.e. off). Its `remove_roi()` assumes a MAISI **tumor** ROI (`remove_tumors`); for any other vocabulary, edit `remove_roi()` in `train_controlnet.py` or set `use_region_contrasive_loss: false`.

## Launch

### Single GPU

```bash
network="rflow"; generate_version="rflow-ct"      # CT; or rflow-mr, or network=ddpm/generate_version=ddpm-ct

python -m scripts.train_controlnet \
    -t ./configs/config_network_${network}.json \
    -c ./configs/config_maisi_controlnet_train_${generate_version}.json \
    -e ./configs/environment_maisi_controlnet_train_${generate_version}.json \
    -g 1
```

> The single-GPU snippet in [docs/training.md](../docs/training.md#execute-training) mistakenly passes the `config_maisi_diff_model_*` / `environment_maisi_diff_model_*` files — those lack a `controlnet_train` block. Use the **`*_controlnet_train_*`** configs above.

### Multi-GPU (torchrun / DDP)

`-g` must equal `--nproc_per_node`; DDP kicks in whenever `-g > 1`.

```bash
export NUM_GPUS_PER_NODE=8
network="rflow"; generate_version="rflow-ct"
torchrun --nproc_per_node=${NUM_GPUS_PER_NODE} --nnodes=1 \
    --master_addr=localhost --master_port=1234 \
    -m scripts.train_controlnet \
    -t ./configs/config_network_${network}.json \
    -c ./configs/config_maisi_controlnet_train_${generate_version}.json \
    -e ./configs/environment_maisi_controlnet_train_${generate_version}.json \
    -g ${NUM_GPUS_PER_NODE}
```

## Version selection

"AE (data-prep)" is the autoencoder used earlier to build the embeddings; "DM (`-e`)" is the frozen diffusion U-Net this script loads.

| `generate_version` | Network (`-t`) | AE (data-prep) | DM (`-e`) | `include_body_region` |
|---|---|---|---|---|
| `rflow-ct` *(recommended CT)* | `config_network_rflow.json` | `autoencoder_v1` | `diff_unet_3d_rflow-ct` | `false` — omit region indices |
| `ddpm-ct` | `config_network_ddpm.json` | `autoencoder_v1` | `diff_unet_3d_ddpm-ct` | `true` — JSON needs `top/bottom_region_index` |
| `rflow-mr` | `config_network_rflow.json` | `autoencoder_v2` | `diff_unet_3d_rflow-mr` | `false` |

## GPU memory (whole-image training, per docs)

| Image size | Latent size | Peak memory |
|---|---|---|
| 256×256×128 | 4×64×64×32 | 5 G |
| 256×256×256 | 4×64×64×64 | 8 G |
| 512×512×128 | 4×128×128×32 | 12 G |
| 512×512×256 | 4×128×128×64 | 21 G |
| 512×512×512 | 4×128×128×128 | 39 G |
| 512×512×768 | 4×128×128×192 | 58 G |

## Outputs

Written to `model_dir`, named by `exp_name`, each holding `{epoch, loss, controlnet_state_dict}`:

- **`{exp_name}_current.pt`** — overwritten every epoch (latest).
- **`{exp_name}_best.pt`** — written when the epoch loss improves.

TensorBoard scalars go to `tfevent_path/exp_name`. To generate with the result, point the inference env's `trained_controlnet_path` at your `{exp_name}_best.pt` (the checkpoint format matches what inference loads) and follow [`infer_image-from-mask`](infer_image-from-mask.md).

## Gotchas checklist

- [ ] Per case: `image` = 4-channel latent embedding, `label` = 1-channel integer mask at **4× the latent per axis** (same FOV/affine as the encoded image).
- [ ] Label resampled with **nearest-neighbor** only — never linear/bspline.
- [ ] `trained_diffusion_path` (frozen DM) set and **required**, matching the modality; the AE is a data-prep dependency, not read by this script. (`existing_ckpt_filepath`: `null`, or a ControlNet ckpt to warm-start.)
- [ ] `-c` / `-e` are the **`*_controlnet_train_*`** configs, not `*_diff_model_*`.
- [ ] Items spread across **multiple folds** so the held-out (validation) fold isn't the whole dataset.
- [ ] `modality` present in every entry + `modality_mapping_path` set (net uses modality conditioning).
- [ ] `ddpm-ct` only: entries carry `top_region_index` / `bottom_region_index`; `rflow` ignores them.
- [ ] `data_base_dir` / `json_data_list` are **lists**; label values are integers in `0–255`.
