---
name: infer_mask-image-paired
description: How to run paired mask + image generation with NV-Generate-CTMR. Generates a 3D mask (either from anatomy_size or by retrieving a real training mask) and then a paired CT/MR image conditioned on that mask via ControlNet. Trigger when the user asks "how do I generate a mask and image together", "how does LDMSampler work", "what does scripts.inference do", or wants help running the README §2.3 CT Paired Image/Mask command.
---

# Mask + image paired inference

This skill covers the **paired generation** pipeline: mask first, then image conditioned on that mask. The CLI is `scripts.inference`, which instantiates `LDMSampler` and calls `sample_multiple_images`. This is the path used in README §2.3 (CT Paired Image/Mask Generation).

Two algorithms run sequentially:

1. **Mask stage** — see the `infer_mask-generation` skill.
2. **Image stage** — see the `infer_image-from-mask` skill.

This skill explains how they're chained, the LDMSampler state required, and the configuration knobs.

## Quick Start command

```bash
export MONAI_DATA_DIRECTORY="./temp_work_dir"
network="rflow"                       # or "ddpm"
generate_version="rflow-ct"           # or "ddpm-ct"
python -m scripts.inference \
    -t ./configs/config_network_${network}.json \
    -i ./configs/config_infer.json \
    -e ./configs/environment_${generate_version}.json \
    --random-seed 0 --version ${generate_version}
```

> ⚠️ **`ddpm-ct` requires `num_inference_steps = 1000`** (vs 30 for `rflow-ct`). The notebook auto-applies this when `generate_version == "ddpm-ct"` (see cell 12). If you call the API directly, set this explicitly — DDPM scheduler will not produce usable output with fewer steps. This is 33× slower than `rflow-ct` but produces equivalent quality.

Three configs are passed:

- `-t` network architecture (`config_network_rflow.json` or `config_network_ddpm.json`).
- `-i` inference parameters (`config_infer.json` — `body_region`, `anatomy_list`, `output_size`, `spacing`, `controllable_anatomy_size`, etc.).
- `-e` environment paths (`environment_rflow-ct.json` or `environment_ddpm-ct.json` — checkpoint paths, label dicts, mask database).

## How `LDMSampler.sample_multiple_images` chooses the mask path

```text
controllable_anatomy_size non-empty?
            │
   ┌────────┴─────────┐
  YES                 NO
   │                   │
   ▼                   ▼
prepare_anatomy_size_  find_masks(body_region, anatomy_list, ...)
condition()            (look up real training masks; resample if needed)
   │                   │
   ▼                   ▼
sample_one_mask()      read_mask_information(mask_file)
(diffusion-generated)  (no diffusion, just load + transform)
   │                   │
   └────────┬──────────┘
            ▼
   prepare_one_mask_and_meta_info()  (assign 1.5mm iso affine, derive
                                      top/bottom_region_index)
            │
            ▼
   sample_one_pair()  (ControlNet + image DM — see infer_image-from-mask skill)
            │
            ▼
   quality_check_ct(image, mask)
            │
        passed?
            │
   ┌────────┴────────┐
  YES               NO
   │                 │
save image+label   re-generate (up to LDMSampler.max_try_time=2 retries)
```

### Two paths in detail

**Path A — `controllable_anatomy_size` non-empty** (diffusion-generated mask):

- User provides e.g. `[("pancreas", 0.5), ("liver", 0.7)]` in `config_infer.json`.
- `prepare_anatomy_size_condition` builds the 10-d vector (see `infer_mask-generation` skill).
- `sample_one_mask` runs the mask DDPM.
- Result is at fixed shape `256×256×256 × 1.5mm iso` (the mask DM's pretrained shape).
- `ensure_output_size_and_spacing` resamples to the user's requested `output_size` + `spacing`.

**Path B — `controllable_anatomy_size` empty** (real training mask):

- `find_masks` queries `configs/all_mask_files_*.json` for masks matching `body_region` + `anatomy_list` + `spacing` + `output_size`.
- If no exact match, `find_closest_masks` picks the closest by FOV / dim / spacing.
- `read_mask_information` loads the mask via `val_transforms` (LoadImaged + Orientationd("RAS") + spacing scaling).
- Optional `augmentation()` applies training-style mask augmentation if `if_aug` is set.

Both paths then call `sample_one_pair` for the image stage.

## `LDMSampler.__init__` — required state

| Group | Argument | Source |
|---|---|---|
| Mask DM | `mask_generation_autoencoder` | `models/mask_generation_autoencoder.pt` |
| Mask DM | `mask_generation_diffusion_unet` | `models/mask_generation_diffusion_unet.pt` |
| Mask DM | `mask_generation_noise_scheduler` | DDPM scheduler (from network def) |
| Mask DM | `mask_generation_scale_factor` | `1.0055984258651733` |
| Mask DM | `mask_generation_latent_shape` | `(4, 64, 64, 64)` for 256³ output |
| Image DM | `autoencoder`, `diffusion_unet`, `controlnet` | variant-specific checkpoints under `models/` |
| Image DM | `noise_scheduler` | RFlow (rflow-ct/mr) or DDPM (ddpm-ct) |
| Image DM | `scale_factor`, `latent_shape` | from the variant's network config |
| Mask DB | `all_mask_files_json`, `all_mask_files_base_dir` | for Path B only |
| Vocabularies | `label_dict_json`, `label_dict_remap_json` | `configs/label_dict.json`, `configs/label_dict_124_to_132.json` |
| Anatomy size DB | `all_anatomy_size_conditions_json` | `configs/all_anatomy_size_conditions.json` (used by Path A) |
| QC | `real_img_median_statistics` | `configs/image_median_statistics_ct.json` (CT-only quality check) |
| User intent | `body_region`, `anatomy_list`, `controllable_anatomy_size`, `output_size`, `spacing`, `modality` | from `config_infer.json` |
| Other | `device`, `output_dir`, `num_inference_steps`, `cfg_guidance_scale_tumor`, etc. | runtime / config |

## `dim` and `spacing` — same FOV rules as image-only

> ⚠️ **FOV (= `dim × spacing`) is the #1 quality knob.** See the **"Why FOV matters"** section at the top of [`infer_image-only.md`](infer_image-only.md) — same warning applies here. Out-of-distribution FOVs produce unusable output even when the validator accepts the inputs.

The mask + image pipeline uses **the same** `output_size` and `spacing` constraints as image-only inference — see the `infer_image-only` skill for the table of recommended `(dim, spacing)` per anatomical target and the hard constraints from `check_input_ct` / `check_input_mr`.

Additional FOV considerations specific to the paired pipeline:

- The **mask DM** was pretrained at **256³ × 1.5 mm iso** (= 384 mm cube FOV). Generating a mask at significantly different shape forces the `ensure_output_size_and_spacing` resampling, which degrades label boundaries. Stay at or near 256³ × 1.5mm for Path A.
- For Path B (mask DB lookup), the candidate masks are themselves drawn from a training-FOV distribution — `find_closest_masks` picks the closest matches, but the closer your requested FOV is to a mode of that distribution, the less reshaping is needed.

## How to configure a run

The five things you actually set in `config_infer.json`:

### 1. `modality` → driven by your anatomy

Pick the modality code matching what you want to generate (full list in `configs/modality_mapping.json`). The mask-image paired pipeline in this skill is **CT-only** (the mask DM and ControlNet are CT-only — no MR ControlNet exists), so for this pipeline `modality = 1`. For MR image-only generation use [`infer_image-only`](infer_image-only.md).

### 2. `output_size` and `spacing` → from FOV

FOV (mm per axis) is the model's anchoring signal — it must match the training-data distribution for your anatomy. Pick FOV first, then split it into `output_size` (voxels) and `spacing` (mm/voxel):

```text
spacing[i] = FOV[i] / output_size[i]
```

For recommended FOVs per CT body region, see `docs/inference.md#recommended-spacing-for-ct`. For MR (image-only path) see the recommended-FOV table in `docs/inference.md`. The hard constraints (`check_input_ct`) are documented in [`infer_image-only`](infer_image-only.md).

### 3. `autoencoder_sliding_window_infer_size`, `autoencoder_sliding_window_infer_overlap`, `autoencoder_tp_num_splits` → from GPU memory + output_size

The image AE decodes the latent in tiles (sliding-window) to fit memory. Three knobs control the memory/speed/quality trade-off:

- `autoencoder_sliding_window_infer_size` — ROI per tile, must be divisible by 16. Larger = fewer tiles, faster, more VRAM.
- `autoencoder_sliding_window_infer_overlap` — `[0, 1)`. Higher = smoother seams, more compute (each voxel decoded by more tiles).
- `autoencoder_tp_num_splits` — tensor-parallel splits inside each AE forward (`∈ {1, 2, 4, 8, 16}`). Higher = lower per-GPU VRAM, slower.

Validated presets (drawn from the `configs/config_infer_<XXg>_<dim>.json` files — pick the row matching your GPU + target `output_size`):

| GPU mem | `output_size` | `spacing` (mm) | `autoencoder_sliding_window_infer_size` | `..._overlap` | `autoencoder_tp_num_splits` | `num_inference_steps` | Config file |
|---|---|---|---|---|---|---|---|
| 16 GB | 256×256×128 | 1.5, 1.5, 4.0 | [96, 96, 96] | 0.25 | 2 | 30 | `config_infer_16g_256x256x128.json` |
| 16 GB | 256×256×256 | 1.5, 1.5, 2.0 | [48, 48, 64] | 0.6666 | 4 | 30 | `config_infer_16g_256x256x256.json` |
| 16 GB | 512×512×128 | 0.75, 0.75, 4.0 | [64, 64, 32] | 0.5 | 2 | 30 | `config_infer_16g_512x512x128.json` |
| 24 GB | 256×256×256 | 1.5, 1.5, 2.0 | [64, 64, 64] | 0.25 | 4 | 1000 (DDPM) | `config_infer_24g_256x256x256.json` |
| 24 GB | 512×512×128 | 0.75, 0.75, 4.0 | [80, 80, 32] | 0.4 | 2 | 30 | `config_infer_24g_512x512x128.json` |
| 24 GB | 512×512×512 | 0.75, 0.75, 1.0 | [64, 64, 48] | 0.4 | 2 | 30 | `config_infer_24g_512x512x512.json` |
| 32 GB | 512×512×512 | 0.75, 0.75, 1.0 | [80, 80, 48] | 0.4 | 4 | 30 | `config_infer_32g_512x512x512.json` |
| 80 GB | 512×512×512 | 0.75, 0.75, 1.0 | [80, 80, 80] | 0.4 | 4 | 1000 (DDPM) | `config_infer_80g_512x512x512.json` |
| 80 GB | 512×512×768 | 0.75, 0.75, 0.667 | [80, 80, 96] | 0.4 | 4 | 30 | `config_infer_80g_512x512x768.json` |

Tuning rules of thumb if no preset matches:

- **OOM** → reduce `autoencoder_sliding_window_infer_size`, or increase `autoencoder_tp_num_splits` (try the next value in `{2, 4, 8, 16}`).
- **Seam artifacts** → raise `autoencoder_sliding_window_infer_overlap` toward `0.6667`.
- **Speed** → lower the overlap (toward `0.25`), then enlarge the sliding-window size if VRAM permits.

### 4. `cfg_guidance_scale_tumor`

CT-only, controls tumor signal strength. `0` = off (correct default). `1..5` = stronger tumor enforcement, growing artifact risk above 5. Doubles per-step compute when `> 0`. Distinct from the modality-CFG (`cfg_guidance_scale_modality`) used by MR image-only inference — see [`infer_image-only`](infer_image-only.md).

### 5. `num_inference_steps`

`rflow-ct` → **30**. `ddpm-ct` → **1000** (mandatory; the DDPM scheduler runs with fewer but emits a warning and degrades quality). `mask_generation_num_inference_steps` is always **1000** — the mask DM is DDPM regardless of which image-DM variant you pick.

## Quality check loop

`LDMSampler.quality_check_ct` runs after each image is generated (CT only; MR codes ≥ 8 skip the check):

- For each label (liver, spleen, pancreas, kidney, lung, brain, tumors, bone), check that the **median HU value** of voxels with that label falls in the per-organ acceptable range stored in `configs/image_median_statistics_ct.json`.
- If any label is an outlier → fail; retry mask + image generation up to `max_try_time=2` times.
- If still failing after retries: save the last attempt and log a warning.

## Configuration knobs

Live in the three configs:

- `config_network_*.json` — fixed network architecture; not usually edited.
- `config_infer.json` — user intent (see below).
- `environment_*.json` — paths.

Key `config_infer.json` knobs:

| Key | Effect |
|---|---|
| `body_region` | List of regions present in the requested mask: any of `["head", "chest", "thorax", "abdomen", "pelvis", "lower"]`. Used by Path B only (`find_masks` filter). |
| `anatomy_list` | List of organ names from `configs/label_dict.json` that must be present. Used by `find_masks` (Path B) AND as the post-process filter (`filter_mask_with_organs`) for both paths. |
| `controllable_anatomy_size` | Empty list → Path B. Non-empty list of `(organ_name, size)` tuples → Path A (diffusion-generated mask). At most 10 entries; at most 1 tumor. |
| `output_size` | Target volume shape. Hard constraints apply (see `infer_image-only` skill). |
| `spacing` | Target voxel spacing (mm). Hard constraints apply. |
| `modality` | Modality code (1=CT, 8..32=MR variants). |
| `num_inference_steps` | RFlow → 30, **DDPM → 1000**. ⚠️ For `ddpm-ct` you must set this to 1000; the notebook auto-applies this override in cell 12. |
| `mask_generation_num_inference_steps` | **1000** — the mask DM always uses DDPM regardless of which image-DM variant you pick. Setting this lower silently degrades mask quality. |
| `cfg_guidance_scale_tumor` | Strengthens **tumor** signal (this pipeline is CT-only). `0` (default) = off; `1..5` = stronger tumor enforcement, more artifact risk. Distinct from the modality-CFG (`cfg_guidance_scale_modality`) used by MR inference — see [`infer_image-only`](infer_image-only.md). |

## Output

For each successful generation, two files are saved to `output_dir`:

- `sample_<timestamp>_image.nii.gz` — synthetic CT/MR
- `sample_<timestamp>_label.nii.gz` — paired mask (filtered to `anatomy_list`)

## Code references

| Symbol | File |
|---|---|
| `LDMSampler` | `scripts/sample.py` |
| `LDMSampler.sample_multiple_images` (orchestrator) | `scripts/sample.py` |
| `LDMSampler.prepare_anatomy_size_condition` (Path A) | `scripts/sample.py` |
| `LDMSampler.find_closest_masks`, `read_mask_information` (Path B) | `scripts/sample.py` |
| `LDMSampler.sample_one_pair` (image stage) | `scripts/sample.py` |
| `LDMSampler.quality_check_ct` | `scripts/sample.py` |
| `ldm_conditional_sample_one_mask` (mask DM) | `scripts/sample_mask.py` |
| `ldm_conditional_sample_one_image` (image DM + ControlNet) | `scripts/infer_image_from_mask.py` |
| `find_masks` (mask DB lookup) | `scripts/find_masks.py` |
| `augmentation`, `remove_tumors` | `scripts/augmentation.py` |
| `is_outlier` (quality check) | `scripts/quality_check.py` |
| CLI entry point | `scripts/inference.py` |

## Related skills

- `infer_mask-generation` — algorithm details for the mask stage.
- `infer_image-from-mask` — algorithm details for the image stage.
- `infer_image-only` — image-only path (no mask); covers FOV/dim/spacing table.
- `download-models` — fetch checkpoints first.
