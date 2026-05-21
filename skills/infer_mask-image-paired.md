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
| Other | `device`, `output_dir`, `num_inference_steps`, `cfg_guidance_scale`, etc. | runtime / config |

## `dim` and `spacing` — same FOV rules as image-only

> ⚠️ **FOV (= `dim × spacing`) is the #1 quality knob.** See the **"Why FOV matters"** section at the top of [`infer_image-only.md`](infer_image-only.md) — same warning applies here. Out-of-distribution FOVs produce unusable output even when the validator accepts the inputs.

The mask + image pipeline uses **the same** `output_size` and `spacing` constraints as image-only inference — see the `infer_image-only` skill for the table of recommended `(dim, spacing)` per anatomical target and the hard constraints from `check_input_ct` / `check_input_mr`.

Additional FOV considerations specific to the paired pipeline:

- The **mask DM** was pretrained at **256³ × 1.5 mm iso** (= 384 mm cube FOV). Generating a mask at significantly different shape forces the `ensure_output_size_and_spacing` resampling, which degrades label boundaries. Stay at or near 256³ × 1.5mm for Path A.
- For Path B (mask DB lookup), the candidate masks are themselves drawn from a training-FOV distribution — `find_closest_masks` picks the closest matches, but the closer your requested FOV is to a mode of that distribution, the less reshaping is needed.

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
| `cfg_guidance_scale` | Image-DM tumor CFG; `0` disables. |

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
- `infer_download-models` — fetch checkpoints first.
