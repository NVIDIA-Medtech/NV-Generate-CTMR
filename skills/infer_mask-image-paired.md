---
name: infer_mask-image-paired
description: How to run paired mask + image generation with NV-Generate-CTMR. Generates a 3D mask (either from anatomy_size or by retrieving a real training mask) and then a paired CT/MR image conditioned on that mask via ControlNet. Trigger when the user asks "how do I generate a mask and image together", "how does LDMSampler work", "what does scripts.inference do", or wants help running the README §2.3 CT Paired Image/Mask command.
---

# Mask + image paired inference

This skill covers the **paired generation** pipeline: mask first, then image conditioned on that mask. The CLI is `scripts.inference`, which instantiates `LDMSampler` and calls `sample_multiple_images`. This is the path used in README §2.3 (CT Paired Image/Mask Generation).

Two algorithms run sequentially:

1. **Mask stage** — see the `infer_mask-only` skill.
2. **Image stage** — see the `infer_image-from-mask` skill.

This skill explains how they're chained, the LDMSampler state required, and the configuration knobs.

## Command to run

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

An optional fourth config is available:

- `-x` / `--extra-config-file` — overrides any key already set by the above configs. Used for **TensorRT acceleration**: pass `-x ./configs/config_trt.json` to enable TRT compilation of select modules via `trt_compile()` (CT only). See [`docs/inference.md#accelerated-inference-with-tensorrt-ct-only`](../docs/inference.md#accelerated-inference-with-tensorrt-ct-only) for the full TRT command and which modules are compiled.

### End-to-end example: paired chest CT (Path B — training-mask DB lookup)

Concrete worked example for a 24 GB GPU. Path B is the simpler default — you ask for a chest CT and the pipeline finds a matching training mask, augments it, and synthesizes the paired image.

```bash
# 1. Download all required weights + the mask database (one-time, ~10 GB).
#    No --model_only flag — the mask DB and anatomy-size JSON are also needed.
python -m scripts.download_model_data --version rflow-ct --root_dir "./"

# 2. Pick a config_infer_<XXg>_<dim>.json preset matching your GPU + output_size.
#    For 24 GB + 512×512×128 chest CT, use config_infer_24g_512x512x128.json.
#    Edit it to set:
#      "body_region":                   ["chest"],
#      "anatomy_list":                  ["liver", "spleen", "right lung lower lobe"],   # organ names from label_dict.json; "lung" is NOT valid here
#      "controllable_anatomy_size":     [],                            # empty list → Path B
#      "num_output_samples":            1,
#      # leave the AE knobs, output_size, spacing, cfg_guidance_scale,
#      # num_inference_steps at the preset's shipped values.

# 3. Run inference.
export MONAI_DATA_DIRECTORY="./temp_work_dir"
python -m scripts.inference \
    -t ./configs/config_network_rflow.json \
    -i ./configs/config_infer_24g_512x512x128.json \
    -e ./configs/environment_rflow-ct.json \
    --random-seed 0 --version rflow-ct
```

**Expected output**: a pair of NIfTIs under the `output_dir` set in `environment_rflow-ct.json` — `sample_<timestamp>_image.nii.gz` (synthesized CT, HU `[-1000, 1000]`) and `sample_<timestamp>_label.nii.gz`. **Path A**: full 132-label MAISI mask. **Path B**: label filtered to `anatomy_list` (only the requested organs are kept in the saved label).

For **Path A** (control organ/tumor size), set `controllable_anatomy_size` to a single `(organ_name, size)` entry, e.g. `[["pancreas", 0.5]]`, and leave `body_region` empty.

## Mask stage

See [`infer_mask-only`](infer_mask-only.md) for how the mask is generated (Path A vs Path B, conditioning, FOV/spacing rules, config knobs). The generated mask feeds directly into the image stage ([`infer_image-from-mask`](infer_image-from-mask.md)).

## `output_size` and `spacing` — FOV matters

See [`infer_mask-only`](infer_mask-only.md#output_size-and-spacing--fov-matters) for FOV guidance (training FOV, spacing derivation, Path A vs Path B considerations). The same rules apply here.

## How to configure a run

### 1. `modality` → driven by your anatomy

Pick the modality code matching what you want to generate (full list in `configs/modality_mapping.json`). This mask-image paired pipeline is **CT-only** (the mask DM and ControlNet are CT-only — no MR ControlNet exists), so `modality = 1`. For MR generation use [`infer_image-only`](infer_image-only.md). For recommended FOVs per anatomy, see `docs/inference.md#recommended-spacing-for-ct`.

### 2. `autoencoder_sliding_window_infer_size`, `autoencoder_sliding_window_infer_overlap`, `autoencoder_tp_num_splits` → from GPU memory + `output_size`

Validated presets (drawn from `configs/config_infer_<XXg>_<dim>.json`):

| GPU mem | `output_size` | `autoencoder_sliding_window_infer_size` | `autoencoder_sliding_window_infer_overlap` | `autoencoder_tp_num_splits` |
|---|---|---|---|---|
| 16 GB | 256×256×128 | [96, 96, 96] | 0.25 | 2 |
| 16 GB | 256×256×256 | [48, 48, 64] | 0.6666 | 4 |
| 16 GB | 512×512×128 | [64, 64, 32] | 0.5 | 2 |
| 24 GB | 256×256×256 | [64, 64, 64] | 0.25 | 4 |
| 24 GB | 512×512×128 | [80, 80, 32] | 0.4 | 2 |
| 24 GB | 512×512×512 | [64, 64, 48] | 0.4 | 2 |
| 32 GB | 512×512×512 | [80, 80, 48] | 0.4 | 4 |
| 80 GB | 512×512×512 | [80, 80, 80] | 0.4 | 4 |
| 80 GB | 512×512×768 | [80, 80, 96] | 0.4 | 4 |

Tuning rules if no preset matches:

- **OOM** → shrink `autoencoder_sliding_window_infer_size` (must be divisible by 16), or raise `autoencoder_tp_num_splits` to the next value in `{2, 4, 8, 16}`.
- **Seam artifacts** → raise `autoencoder_sliding_window_infer_overlap` toward `0.6667`.
- **Speed** → lower the overlap toward `0.25`, then enlarge the sliding-window size if VRAM permits.

### 3. `spacing` → from FOV and `output_size`

```text
spacing[i] = FOV[i] / output_size[i]
```

Pick FOV from the anatomy-recommended table (step 1), pick `output_size` from the GPU preset (step 2), compute `spacing`.

### 4. Modality-CFG → not used in this pipeline

This pipeline is CT-only and modality is fixed at `CT=1`, so modality-CFG has nothing to amplify. The modality-CFG version of `cfg_guidance_scale` lives in `config_maisi_diff_model_*.json` and is read by `scripts.diff_model_infer` ([`infer_image-only`](infer_image-only.md)), where it is required for MR — see that skill.

### 5. `cfg_guidance_scale` (tumor-CFG in this pipeline)

Classifier-free guidance (CFG) scale on tumor presence. CFG runs the model twice per step (mask as-is vs mask with `remove_tumors()`) and amplifies the difference, strengthening tumor signal in the synthesized image. CT-only. `0` (default) = off. `1..5` = stronger tumor enforcement, growing artifact risk above 5. Doubles per-step compute when `> 0`. Same key name as the modality-CFG (step 4) — semantics depend on which script reads the config: tumor here, modality in `scripts.diff_model_infer`.

### 6. `num_inference_steps`

Driven by the scheduler the variant uses, not by GPU memory:

- `rflow-ct` → **30** (RFlow scheduler).
- `ddpm-ct` → **1000** (DDPM scheduler). Lower values emit a warning and degrade quality — not optional.
- `mask_generation_num_inference_steps` → always **100**: the v2 mask DM uses RFlow.

## Configuration knobs

Live in the three configs:

- `config_network_*.json` — fixed network architecture; not usually edited.
- `config_infer.json` — user intent (see below).
- `environment_*.json` — paths.

Key `config_infer.json` knobs:

Mask-stage knobs (`body_region`, `anatomy_list`, `controllable_anatomy_size`, `controllable_demographics`, `mask_generation_num_inference_steps`, `mask_generation_cfg_guidance_scale`) — see [`infer_mask-only`](infer_mask-only.md).

| Key | Effect |
|---|---|
| `output_size` | Target volume shape. Hard constraints apply (see `infer_image-only` skill). |
| `spacing` | Target voxel spacing (mm). Hard constraints apply. |
| `modality` | Modality code (1=CT, 8..32=MR variants). |
| `num_inference_steps` | RFlow → 30, **DDPM → 1000**. ⚠️ For `ddpm-ct` you must set this to 1000; the notebook auto-applies this override in cell 12. |
| `cfg_guidance_scale` | Strengthens **tumor** signal (this pipeline is CT-only). `0` (default) = off; `1..5` = stronger tumor enforcement, more artifact risk. The same key name in `config_maisi_diff_model_*.json` is the modality-CFG used by MR image-only inference — see [`infer_image-only`](infer_image-only.md). |

## Output

For each successful generation, two files are saved to `output_dir`:

- `sample_<timestamp>_image.nii.gz` — synthetic CT/MR
- `sample_<timestamp>_label.nii.gz` — full 132-label MAISI mask (Path A) or filtered to `anatomy_list` (Path B)

## Related scripts

| Script | Role |
|---|---|
| `scripts/inference.py` | CLI entry point for this skill. |
| `scripts/sample.py` (`LDMSampler`) | Orchestrator: dispatches the mask stage and the image stage, applies the QC retry loop. |
| `scripts/sample_mask.py` | Path A mask DM (`ldm_conditional_sample_one_mask`). |
| `scripts/find_masks.py` | Path B exact-match DB lookup (`find_masks`). |
| `scripts/infer_image_from_mask.py` | Image-from-mask pipeline (called from the orchestrator's image stage). |
| `scripts/download_model_data.py` | Downloads mask DM + image DM + ControlNet weights. |

## Related skills

- [`infer_mask-only`](infer_mask-only.md) — mask-stage details.
- [`infer_image-from-mask`](infer_image-from-mask.md) — image-stage details.
- [`infer_image-only`](infer_image-only.md) — image-only path (no mask, including MR); covers the FOV / `dim` / `spacing` recommendations.
- [`download-models`](download-models.md) — fetch checkpoints first.
