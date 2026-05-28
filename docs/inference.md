# Inference Guide

This guide covers how to run inference with each NV-Generate-CTMR variant, how to size the output volume (FOV) for your anatomy, and how to tune the runtime knobs. For a quick orientation see the README §2 quick-start; for end-to-end skill-style walkthroughs see [`skills/`](../skills/).

## Overview

NV-Generate-CTMR supports four inference modes, each backed by a different model variant:

| Mode | Variant | Skill |
|---|---|---|
| Paired CT image + segmentation mask | `rflow-ct`, `ddpm-ct` | [`infer_mask-image-paired`](../skills/infer_mask-image-paired.md) |
| CT image only (no mask) | `rflow-ct`, `ddpm-ct` | [`infer_image-only`](../skills/infer_image-only.md) |
| MR image only (non-brain — prostate / breast / abdomen) | `rflow-mr` | [`infer_image-only`](../skills/infer_image-only.md) |
| MR brain image only (T1 / T2 / FLAIR / SWI, whole-brain or skull-stripped) | `rflow-mr-brain` | [`infer_image-only`](../skills/infer_image-only.md) |

CT supports a ControlNet pipeline (mask-conditioned image synthesis); MR does not — there is no MR ControlNet in this repo.

## Execute inference

### Paired CT image + mask

```bash
export MONAI_DATA_DIRECTORY=<dir_you_will_download_data>
network="rflow"                       # or "ddpm" for ddpm-ct
generate_version="rflow-ct"           # or "ddpm-ct"

python -m scripts.inference \
    -t ./configs/config_network_${network}.json \
    -i ./configs/config_infer.json \
    -e ./configs/environment_${generate_version}.json \
    --random-seed 0 --version ${generate_version}
```

> ⚠️ `ddpm-ct` requires `"num_inference_steps": 1000` in `config_infer.json`. `rflow-ct` uses `30`. Lower DDPM step counts emit a warning and produce low-quality output.

There is currently no ControlNet for MRI — MR variability is too large to train one whole-body model. See [inference_tutorial.ipynb](../inference_tutorial.ipynb) for an end-to-end notebook walkthrough of the CT paired case.

### CT image only (no mask)

```bash
network="rflow"                       # or "ddpm" for ddpm-ct
generate_version="rflow-ct"           # or "ddpm-ct"

python -m scripts.download_model_data --version ${generate_version} --root_dir "./" --model_only
python -m scripts.diff_model_infer \
    -t ./configs/config_network_${network}.json \
    -e ./configs/environment_maisi_diff_model_${generate_version}.json \
    -c ./configs/config_maisi_diff_model_${generate_version}.json
```

### MR image only (non-brain — `rflow-mr`)

```bash
network="rflow"
generate_version="rflow-mr"

python -m scripts.download_model_data --version ${generate_version} --root_dir "./" --model_only
python -m scripts.diff_model_infer \
    -t ./configs/config_network_${network}.json \
    -e ./configs/environment_maisi_diff_model_${generate_version}.json \
    -c ./configs/config_maisi_diff_model_${generate_version}.json
```

Set `"modality"` in `config_maisi_diff_model_rflow-mr.json` per the [Modality codes](#modality-codes) table below. For brain MRI prefer the dedicated `rflow-mr-brain` model.

### MR brain image only (`rflow-mr-brain`)

```bash
network="rflow"
generate_version="rflow-mr-brain"

python -m scripts.download_model_data --version ${generate_version} --root_dir "./" --model_only
python -m scripts.diff_model_infer \
    -t ./configs/config_network_${network}.json \
    -e ./configs/environment_maisi_diff_model_${generate_version}.json \
    -c ./configs/config_maisi_diff_model_${generate_version}.json
```

Whole-brain (modality 9, 10, 11, 20) and skull-stripped (29, 30, 31, 32) outputs are both supported — see [Modality codes](#modality-codes).

### Accelerated inference with TensorRT (CT only)

Pass an extra `-x ./configs/config_trt.json` to the CT paired-inference command:

```bash
python -m scripts.inference \
    -t ./configs/config_network_rflow.json \
    -i ./configs/config_infer.json \
    -e ./configs/environment_rflow-ct.json \
    --random-seed 0 --version rflow-ct \
    -x ./configs/config_trt.json
```

[`config_trt.json`](../configs/config_trt.json) uses MONAI's `trt_compile()` to convert select modules to TensorRT, overriding their definitions from `config_infer.json`.

## Recommended Spacing for CT

These are the median `output_size` / `spacing` pairs in the CT training set, grouped by `body_region`.

| `"body_region"` | percentage of training data | Recommended `"output_size"` | Recommended `"spacing"` [mm] |
|:---|:---|:---|---:|
| ['chest', 'abdomen'] | 58.55% | [512, 512, 128] | [0.781, 0.781, 2.981] |
| ['chest'] | 38.35% | [512, 512, 128] | [0.684, 0.684, 2.422] |
| ['chest', 'abdomen', 'lower'] | 1.42% | [512, 512, 256] | [0.793, 0.793, 1.826] |
| ['lower'] | 0.61% | [512, 512, 384] | [0.839, 0.839, 0.728] |
| ['abdomen', 'lower'] | 0.37% | [512, 512, 384] | [0.808, 0.808, 0.729] |
| ['head', 'chest', 'abdomen'] | 0.33% | [512, 512, 384] | [0.977, 0.977, 2.103] |
| ['abdomen'] | 0.13% | [512, 512, 128] | [0.723, 0.723, 1.182] |
| ['head', 'chest', 'abdomen', 'lower'] | 0.13% | [512, 512, 384] | [1.367, 1.367, 4.603] |
| ['head', 'chest'] | 0.10% | [512, 512, 128] | [0.645, 0.645, 2.219] |

If you need a different `output_size`, adjust `spacing` so `output_size × spacing` keeps a reasonable FOV (e.g.):

| `"output_size"` | Recommended `"spacing"` |
|:---|:---|
| [256, 256, 256] | [1.5, 1.5, 1.5] |
| [512, 512, 128] | [0.8, 0.8, 2.5] |
| [512, 512, 512] | [1.0, 1.0, 1.0] |

## Recommended FOV for MR `rflow-mr` model

Recommended FOV is computed from the median FOV of the training data.

| Body region | Modality | Number of training images | Recommended FOV x × y × z (mm) |
|---|---|---:|---|
| brain | mri_t1 (9) | 4,659 | 160.0 × 256.0 × 256.0 |
| brain | mri_t2 (10) | 577 | 240.0 × 240.0 × 162.5 |
| brain | mri_flair (11) | 152 | 199.9 × 250.0 × 250.0 |
| prostate | mri_t2 (10) | 898 | 170.0 × 170.0 × 90.0 |
| breast | mri_t1 (9) | 2,162 | 174.0 × 200.0 × 200.0 |
| abdomen | mri_t1 (9) | 715 | 380.0 × 308.8 × 288.0 |
| abdomen | mri_t2 (10) | 78 | 350.0 × 350.0 × 245.6 |

Contrast-enhanced MRI is not supported. For brain MRI, prefer the dedicated `rflow-mr-brain` model — see below.

## Recommended FOV for MR `rflow-mr-brain` model

Recommended FOV per (modality, acquisition plane) across the MR-RATE training set. `N` counts unique source images; whole-brain and skull-stripped share the same FOV since they're two preprocessings of the same subject. Total unique images per skull condition (sum of the rows below): **318 825**.

> ℹ️ The table covers axial, sagittal, and coronal scans only. The training set also includes images acquired in oblique orientations (not summarized here); the model has seen those during training but they are excluded from this reference table because the axial/sagittal/coronal cases are what users typically request at inference.
>
> ⚠️ Some (modality, plane) combinations have very few training images — output quality is not guaranteed for: **MRA** all planes (37 / 98 / 11), **SWI sagittal** (2), **SWI coronal** (4).

| Modality | Plane | Recommended FOV (mm) | Number of training images |
|---|---|---|---:|
| T1 | axial | 240 × 240 × 174 | 47 810 |
| T1 | sagittal | 176 × 250 × 250 | 69 268 |
| T1 | coronal | 240 × 200 × 240 | 38 756 |
| T2 | axial | 240 × 240 × 158 | 195 |
| T2 | sagittal | 162 × 240 × 240 | 551 |
| T2 | coronal | 200 × 180 × 200 | 125 |
| FLAIR | axial | 250 × 250 × 175 | 27 990 |
| FLAIR | sagittal | 176 × 250 × 250 | 58 421 |
| FLAIR | coronal | 250 × 200 × 250 | 27 698 |
| SWI | axial | 230 × 230 × 145 | 47 859 |
| SWI | sagittal | 140 × 230 × 230 | 2 |
| SWI | coronal | 230 × 155 × 230 | 4 |
| MRA | axial | 220 × 220 × 158 | 37 |
| MRA | sagittal | 158 × 250 × 250 | 98 |
| MRA | coronal | 240 × 179 × 240 | 11 |

Pick the row matching your target acquisition plane, then pick the modality code (see [Modality codes](#modality-codes); `9..20` for whole-brain or `29..33` for skull-stripped). The Recommended FOV is a starting point — feel free to vary `dim` and `spacing` as long as `dim[i] × spacing[i]` lands near it.

## Modality codes

Control the MR contrast (and skull-stripping state for `rflow-mr-brain`) by setting `"modality"` in the variant's `config_maisi_diff_model_*.json`. Full mapping in [`configs/modality_mapping.json`](../configs/modality_mapping.json).

| Code | Modality | Used by |
|---:|---|---|
| 1 | CT | `rflow-ct`, `ddpm-ct` (always) |
| 8 | mri (unspecified contrast) | `rflow-mr` |
| 9 | mri_t1 | `rflow-mr`, `rflow-mr-brain` (whole-brain) |
| 10 | mri_t2 | `rflow-mr`, `rflow-mr-brain` (whole-brain) |
| 11 | mri_flair | `rflow-mr`, `rflow-mr-brain` (whole-brain) |
| 16 | mri_mra | `rflow-mr-brain` (whole-brain) |
| 20 | mri_swi | `rflow-mr-brain` (whole-brain) |
| 29 | mri_t1_skull_stripped | `rflow-mr-brain` (skull-stripped) |
| 30 | mri_t2_skull_stripped | `rflow-mr-brain` (skull-stripped) |
| 31 | mri_flair_skull_stripped | `rflow-mr-brain` (skull-stripped) |
| 32 | mri_swi_skull_stripped | `rflow-mr-brain` (skull-stripped) |
| 33 | mri_mra_skull_stripped | `rflow-mr-brain` (skull-stripped) |

## Configuration parameter reference

The user-facing config knobs live in [`../configs/config_infer.json`](../configs/config_infer.json) (paired pipeline) and `../configs/config_maisi_diff_model_<variant>.json` (image-only pipelines).

| Key | Purpose |
|---|---|
| `num_output_samples` | Number of output samples to generate. |
| `output_size` (paired) / `dim` (image-only) | Output volume shape in voxels. Must be divisible by 16. Reduce if GPU memory is limited. |
| `spacing` | Voxel size in mm. `output_size × spacing` = FOV. See the recommended-FOV tables above for the training distribution. |
| `controllable_anatomy_size` | (Paired CT only) List of `[organ_name, size]` tuples — e.g. `[["liver", 0.5], ["hepatic tumor", 0.3]]`. Triggers Path A (diffusion-generated mask) when non-empty. |
| `body_region` | (Paired `ddpm-ct` only when `controllable_anatomy_size` is empty) One or more of `head`, `chest`, `thorax`, `abdomen`, `pelvis`, `lower`. Deprecated for `rflow-ct` — leave `[]`. |
| `anatomy_list` | List of organ names from [`configs/label_dict.json`](../configs/label_dict.json) to keep in the output mask. |
| `modality` | Modality code — see [Modality codes](#modality-codes). |
| `num_inference_steps` | RFlow → 30, DDPM → 1000 (mandatory; lower values warn and degrade). For the mask DM, always 1000. |
| `cfg_guidance_scale` (in `config_maisi_diff_model_*.json`) | Image-only path: classifier-free guidance on the **modality**. CT → 0, MR → 10 (shipped defaults — keep). |
| `cfg_guidance_scale` (in `config_infer.json`) | Paired CT path: classifier-free guidance on **tumor** presence. `0` (default) = off. `1..5` = stronger tumor enforcement. Same key name as above; semantics depend on which script reads the config. |
| `autoencoder_sliding_window_infer_size` | AE-decode ROI per tile (paired pipeline only — hardcoded `[80,80,80]` in `diff_model_infer`). Must be divisible by 16. Larger = fewer tiles, more VRAM. |
| `autoencoder_sliding_window_infer_overlap` | `[0, 1)`. Higher = smoother seams, more compute. |
| `autoencoder_tp_num_splits` | `∈ {1, 2, 4, 8, 16}`. Higher = lower per-GPU VRAM, slower. |

For validated `(GPU memory, output_size) → (AE sliding-window knobs)` presets, see the "How to configure a run" section of [`infer_mask-image-paired`](../skills/infer_mask-image-paired.md).

## Quality check (CT only)

The paired CT pipeline runs a quality check after each generated image: for each major organ, it verifies the **median HU intensity** falls within the per-organ range stored in [`configs/image_median_statistics_ct.json`](../configs/image_median_statistics_ct.json). If any organ is an outlier, the mask + image are regenerated (up to 2 retries). MR variants skip this check.

For inference time cost and GPU memory usage, see [Performance](performance.md).

## Tuning checklist

- **GPU OOM** → reduce `autoencoder_sliding_window_infer_size` (must stay divisible by 16) or raise `autoencoder_tp_num_splits` to the next value in `{2, 4, 8, 16}`.
- **Seam / stitching artifacts** → raise `autoencoder_sliding_window_infer_overlap` toward `0.6667`.
- **Speed** → lower the overlap toward `0.25`, then enlarge the sliding-window size if VRAM permits.
- **Washed-out MR contrast** → check `cfg_guidance_scale` (in the MR variant's `config_maisi_diff_model_*.json`) is at the shipped default of 10 (not 0).
- **Unusable output despite valid inputs** → FOV is probably out-of-distribution for the variant. Match a row in the FOV tables above.

## Architecture

The pipeline trains an autoencoder in pixel space to encode images into latent features, then trains a diffusion model in the latent space. During inference, latent features are generated from random noise via denoising steps, then decoded by the autoencoder.

![Training scheme](../figures/maisi_train.png)

![Inference scheme](../figures/maisi_infer.png)

Network definitions: [config_network_rflow.json](../configs/config_network_rflow.json), [config_network_ddpm.json](../configs/config_network_ddpm.json). Key references: [Latent Diffusion (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf), [ControlNet (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.pdf), [Rectified Flow (ICLR 2023)](https://arxiv.org/pdf/2209.03003).
