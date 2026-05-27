---
name: infer_image-only
description: How to run image-only inference (no mask, no ControlNet) with NV-Generate-CTMR. Covers picking the right model variant (rflow-ct / rflow-mr / rflow-mr-brain / ddpm-ct), choosing dim/spacing for a target field-of-view, and the modality + body-region knobs. Trigger when the user asks "how do I generate a CT image", "what dim/spacing should I use", "how do I set the FOV", "which model variant for brain MRI / abdomen CT / chest CT", or wants help running the README §2.2, §2.4, §2.5 commands.
---

# Image-only inference (no mask)

> ## ⚠️ Why FOV matters (read this first)
>
> **FOV = `dim × spacing`** (mm per axis). This is the single biggest knob for output quality. The model has only seen FOVs from the **training-data distribution** for its target anatomy — asking it to synthesize at a numerically-valid but out-of-distribution FOV produces unrealistic output, even when `check_input_ct`/`check_input_mr` accept the inputs.
>
> **The "Recommended (dim, spacing) by anatomical target" table below is not a list of preferences** — those values are where the training data actually lives. Stay close to them; the further you deviate the worse the output.
>
> **Common failure mode**: user picks `dim=(256,256,256), spacing=(0.5,0.5,0.5)` to "make a high-res small volume." Validator accepts it (FOV=128mm cube). The DM produces noise because it never saw 128 mm body FOVs at training. **Fix**: match a row in the recommended table below.

This skill covers running the **image-only** diffusion model — no ControlNet, no mask input. The CLI is `scripts.diff_model_infer`. Three Quick Start subsections of the README use this path:

- §2.2 MR Brain Image Generation (`rflow-mr-brain`)
- §2.4 CT Image Generation (`rflow-ct` or `ddpm-ct`)
- §2.5 MR Image Generation (`rflow-mr`)

This is distinct from the mask-image-paired pipeline in §2.3, which uses `scripts.inference` and the `LDMSampler` orchestrator (see the `infer_mask-image-paired` skill).

## Picking a model variant

| Variant | Modality | Architecture | Inference steps | Body region input? | Max volume | Use case |
|---|---|---|---|---|---|---|
| `rflow-mr-brain` | MRI (brain) | MAISI-v2 (Rectified Flow) | 30 | No | 512×512×256 | T1/T2/FLAIR/SWI whole-brain and skull-stripped |
| `rflow-mr` | MRI (other) | MAISI-v2 (Rectified Flow) | 30 | No | 512×512×128 | T2 prostate, T1 breast, T1/T2 abdomen, etc. Recommend fine-tuning. |
| `rflow-ct` | CT | MAISI-v2 (Rectified Flow) | **30** (33× faster) | No | **512×512×768** | Whole-body CT |
| `ddpm-ct` | CT | MAISI-v1 (DDPM) | 1000 | **Yes** | 512×512×768 | Whole-body CT with explicit body-region indices |

Pick the variant by:

1. Modality + anatomy (brain MRI → `rflow-mr-brain`; CT → `rflow-ct`; other MRI → `rflow-mr`).
2. Whether you need explicit body-region conditioning (use `ddpm-ct` if you want `top_region_index` / `bottom_region_index` as inputs; else prefer `rflow-ct` — 33× faster, similar FID).

## Quick Start commands

Each variant follows the same two-step pattern: download weights, then run inference.

```bash
network="rflow"   # or "ddpm" for ddpm-ct
generate_version="rflow-mr-brain"   # or rflow-ct / rflow-mr / ddpm-ct

python -m scripts.download_model_data --version ${generate_version} --root_dir "./" --model_only

python -m scripts.diff_model_infer \
    -t ./configs/config_network_${network}.json \
    -e ./configs/environment_maisi_diff_model_${generate_version}.json \
    -c ./configs/config_maisi_diff_model_${generate_version}.json
```

For `ddpm-ct`: use `network="ddpm"` and the corresponding `config_network_ddpm.json` / `environment_maisi_diff_model_ddpm-ct.json` / `config_maisi_diff_model_ddpm-ct.json`.

> ⚠️ **`ddpm-ct` requires `num_inference_steps = 1000`** (vs 30 for `rflow-ct` / `rflow-mr*`). Lower values silently degrade output — the DDPM scheduler emits a warning but still runs. This makes `ddpm-ct` ~33× slower than `rflow-ct`. Prefer `rflow-ct` unless you specifically need body-region indices.

## How to configure a run

All knobs live in `configs/config_maisi_diff_model_<variant>.json` under the `diffusion_unet_inference` block. The numbered steps below mirror the parallel **"How to configure a run"** in [`infer_mask-image-paired.md`](infer_mask-image-paired.md) so the two skills are easy to compare. Steps that don't apply here (AE sliding-window knobs, `cfg_guidance_scale_tumor`) are flagged N/A.

### 1. `modality` → driven by your anatomy

Set `"modality"` to the modality code matching what you want to generate. Codes from `configs/modality_mapping.json`:

| Code | Modality | Notes |
|---|---|---|
| 1 | CT | always set for `rflow-ct` / `ddpm-ct` |
| 8 | MRI (no contrast specified) | |
| 9 | mri_t1 | T1w whole-brain |
| 10 | mri_t2 | T2w whole-brain |
| 11 | mri_flair | FLAIR whole-brain |
| 20 | mri_swi | SWI whole-brain |
| 29 | mri_t1_skull_stripped | T1w skull-stripped |
| 30 | mri_t2_skull_stripped | T2w skull-stripped |
| 31 | mri_flair_skull_stripped | FLAIR skull-stripped |
| 32 | mri_swi_skull_stripped | SWI skull-stripped |

**For `ddpm-ct` only**, also set one-hot body-region indices:

```json
"top_region_index":    [0, 1, 0, 0],   // chest
"bottom_region_index": [0, 0, 1, 0]    // abdomen
```

Slots are `[head, chest, abdomen, pelvis]`. `rflow-ct` / `rflow-mr` / `rflow-mr-brain` do **not** use these — `include_top_region_index_input` is False.

### 2. AE sliding-window knobs → N/A in this path

`autoencoder_sliding_window_infer_size` / `_overlap` / `autoencoder_tp_num_splits` are **not exposed** by `scripts.diff_model_infer` — they're hardcoded (`roi_size=[80, 80, 80]`, `overlap=0.4`, no TP split). If you hit OOM on the AE decode, your only knob is reducing `dim`. See the GPU-memory presets table in [`infer_mask-image-paired.md`](infer_mask-image-paired.md#2-autoencoder_sliding_window_infer_size-autoencoder_sliding_window_infer_overlap-autoencoder_tp_num_splits--from-gpu-memory--output_size) if you need those.

### 3. `dim` and `spacing` → from FOV

The most important knobs. Live in the `diffusion_unet_inference` block of `configs/config_maisi_diff_model_<variant>.json`:

```json
"diffusion_unet_inference": {
    "dim": [256, 256, 256],
    "spacing": [1, 1, 1],
    ...
}
```

The relationship is **`FOV[i] = dim[i] × spacing[i]`**, i.e. **`spacing[i] = FOV[i] / dim[i]`**. Pick FOV first (anatomy-driven), then `dim` (resolution / VRAM budget); `spacing` falls out.

**Recommended `(dim, spacing)` by anatomical target** (these values are where the training data actually lives — stay close):

| Target | `dim` | `spacing` (mm) | Resulting FOV (mm) | Variant |
|---|---|---|---|---|
| Brain (whole-brain, T1/T2/FLAIR/SWI) | `(256, 256, 256)` | `(1.0, 1.0, 1.0)` | `256 × 256 × 256` | `rflow-mr-brain` |
| Brain skull-stripped | `(256, 256, 256)` | `(1.0, 1.0, 1.0)` | `256 × 256 × 256` | `rflow-mr-brain` |
| Chest (single-slice axial coverage) | `(512, 512, 128)` | `(0.78, 0.78, 4.0)` | `400 × 400 × 512` | `rflow-ct` |
| Abdomen | `(512, 512, 256)` | `(1.0, 1.0, 1.5)` | `512 × 512 × 384` | `rflow-ct` |
| Whole body (torso → mid-femur) | `(512, 512, 512)` | `(1.5, 1.5, 1.5)` | `768 × 768 × 768` | `rflow-ct` |
| Long-axis whole-body (head → feet) | `(512, 512, 768)` | `(1.5, 1.5, 1.5)` | `768 × 768 × 1152` | `rflow-ct` (max supported) |

**Hard constraints** (validated by `check_input_ct` / `check_input_mr`):

For CT (`rflow-ct`, `ddpm-ct`):

- `dim[0] == dim[1]`, `dim[0] ∈ {256, 384, 512}`, `dim[2] ∈ {128, 256, 384, 512, 640, 768}`
- `spacing[0] == spacing[1]`, `spacing[0] ∈ [0.5, 3.0]` mm, `spacing[2] ∈ [0.5, 5.0]` mm
- Recommended `FOV_xy ≥ 256` mm for head, `≥ 384` mm for abdomen/body

For MR (`rflow-mr`, `rflow-mr-brain`):

- At least two of `dim[0..2]` must be equal
- If `dim[2]=128`: `dim[0]=dim[1] ∈ {128, 256, 384, 512}`
- If `dim[2]=256`: `dim ∈ {[128,256,256], [256,128,256], [256,256,256]}`
- `spacing ∈ [0.4, 5.0]` mm per axis

Sanity-check the resulting FOV with `print([dim[i]*spacing[i] for i in range(3)])` before running. See `docs/inference.md` for the full per-modality FOV table.

### 4. `cfg_guidance_scale_modality`

This knob steers the output toward the requested **modality** (the `class_labels` / modality tensor). Effect of the value:

- **`0`** — modality conditioning is effectively ignored. For CT this is fine (modality is fixed at `CT=1`, so guidance has nothing to amplify). For **MR variants this is the failure mode** ([issue #29](https://github.com/NVIDIA-Medtech/NV-Generate-CTMR/issues/29)): the output is washed-out and doesn't commit to the requested contrast (T1 / T2 / FLAIR / SWI).
- **`~10`** — validated value for MR. Output looks like the requested contrast.
- **Much above 10** — over-saturates contrast, introduces artifacts.
- **Any value `> 0` roughly doubles UNet compute and VRAM** (conditional + unconditional run in one doubled batch). MR inference therefore costs ~2× CT at the same `dim`.

Recommended values per variant (these are the shipped defaults — keep them):

| Variant | `cfg_guidance_scale_modality` |
|---|---|
| `rflow-ct` | **0** |
| `ddpm-ct` | **0** |
| `rflow-mr-brain` | **10** |
| `rflow-mr` | **10** |

### 5. `cfg_guidance_scale_tumor` → N/A in this path

Tumor-CFG is for the CT mask-conditioned pipelines only. See [`infer_image-from-mask`](infer_image-from-mask.md) and [`infer_mask-image-paired`](infer_mask-image-paired.md). Same key-name pattern, different unconditional branch — don't transfer the "`0` is fine" rule from those skills to MR inference here.

### 6. `num_inference_steps`

Driven by the scheduler the variant uses, not by GPU memory:

- `rflow-ct` / `rflow-mr` / `rflow-mr-brain` → **30** (RFlow scheduler).
- `ddpm-ct` → **1000** (DDPM scheduler). Lower values emit a warning and degrade output quality — not optional.

## Output

One file per sample is saved into `output_dir` (set in the environment config), named like `unet_3d_seed<seed>_size<H>x<W>x<D>_spacing<sx>x<sy>x<sz>_<timestamp>_rank<r>_modality<m>.nii.gz`. Intensity ranges:

| Modality | dtype | Voxel value range |
|---|---|---|
| CT (modality `1..7`) | int16 NIfTI | HU, clipped to `[-1000, 1000]` |
| MR (codes `8..32`) | int16 NIfTI | `[0, +∞)` |

## Related scripts

| Script | Role |
|---|---|
| `scripts/diff_model_infer.py` | CLI for this skill. Runs the image DM in isolation (no mask, no ControlNet). |
| `scripts/download_model_data.py` | Downloads image-DM + AE weights for the chosen variant. |
| `scripts/diff_model_setting.py` | Helper: distributed-init / config-loading / logger setup. |

## Related skills

- [`download-models`](download-models.md) — fetch the right checkpoints.
- [`infer_mask-only`](infer_mask-only.md) — mask-generation stage (the other half of the paired pipeline).
- [`infer_image-from-mask`](infer_image-from-mask.md) — generate an image from an existing mask (CT-only, uses ControlNet).
- [`infer_mask-image-paired`](infer_mask-image-paired.md) — full mask + image paired pipeline (chains both).
