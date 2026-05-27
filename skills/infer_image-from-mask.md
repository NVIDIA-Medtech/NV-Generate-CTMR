---
name: infer_image-from-mask
description: Explains how NV-Generate-CTMR generates a CT or MR image from an existing 3D label mask using the ControlNet-conditioned image latent diffusion model. Trigger when the user asks "how do I generate an image from a mask", "how does the ControlNet work", "what does ldm_conditional_sample_one_image do", "explain the image diffusion model in NV-Generate-CTMR", or any low-level question about the image LDM pipeline.
---

# Image-from-mask inference (NV-Generate-CTMR)

This skill explains how NV-Generate-CTMR takes a **3D label mask** (typically produced by the `infer_mask-generation` skill or pulled from the training-mask database) and **synthesizes a paired CT or MR image** from it. The image LDM is conditioned on the mask via a ControlNet branch.

Code entry point: `scripts.infer_image_from_mask.ldm_conditional_sample_one_image`.

## TL;DR

```text
                       [mask label NIfTI]
                                │ binarize_labels (8-bit encoding)
                                ▼
                    [ControlNet conditioning (8-ch)]
[random noise]──┐               │
        ×       │               ▼
   noise_factor │       ┌─────────────┐
                ▼       ▼             │
              [Image Diffusion UNet]  │  controlnet residuals
                       │  RFlow / DDPM loop  injected per timestep
                       ▼
              [image latent (4-ch)]
                       │ sliding-window image-AE decode
                       ▼
              [synthetic image, range [0,1]]
                       │ HU range mapping (CT) or > 0 clip (MR)
                       │ crop_img_body_mask (background → a_min)
                       ▼
              [final CT/MR volume]
```

## Valid input mask format

> ⚠️ **Scope of this section: the released CT ControlNet checkpoints only.**
>
> Everything below applies specifically to the pretrained
> `controlnet_3d_rflow-ct.pt` and `controlnet_3d_ddpm-ct.pt` checkpoints
> (downloaded by `scripts.download_model_data --version {rflow-ct,ddpm-ct}`).
> Those checkpoints were trained on CT masks in the **MAISI 132-class label
> vocabulary**, so a user-provided mask must use the same convention or the
> generated CT will be unusable.
>
> If you train your own ControlNet on a different label vocabulary (e.g.
> binary body silhouette, a different segmentation tool's output, or a
> non-CT modality), the input format is whatever your training pipeline
> used — see `scripts/train_controlnet.py` line 190
> (`controlnet_cond = binarize_labels(labels.as_tensor().to(torch.long))`)
> — labels are bit-encoded as-is, so inference must match training.
>
> Released NV-Generate-CTMR also ships `rflow-mr-brain` and `rflow-mr`
> variants, but those models are **image-only** (no ControlNet, no mask
> input) — they go through `scripts.diff_model_infer` and the
> `infer_image-only` skill, not this skill.

### What "valid" means (for the released CT ControlNet checkpoints)

The mask is a **1-channel NIfTI** (`.nii` or `.nii.gz`), integer dtype, with voxel values drawn from the MAISI 132-class vocabulary (see [`configs/label_dict.json`](../configs/label_dict.json)):

| Value | Meaning |
|---|---|
| `0` | background |
| `1..132` (with some unused values — see `label_dict.json`) | organ / structure labels (e.g. `1`=liver, `3`=spleen, `4`=pancreas, `5`=right kidney, `14`=left kidney, `28..32`=lung lobes, `33..57`=vertebrae) |
| `200` | **body envelope** — every body voxel not labeled with a specific organ |

`200` is the critical "outer-body" label. **The CT ControlNet expects it; nv-segment does not produce it. You must add it yourself** during preprocessing.

### Producing a valid mask from a CT image

The released CT ControlNet expects the MAISI 132-class vocabulary. In concrete terms, **the MAISI 132-class vocabulary is the `nv-segment-ct` output label definition plus the body envelope (label 200)** — so producing a valid mask is really "produce the same labels nv-segment-ct would emit, then add `body=200`". Two practical paths:

#### Option A (recommended): `nv-segment-ct` + add body envelope

1. **Start with a CT image.**
2. **Run [`nv-segment`](https://github.com/NVIDIA-Medtech/NV-Segment-CTMR)** (the NV-Segment-CTMR bundle, modality `CT_BODY`). It outputs a 1-channel NIfTI **already in the MAISI vocabulary** — no label remapping needed — but excludes ~15 label values (dummies + tumors + airway — see `ct_body` in [`NV-Segment-CTMR/configs/inference.json`](https://github.com/NVIDIA-Medtech/NV-Segment-CTMR/blob/main/NV-Segment-CTMR/configs/inference.json)). The output contains ~117 organ labels and no `body=200`.
3. **Add the body envelope** (label `200`): call [`scripts.utils.add_body_envelope(seg_mask, ct_image)`](../scripts/utils.py). It finds the largest connected component of air (default `HU < -800`), morphologically closes it, treats labeled voxels as not-air, inverts to get the body silhouette, then runs an erode→largest-CC→dilate cleanup that drops the patient bed/table (which often touches the body in CT scans). Every voxel inside the silhouette that nv-segment didn't already label is set to `body_label` (default `200`). Algorithm follows `find_body_maskv2` from pengfeig's `3d_ldm_monai`.
4. **Save** as a 1-channel integer NIfTI.

#### Option B: another segmenter + remap + add body envelope

If your mask comes from TotalSegmentator or any other segmenter whose label IDs differ from MAISI, you must first remap the integer label IDs to the MAISI 132-class space defined in [`configs/label_dict.json`](../configs/label_dict.json).

1. **Start with a CT image.**
2. **Run your segmenter** (e.g. TotalSegmentator) on the CT.
3. **⚠️ Remap label IDs to the MAISI 132-class space.** This is the critical step that distinguishes Option B from Option A. Build a label-ID map from your segmenter's IDs → MAISI 132-class IDs by matching anatomical structure name to the entries in [`configs/label_dict.json`](../configs/label_dict.json). Structures not present in MAISI's 132 classes must be dropped (set to `0`). Apply the map voxel-wise. **If you skip this step or get the mapping wrong, the generated CT will be unusable** — the ControlNet was trained on a specific label vocabulary and silently produces garbage for unfamiliar label IDs.
4. **Add the body envelope** (label `200`) as in Option A step 3.
5. **Save** as a 1-channel integer NIfTI.

Either way, the final output is the `--mask` argument the CLI accepts, or the `combine_label_or` argument when calling the library function directly.

### Common pitfall: the AE-channel space (0..124) is NOT the right space

The mask AE in the repo decodes to a **125-channel softmax** that gets `argmax`'d to integer labels in `[0, 124]` (the "AE-channel space"). Those `0..124` values are then **remapped to the MAISI 132-class vocabulary** via [`configs/label_dict_124_to_132.json`](../configs/label_dict_124_to_132.json) (`remap_labels`) before the CT ControlNet ever sees them. So:

- ✅ **Correct input to the released CT ControlNet** (and to this CLI): MAISI 132-class labels, including `body=200`.
- ❌ **Incorrect**: feeding `0..124` AE-channel-space labels directly. The remap is internal to the **mask DM** decoding pipeline only; it does NOT belong in user-mask preprocessing for the released CT ControlNet.

If you have a mask in `0..124` space (e.g. from intermediate steps of a custom mask-DM pipeline), apply `remap_labels(mask, configs/label_dict_124_to_132.json)` first to convert it to the 132-class space before passing it here.

### Validation in the CLI

`scripts/infer_image_from_mask.py::validate_user_mask` assumes the released CT ControlNet convention and will:

- Confirm the mask is 1-channel integer NIfTI
- Warn (not error) if any voxel value is outside the MAISI 132-class vocabulary (`{0..132} ∪ {200}`)
- Auto-resample shape/spacing to a valid `(dim, spacing)` target (with a warning) if needed

If many voxel values fall outside the vocabulary you almost certainly forgot a remap step — or you're trying to use a custom-trained ControlNet whose vocab differs from the released one.

## Inputs to `ldm_conditional_sample_one_image`

| Argument | Type | Description |
|---|---|---|
| `autoencoder` | `AutoencoderKlMaisi` | The image AE (1-channel input/output, 4-ch latent). |
| `diffusion_unet` | `DiffusionModelUNetMaisi` | The image DM. |
| `controlnet` | `ControlNetMaisi` | ControlNet that conditions on the mask. |
| `noise_scheduler` | `RFlowScheduler` or `DDPMScheduler` | Scheduler matching the model variant (`rflow-ct`/`rflow-mr-brain` use RFlow; `ddpm-ct` uses DDPM). |
| `scale_factor` | float | Image-AE latent normalization factor. |
| `combine_label_or` | `Tensor` (1,1,H,W,D) | The input mask in MAISI label vocabulary. |
| `spacing_tensor` | `Tensor` | Per-axis voxel spacing × 100 (encoder-side scaling). |
| `latent_shape` | tuple | Image latent shape, e.g. `(4, 64, 64, 64)` for 256³ output. |
| `output_size` | tuple | Target volume shape (e.g. `(512, 512, 512)`); mask is interpolated to this shape if needed. |
| `noise_factor` | float | Multiplier on the initial noise (default 1.0 in `LDMSampler`). |
| `top_region_index_tensor`, `bottom_region_index_tensor` | `Tensor` | One-hot body-region indices (only used by `ddpm-ct`; `include_body_region=True`). |
| `modality_tensor` | `Tensor` long | Integer modality code (see `configs/modality_mapping.json`): CT=1, MRI variants 8..32. |
| `num_inference_steps` | int | RFlow → 30; **DDPM → 1000 (must, not optional)**. DDPM at < 1000 steps emits a warning and produces low-quality output. |
| `autoencoder_sliding_window_infer_size` | list[int] | ROI for AE decode; default `[96, 96, 96]`. |
| `autoencoder_sliding_window_infer_overlap` | float | Default `0.6667`. |
| `cfg_guidance_scale_tumor` | float | Strengthens **tumor** signal (CT-only). `0` (default) = off; `1..5` = stronger tumor enforcement, more artifact risk. See §6 — and note this is a different CFG from the modality-CFG (`cfg_guidance_scale_modality`) used by MR inference in [`infer_image-only`](infer_image-only.md). |

## Algorithm step by step

### 1. Decide CT vs MR intensity range

```python
if modality_tensor <= 7:        # CT codes
    a_min, a_max = -1000, 1000
else:                           # MRI codes
    a_min, a_max = 0, 1000
b_min, b_max = 0.0, 1.0         # AE output range
```

`a_*` are the target HU/intensity range; `b_*` are the AE's normalized output range. Step 7 below maps between them.

### 2. Interpolate the mask to `output_size`

```python
if combine_label.shape[2:] != output_size:
    combine_label = F.interpolate(combine_label, size=output_size, mode="nearest")
```

Major reshaping degrades quality — `LDMSampler.ensure_output_size_and_spacing` aims to feed a mask that's already at `output_size`.

### 3. Build the ControlNet conditioning tensor

```python
controlnet_cond_vis = binarize_labels(combine_label.as_tensor().long()).half()
# shape (1, 8, H, W, D)
```

`binarize_labels` is an 8-bit-encoding of the integer label per voxel (bit `b` of label → channel `b` of the tensor).

### 4. Initialize noise

```python
latents = initialize_noise_latents(latent_shape, device) * noise_factor
```

### 5. Per-timestep ControlNet + DM forward (denoising loop)

```python
for t, next_t in zip(timesteps, next_timesteps):
    # 5a. ControlNet forward — produces residuals
    down_block_res, mid_block_res = controlnet(
        x=latents, timesteps=t, controlnet_cond=controlnet_cond_vis,
        class_labels=modality_tensor,                                # if include_modality
    )

    # 5b. Image DM forward — consumes the residuals
    eps_hat = diffusion_unet(
        x=latents, timesteps=t, spacing_tensor=spacing_tensor,
        down_block_additional_residuals=down_block_res,
        mid_block_additional_residual=mid_block_res,
        top_region_index_tensor=top_region_index_tensor,             # if include_body_region (ddpm-ct only)
        bottom_region_index_tensor=bottom_region_index_tensor,       # if include_body_region
        class_labels=modality_tensor,                                # if include_modality
    )

    # 5c. Scheduler step
    if isinstance(noise_scheduler, RFlowScheduler):
        latents, _ = noise_scheduler.step(eps_hat, t, latents, next_t)
    else:
        latents, _ = noise_scheduler.step(eps_hat, t, latents)
```

Two model-variant differences:

- **rflow-ct / rflow-mr / rflow-mr-brain** use `RFlowScheduler` (30 steps, much faster). The `set_timesteps` call also passes `input_img_size_numel` so step sizes adapt to volume.
- **ddpm-ct** uses `DDPMScheduler` (1000 steps). It also sets `include_body_region=True` so the UNet receives `top_region_index_tensor` / `bottom_region_index_tensor`.

### 6. Classifier-free guidance (CFG) — optional, CT-only

In this script, `cfg_guidance_scale_tumor` strengthens the **tumor signal** by contrasting a tumor-bearing mask against a tumor-free version (`remove_tumors()`) of the same mask:

```python
combine_label_no_tumor = F.interpolate(remove_tumors(combine_label.squeeze(0)).unsqueeze(0).float(),
                                       size=output_size, mode="nearest")
controlnet_cond_vis_no_tumor = binarize_labels(combine_label_no_tumor.as_tensor().long()).half()

eps_t, eps_uncond = diffusion_unet(...).chunk(2)
eps = eps_uncond + cfg_guidance_scale_tumor * (eps_t - eps_uncond)
```

Effect of the value:

- **`0`** (default) — no tumor enforcement; the model synthesizes tumors as it learned them. Correct default for this pipeline; use it whenever your mask has no tumors or you want unsteered output.
- **`1..5`** — typical tumor-enforcement values. Higher = stronger tumor commitment but more risk of textural artifacts.
- **Much above 5** — over-saturated, visibly artifactual tumors.
- **Any value `> 0`** doubles UNet compute and VRAM per step.

This entire pipeline is **CT-only** (the mask DM and ControlNet are CT-only — no MR ControlNet exists in this repo), so this CFG never applies to MR.

> ⚠️ The image-only path ([`infer_image-only`](infer_image-only.md), `scripts.diff_model_infer`) has its own CFG key called `cfg_guidance_scale_modality` — same idea, different semantics: there it steers the *modality* conditioning and `~10` is **required** for MR. The two keys are now distinct (`cfg_guidance_scale_tumor` here vs. `cfg_guidance_scale_modality` there). Don't transfer the "`0` is the default" rule from this skill to MR inference.

### 7. Sliding-window image-AE decode + HU mapping

```python
inferer = SlidingWindowInferer(roi_size=[96,96,96], overlap=0.6667, mode="gaussian", ...)
synthetic_images = dynamic_infer(inferer, recon_model, latents)

# AE output in [b_min, b_max] = [0, 1]
synthetic_images = (synthetic_images - b_min) / (b_max - b_min)
# Map to target HU range
synthetic_images = synthetic_images * (a_max - a_min) + a_min
# Background → a_min using the mask
synthetic_images = crop_img_body_mask(synthetic_images, combine_label, a_min=a_min)
```

`crop_img_body_mask` sets all voxels where the mask is 0 (background) to `a_min` — keeps the body silhouette clean.

## Output

A 2-tuple `(synthetic_images, combine_label)`:

- `synthetic_images`: `(1, 1, H, W, D)` float tensor in HU range (CT) or `[0, +∞)` (MR).
- `combine_label`: the mask at `output_size`, returned for downstream filtering (`filter_mask_with_organs`).

## Configuration knobs

| Knob | Where | Effect |
|---|---|---|
| `num_inference_steps` | `LDMSampler.__init__` | Quality / speed trade-off. RFlow → 30 is the validated setting; DDPM → 1000. |
| `cfg_guidance_scale_tumor` | `LDMSampler.__init__` | `0` = off, typical values `1..5`. Higher = stronger tumor enforcement, but more artifacts. |
| `autoencoder_sliding_window_infer_size` | `LDMSampler.__init__` | Must be divisible by 16. Larger = fewer tiles but more VRAM. |
| `autoencoder_sliding_window_infer_overlap` | `LDMSampler.__init__` | `[0, 1)`. Higher = smoother blending, more compute. |
| `autoencoder_tp_num_splits` | `LDMSampler.__init__` (via `config_infer.json`) | Tensor-parallel splits inside the AE forward, `∈ {1, 2, 4, 8, 16}`. Higher = lower per-GPU VRAM, slower. |
| `noise_factor` | hardcoded `1.0` in `LDMSampler.__init__` | Scales the initial noise. |

For **validated presets** mapping (GPU memory, `output_size`) → (`autoencoder_sliding_window_infer_size`, overlap, `autoencoder_tp_num_splits`), and the per-knob walkthrough (modality, AE knobs by GPU, `spacing = FOV / output_size`, `cfg_guidance_scale_tumor`, `num_inference_steps`), see the **"How to configure a run"** section in [`infer_mask-image-paired.md`](infer_mask-image-paired.md). The same presets apply here — this script consumes the same `config_infer.json`.

## Output-size + spacing constraints

Validated by `check_input_ct` and `check_input_mr` (in `scripts/sample_mask.py`):

- `output_size[0] == output_size[1]`
- `output_size[0] ∈ {256, 384, 512}`
- `output_size[2] ∈ {128, 256, 384, 512, 640, 768}`
- `spacing[0] == spacing[1]`
- `spacing[0] ∈ [0.5, 3.0]` mm, `spacing[2] ∈ [0.5, 5.0]` mm
- FOV_xy ≥ 256 mm for head, ≥ 384 mm for abdomen / body

See the `infer_image-only` skill for recommended `(dim, spacing)` per anatomical target.

## Code references

| Symbol | File | Notes |
|---|---|---|
| `ldm_conditional_sample_one_image` | `scripts/infer_image_from_mask.py` | core sampling function |
| `crop_img_body_mask` | `scripts/infer_image_from_mask.py` | background HU regularization |
| `LDMSampler.sample_one_pair` | `scripts/sample.py` | wraps `ldm_conditional_sample_one_image` with LDMSampler state |
| `remove_tumors`, `augmentation` | `scripts/augmentation.py` | CFG unconditional mask + training-time mask aug |
| `binarize_labels`, `dynamic_infer` | `scripts/utils.py` | encoding + sliding-window glue |
