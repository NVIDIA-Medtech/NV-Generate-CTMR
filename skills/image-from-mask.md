---
name: image-from-mask
description: Explains how NV-Generate-CTMR generates a CT or MR image from an existing 3D label mask using the ControlNet-conditioned image latent diffusion model. Trigger when the user asks "how do I generate an image from a mask", "how does the ControlNet work", "what does ldm_conditional_sample_one_image do", "explain the image diffusion model in NV-Generate-CTMR", or any low-level question about the image LDM pipeline.
---

# Image-from-mask inference (NV-Generate-CTMR)

This skill explains how NV-Generate-CTMR takes a **3D label mask** (typically produced by the `mask-generation` skill or pulled from the training-mask database) and **synthesizes a paired CT or MR image** from it. The image LDM is conditioned on the mask via a ControlNet branch.

Code entry point: `scripts.infer_image_from_mask.ldm_conditional_sample_one_image`.

## TL;DR

```
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
| `cfg_guidance_scale` | float | Classifier-free guidance scale on the tumor signal. `0` disables CFG. |

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

### 6. Classifier-free guidance (CFG) — optional

When `cfg_guidance_scale > 0`:

```python
# Build a tumor-free version of the conditioning mask via remove_tumors()
combine_label_no_tumor = F.interpolate(remove_tumors(combine_label.squeeze(0)).unsqueeze(0).float(),
                                       size=output_size, mode="nearest")
controlnet_cond_vis_no_tumor = binarize_labels(combine_label_no_tumor.as_tensor().long()).half()
```

Each forward then batches `(tumor-conditioned, tumor-free)` together and combines:

```python
eps_t, eps_uncond = diffusion_unet(...).chunk(2)
eps = eps_uncond + cfg_guidance_scale * (eps_t - eps_uncond)
```

The unconditional branch keeps the body+organs but **drops tumor labels**, so CFG specifically strengthens the tumor signal. Set `cfg_guidance_scale=0` to disable.

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
| `cfg_guidance_scale` | `LDMSampler.__init__` | `0` = off, typical values `1..5`. Higher = stronger tumor enforcement, but more artifacts. |
| `autoencoder_sliding_window_infer_size` | `LDMSampler.__init__` | Must be divisible by 16. Larger = fewer tiles but more VRAM. |
| `autoencoder_sliding_window_infer_overlap` | `LDMSampler.__init__` | `[0, 1)`. Higher = smoother blending, more compute. |
| `noise_factor` | hardcoded `1.0` in `LDMSampler.__init__` | Scales the initial noise. |

## Output-size + spacing constraints

Validated by `check_input_ct` and `check_input_mr` (in `scripts/sample_mask.py`):
- `output_size[0] == output_size[1]`
- `output_size[0] ∈ {256, 384, 512}`
- `output_size[2] ∈ {128, 256, 384, 512, 640, 768}`
- `spacing[0] == spacing[1]`
- `spacing[0] ∈ [0.5, 3.0]` mm, `spacing[2] ∈ [0.5, 5.0]` mm
- FOV_xy ≥ 256 mm for head, ≥ 384 mm for abdomen / body

See the `image-only-inference` skill for recommended `(dim, spacing)` per anatomical target.

## Code references

| Symbol | File | Notes |
|---|---|---|
| `ldm_conditional_sample_one_image` | `scripts/infer_image_from_mask.py` | core sampling function |
| `crop_img_body_mask` | `scripts/infer_image_from_mask.py` | background HU regularization |
| `LDMSampler.sample_one_pair` | `scripts/sample.py` | wraps `ldm_conditional_sample_one_image` with LDMSampler state |
| `remove_tumors`, `augmentation` | `scripts/augmentation.py` | CFG unconditional mask + training-time mask aug |
| `binarize_labels`, `dynamic_infer` | `scripts/utils.py` | encoding + sliding-window glue |
