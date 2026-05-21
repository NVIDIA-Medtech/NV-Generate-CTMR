---
name: mask-generation
description: Explains how NV-Generate-CTMR generates a 3D organ-label mask from scratch (no input image needed) using anatomy_size conditioning. Trigger when the user asks "how does mask generation work", "how do I generate a synthetic mask", "what does ldm_conditional_sample_one_mask do", "explain the mask diffusion model", or any low-level question about the mask LDM pipeline.
---

# Mask generation (NV-Generate-CTMR)

This skill explains how NV-Generate-CTMR samples a **3D body-region label mask** from scratch, conditioned on a small organ-size vector. The mask is the input to the ControlNet-conditioned image LDM (see the `image-from-mask` skill).

Code entry point: `scripts.sample_mask.ldm_conditional_sample_one_mask`.

## TL;DR

```text
[anatomy_size  ──┐
 (10-d vector)]   │  cross-attention conditioning
                  ▼
[random noise]──▶[Mask Diffusion UNet]──▶[mask latent (4-ch)]
                        DDPM loop
                                              │
                                              ▼ sliding-window AE decode
                                  [125-channel softmax]
                                              │ argmax
                                              ▼
                                       [labels 0..124]
                                              │ remap_labels via label_dict_124_to_132.json
                                              ▼
                            [MAISI 132-class label NIfTI (with body=200)]
                                              │ tumor-aware + general post-process
                                              ▼
                                          [final mask]
```

## Inputs to `ldm_conditional_sample_one_mask`

| Argument | Type | Description |
|---|---|---|
| `autoencoder` | `AutoencoderKlMaisi` | The frozen 8-bit-input / 125-channel-output mask AE. Loaded from `models/mask_generation_autoencoder.pt`. |
| `diffusion_unet` | `DiffusionModelUNet` | The mask DM (operating on the AE's 4-channel latents). Loaded from `models/mask_generation_diffusion_unet.pt`. |
| `noise_scheduler` | `DDPMScheduler` | **Mask DM is trained with DDPM**, so `num_inference_steps` should equal `num_train_timesteps` (typically 1000). |
| `scale_factor` | float | Latent normalization factor used at AE training. Default `1.0055984258651733`. |
| `anatomy_size` | list/Tensor of length 10 | Conditioning vector — see slot table below. |
| `latent_shape` | tuple | Mask latent shape; e.g. `(4, 64, 64, 64)` for a 256³ output. |
| `label_dict_remap_json` | str | Path to `configs/label_dict_124_to_132.json` (the AE-channel → MAISI-label crosswalk). |
| `num_inference_steps` | int | DDPM sampling steps. Match `noise_scheduler.num_train_timesteps` for best quality. |
| `autoencoder_sliding_window_infer_size` | list[int] | ROI for AE-decode sliding window. Default `[96, 96, 96]`. |
| `autoencoder_sliding_window_infer_overlap` | float | Window overlap fraction. Default `0.6667`. |

### The `anatomy_size` slot table

The 10-d vector has organ-fixed slots:

| Index | Organ | Index | Tumor |
|---|---|---|---|
| 0 | gallbladder | 5 | lung tumor |
| 1 | liver | 6 | pancreatic tumor |
| 2 | stomach | 7 | hepatic tumor |
| 3 | pancreas | 8 | colon cancer primaries |
| 4 | colon | 9 | bone lesion |

Each value is one of:

- A float in `[0, 1]` — desired size on a normalized scale
- `-1.0` — "no preference / don't care"

`LDMSampler.prepare_anatomy_size_condition` constructs the fully-specified vector:

1. Builds a sparse 10-d vector from user-specified `(organ, size)` tuples.
2. Looks up `configs/all_anatomy_size_conditions.json` — a database of size vectors from real training cases.
3. Picks the database entry with minimum L1 distance to the user's specified slots (`-1.0`/None ignored).
4. **Overwrites** the user-specified slots in the chosen database vector with the user's exact values.

The result lies near the training distribution — the model has actually seen similar combinations.

## Algorithm step by step

### 1. Initialize random noise

```python
latents = initialize_noise_latents(latent_shape, device)  # shape (1, 4, H, W, D) in fp16
```

### 2. Prepare the conditioning tensor

```python
anatomy_size = torch.FloatTensor(anatomy_size).unsqueeze(0).unsqueeze(0).half().to(device)
# shape: (1, 1, 10) — batch × seq_len × cross_attention_dim
```

`cross_attention_dim=10` in the mask DM config matches this exactly.

### 3. DDPM sampling loop

```python
noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps)
inferer_ddpm = DiffusionInferer(noise_scheduler)
latents = inferer_ddpm.sample(
    input_noise=latents,
    diffusion_model=diffusion_unet,
    scheduler=noise_scheduler,
    conditioning=anatomy_size,    # passed as cross-attention context
)
```

`DiffusionInferer.sample` runs the standard DDPM denoising loop:

```text
for t = T, T-1, ..., 1:
    eps_hat = diffusion_unet(latents, timesteps=t, context=anatomy_size)
    latents = scheduler.step(eps_hat, t, latents)
```

The UNet has `with_conditioning=true` so the `context=anatomy_size` argument routes through cross-attention layers.

### 4. Decode latent through the mask AE (sliding window)

The mask AE decoder produces a **125-channel softmax** per spatial location. For 3D volumes the full pass doesn't fit in GPU memory, so a sliding-window inferer tiles the volume:

```python
inferer = SlidingWindowInferer(
    roi_size=autoencoder_sliding_window_infer_size,   # default [96, 96, 96]
    sw_batch_size=1,
    mode="gaussian",                                  # blended overlap
    overlap=autoencoder_sliding_window_infer_overlap, # default 0.6667
    sw_device=device,
    device=torch.device("cpu"),                       # store on CPU to save VRAM
)
synthetic_mask = dynamic_infer(inferer, recon_model, latents)
# shape: (1, 125, H_full, W_full, D_full)
```

`ReconModel.forward(z) = autoencoder.decode_stage_2_outputs(z / scale_factor)` undoes the latent normalization before decoding.

### 5. Softmax → argmax → labels 0..124

```python
synthetic_mask = torch.softmax(synthetic_mask, dim=1)
synthetic_mask = torch.argmax(synthetic_mask, dim=1, keepdim=True)
# (1, 1, H, W, D) integer values in 0..124
```

These are **AE-channel indices**, not the final MAISI label values.

### 6. Remap to MAISI vocabulary

```python
synthetic_mask = remap_labels(synthetic_mask, label_dict_remap_json)
```

`configs/label_dict_124_to_132.json` is a 125-entry table where each row reads `organ_name: [ae_channel, maisi_label_value]`. E.g. `"spleen": [2, 3]` means AE channel 2 → MAISI label 3. The remap converts the 0..124 image into the MAISI label vocabulary (1..132 with gaps, plus `body=200`).

### 7. Post-processing

```python
labels = [23, 24, 26, 27, 128]  # lung/panc/hep/colon-cancer/bone-lesion
target_tumor_label = None
for index, size in enumerate(anatomy_size[0, 0, 5:10]):
    if size.item() != -1.0:
        target_tumor_label = labels[index]
data = general_mask_generation_post_process(data, target_tumor_label, device)
```

`general_mask_generation_post_process` (in `scripts/utils.py`):

- Suppresses spurious non-largest connected components for major organs
- Keeps only the largest tumor component for the requested tumor (if any)
- Dilates/erodes to clean boundaries

## Output

A single 3D tensor of integer MAISI labels with shape `(1, 1, H, W, D)`. Includes standard MAISI organ labels (1..132 with gaps) and `body=200`.

## Output-size and spacing constraints

The pretrained mask DM was trained at **256×256×256 × 1.5 mm isotropic**. Downstream resampling (`LDMSampler.ensure_output_size_and_spacing`) maps the generated mask to the user's requested `output_size` + `spacing`; major upsampling will degrade boundaries.

## Two paths to obtain a mask in `LDMSampler.sample_multiple_images`

1. **Generate from scratch** — triggered when `controllable_anatomy_size` is non-empty. Calls `ldm_conditional_sample_one_mask`.
2. **Pick a real training mask** — triggered when `controllable_anatomy_size` is empty. Uses `find_masks` / `find_closest_masks` to retrieve a database entry that matches `body_region` + `anatomy_list`. No diffusion involved.

Both paths produce a label tensor that then feeds the `image-from-mask` skill.

## Code references

| Symbol | File | Notes |
|---|---|---|
| `ldm_conditional_sample_one_mask` | `scripts/sample_mask.py` | core sampling function |
| `LDMSampler.prepare_anatomy_size_condition` | `scripts/sample.py` | sparse user input → fully-specified 10-d vector |
| `LDMSampler.sample_one_mask` | `scripts/sample.py` | thin wrapper |
| `LDMSampler.prepare_one_mask_and_meta_info` | `scripts/sample.py` | wraps `sample_one_mask` + spacing affine + body-region indices |
| `binarize_labels`, `remap_labels`, `general_mask_generation_post_process` | `scripts/utils.py` | label utilities |
