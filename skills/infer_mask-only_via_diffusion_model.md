---
name: infer_mask-only_via_diffusion_model
description: How to generate a synthetic mask from scratch using the mask diffusion model (Path A). Covers the controllable_anatomy_size conditioning vector, the 10-slot anatomy index, DDPM loop settings, and when this path is chosen vs the real-mask DB path. Trigger when the user asks "how do I generate a new mask with the diffusion model", "how does controllable_anatomy_size work", "how many inference steps for the mask", or wants to control organ/tumor sizes in a fully synthetic mask.
---

# Mask generation via diffusion model (Path A)

Path A runs the **mask diffusion UNet** to synthesise a brand-new mask conditioned on a user-specified organ/tumor size vector. It is chosen automatically when `controllable_anatomy_size` in `config_infer.json` is **non-empty**.

## When Path A runs

```json
// config_infer.json — Path A trigger
"controllable_anatomy_size": [["bone lesion", 0.5]]
```

If `controllable_anatomy_size` is an empty list `[]`, the pipeline falls back to Path B (real-mask DB lookup). See [`infer_mask-only_via_real_aug`](infer_mask-only_via_real_aug.md).

## Workflow

```text
controllable_anatomy_size
        │
        ▼
prepare_anatomy_size_condition()
  ├─ snap to nearest entry in configs/all_anatomy_size_conditions.json
  └─ overwrite snapped slots with user's exact values
        │
        ▼ 10-d float vector
[random noise] ──▶ [Mask Diffusion UNet] ──▶ [mask latent (4-ch)]
                         DDPM, 1000 steps
                                │
                                ▼ AE sliding-window decode
                    [125-ch softmax → argmax]
                                │
                                ▼ remap via label_dict_124_to_132.json
                    [MAISI 132-class NIfTI + body=200]
                                │
                                ▼ tumor-aware + general post-process
                          [final mask]
```

## The anatomy_size conditioning vector

A fixed 10-slot float vector; each slot is a normalised size in `[0, 1]` or `-1` (no preference):

| Slot | Organ/Tumor |
|------|-------------|
| 0 | gallbladder |
| 1 | liver |
| 2 | stomach |
| 3 | pancreas |
| 4 | colon |
| 5 | lung tumor |
| 6 | pancreatic tumor |
| 7 | hepatic tumor |
| 8 | colon cancer primaries |
| 9 | bone lesion |

Rules:
- At most **10 entries**, at most **1 tumor slot** non-`-1` at a time.
- Unspecified organs default to `-1` (the model picks a size from the training distribution).
- The pipeline snaps the full vector to the nearest real training-set entry first, then overwrites the specified slots with the user's exact values — this keeps the conditioning near the training distribution.

## Key config knobs

| Key | Default | Notes |
|-----|---------|-------|
| `controllable_anatomy_size` | `[]` | List of `[organ_name, size]` pairs. Non-empty triggers Path A. |
| `mask_generation_num_inference_steps` | 1000 | **Always keep at 1000.** The mask DM is DDPM — lowering this silently degrades mask quality (unlike the image DM which supports DDIM/rFlow). |
| `output_size` | `[512, 512, 512]` | Target shape; the mask DM was trained at 256³ so major upsampling degrades label boundaries. Stay close to 256³ when feasible. |
| `spacing` | `[1.5, 1.5, 1.5]` | Voxel spacing in mm. Training spacing is 1.5 mm isotropic. |

## Relevant scripts

| Script | Role |
|--------|------|
| `scripts/sample_mask.py` | Core sampler: `ldm_conditional_sample_one_mask` — DDPM loop → softmax/argmax → label remap → post-process. |
| `scripts/sample.py` (`LDMSampler.prepare_anatomy_size_condition`) | Snaps user vector to training distribution, prepares the 10-d conditioning tensor. |
| `scripts/inference.py` | CLI entry point that triggers Path A when `controllable_anatomy_size` is non-empty. |
| `configs/all_anatomy_size_conditions.json` | Database of real training-set size vectors used for snapping. |

## Related skills

- [`infer_mask-only`](infer_mask-only.md) — overview of both paths and output format.
- [`infer_mask-only_via_real_aug`](infer_mask-only_via_real_aug.md) — Path B: use a real training mask instead.
- [`infer_mask-image-paired`](infer_mask-image-paired.md) — run command and GPU-memory presets.
