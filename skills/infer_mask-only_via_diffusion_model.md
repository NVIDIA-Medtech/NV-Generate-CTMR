---
name: infer_mask-only_via_diffusion_model
description: How to generate a synthetic mask from scratch using the v2 mask diffusion model (Path A). Covers the 19-d conditioning vector (14 anatomy + 5 demographics), RFlow scheduler, CFG guidance, and when this path is chosen vs the real-mask DB path. Trigger when the user asks "how do I generate a new mask with the diffusion model", "how does controllable_anatomy_size work", "how many inference steps for the mask", or wants to control organ/tumor sizes or patient demographics in a fully synthetic mask.
---

# Mask generation via diffusion model — v2 (Path A)

Path A runs the **v2 mask diffusion UNet** (`DiffusionModelUNetMaisiAdaGN`, AdaGN conditioning) to synthesise a brand-new mask conditioned on a 19-d vector: 14 anatomy size slots + 5 demographics slots. It is chosen automatically when `controllable_anatomy_size` in `config_infer.json` is **non-empty**.

## When Path A runs

```json
// config_infer.json — Path A trigger
"controllable_anatomy_size": [["bone lesion", 0.5]],
"controllable_demographics": null
```

If `controllable_anatomy_size` is an empty list `[]`, the pipeline falls back to Path B (real-mask DB lookup). See [`infer_mask-only_via_real_aug`](infer_mask-only_via_real_aug.md).

## Workflow

```text
controllable_anatomy_size + controllable_demographics
        │
        ▼
prepare_anatomy_size_condition()   (LDMSampler, scripts/sample.py)
  ├─ anatomy part: fill named slot(s), rest = -1
  └─ demographics part: normalize to [0,1], missing = -1
        │
        ▼ 19-d float vector [anatomy(14) | demographics(5)]
[random noise] ──▶ [Mask Diffusion UNet v2 (AdaGN)]  ──▶ [mask latent (4-ch)]
                     RFlowScheduler, 100 steps
                     CFG scale 2.0 (uncond + scale*(cond - uncond))
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

## The 19-d conditioning vector

### Anatomy slots (0–13)

A fixed 14-slot float vector; each slot is a normalised size in `[0, 1]` or `-1` (no preference). **At most ONE slot may be set** — the v2 model is single-target conditioned.

| Slot | Name |
|------|------|
| 0 | liver |
| 1 | spleen |
| 2 | stomach |
| 3 | pancreas |
| 4 | colon |
| 5 | left kidney |
| 6 | right kidney |
| 7 | lung |
| 8 | gallbladder |
| 9 | lung tumor |
| 10 | pancreatic tumor |
| 11 | hepatic tumor |
| 12 | colon cancer primaries |
| 13 | bone lesion |

### Demographics slots (14–18)

Optional patient demographics, normalized to `[0, 1]`. Any unspecified slot uses `-1`.

| Slot | Name | Units | Normalization max | Training range |
|------|------|-------|-------------------|----------------|
| 14 | age | years | 120 | 19–87 |
| 15 | sex | M=1 / F=0 | — | — |
| 16 | weight | kg | 200 | 22–144 |
| 17 | bmi | kg/m² | 75 | 18–63 |
| 18 | height | cm | 200 | 160–190 |

Rules:
- Provide **only one** of `weight` or `bmi` (not both).
- `height` conditioning has unreliable effect — prefer `age`, `sex`, `weight`, or `bmi`.
- Setting `controllable_anatomy_size` together with 3+ demographics is heavily constrained and out-of-distribution; generation quality is not guaranteed.
- Demographics in `config_infer.json` use original units; the pipeline normalizes internally.

## Key config knobs

| Key | Default | Notes |
|-----|---------|-------|
| `controllable_anatomy_size` | `[["bone lesion", 0.5]]` | A single `[organ_name, size]` pair (list-of-lists). Non-empty triggers Path A. |
| `controllable_demographics` | `null` | Optional list of `[name, value]` pairs in original units, or `null`. |
| `mask_generation_num_inference_steps` | `100` | RFlow steps. **Do not set to 1000** — the v2 model uses RFlow, not DDPM. |
| `mask_generation_cfg_guidance_scale` | `2.0` | CFG scale. `0.0` disables guidance (unconditioned). |
| `output_size` | `[256, 256, 256]` | Target shape. The mask DM is trained at 256³ — stay close to this. |
| `spacing` | `[1.5, 1.5, 2.0]` | Voxel spacing in mm. Training spacing is 1.5 mm isotropic; mild anisotropy is supported. |

## Checkpoint

`mask_generation_diffusion_unet_v2.pt` — `DiffusionModelUNetMaisiAdaGN` with AdaGN cross-attention. Not compatible with the v1 DDPM checkpoint.

## Relevant scripts

| Script | Role |
|--------|------|
| `scripts/sample_mask.py` | Core sampler: `ldm_conditional_sample_one_mask` — RFlow+CFG loop → softmax/argmax → label remap → post-process. Also defines `ANATOMY_SIZE_IDX` slot map and `_DEMOG_MAX` normalization. |
| `scripts/sample.py` (`LDMSampler.prepare_anatomy_size_condition`) | Builds the 19-d conditioning tensor from user inputs. |
| `scripts/adagn_unet.py` | `DiffusionModelUNetMaisiAdaGN` model definition. |
| `scripts/inference.py` | CLI entry point that triggers Path A when `controllable_anatomy_size` is non-empty. |

## Related skills

- [`infer_mask-only`](infer_mask-only.md) — overview of both paths and output format.
- [`infer_mask-only_via_real_aug`](infer_mask-only_via_real_aug.md) — Path B: use a real training mask instead.
- [`infer_mask-image-paired`](infer_mask-image-paired.md) — run command and GPU-memory presets.
