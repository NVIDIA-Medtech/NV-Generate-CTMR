---
name: infer_mask-only
description: Overview of the mask-generation stage in NV-Generate-CTMR — Path A (diffusion from scratch) vs Path B (training-mask DB lookup), key config knobs, and output format. Trigger when the user asks "how do I control the mask shape", "what does controllable_anatomy_size do", or "how does Path A / Path B differ".
---

# Mask-only generation (NV-Generate-CTMR)

The mask-generation stage runs inside `scripts.inference` (not a standalone CLI — see [`infer_mask-image-paired`](infer_mask-image-paired.md)). It produces a 3D MAISI-labeled volume that conditions the image LDM. **CT-only.**

## Path A vs Path B

| | **Path A — diffusion from scratch** | **Path B — real mask + augmentation** |
|---|---|---|
| Trigger | `controllable_anatomy_size` non-empty **or** `controllable_demographics` non-null | `controllable_anatomy_size: []` and `controllable_demographics: null` |
| How | v2 mask DM (AdaGN) samples a new mask conditioned on a 19-d vector (14 anatomy + 5 demographics slots) | Looks up a real training mask matching `body_region` + `anatomy_list`; applies random augmentation |
| Deep-dive | [`infer_mask-only_via_diffusion_model`](infer_mask-only_via_diffusion_model.md) | [`infer_mask-only_via_real_aug`](infer_mask-only_via_real_aug.md) |

## Key config knobs

| Key | Path | Notes |
|-----|------|-------|
| `controllable_anatomy_size` | switch | `[["organ_name", size]]` → Path A. `[]` → Path B (unless demographics set). |
| `controllable_demographics` | switch | Non-null → Path A. `null` → Path B (unless anatomy size set). |
| `body_region` | B | Filters the mask DB, e.g. `["chest", "abdomen"]`. |
| `anatomy_list` | A + B | Required organ label IDs; used by Path B filter and both paths' post-process. |
| `output_size` | A | Fixed `[256, 256, 256]` for Path A. |
| `spacing` | A + B | Target voxel spacing in mm. |
| `mask_generation_num_inference_steps` | A | **100** — v2 mask DM uses RFlow, not DDPM. |
| `mask_generation_cfg_guidance_scale` | A | CFG scale, default `2.0`. |

## Output

A 3D integer NIfTI of MAISI labels (1–132 with gaps) plus body envelope `200`, saved as `sample_<timestamp>_label.nii.gz` alongside the paired image.

## Related skills

- [`infer_mask-only_via_diffusion_model`](infer_mask-only_via_diffusion_model.md) — Path A: 19-d conditioning, RFlow settings, demographics format.
- [`infer_mask-only_via_real_aug`](infer_mask-only_via_real_aug.md) — Path B: DB filtering, closest-match fallback, augmentation pipeline.
- [`infer_mask-image-paired`](infer_mask-image-paired.md) — the CLI that drives this stage end-to-end.
- [`infer_image-from-mask`](infer_image-from-mask.md) — what happens to the mask after this stage.
- [`infer_image-only`](infer_image-only.md) — image-only generation (no mask DM involved).
