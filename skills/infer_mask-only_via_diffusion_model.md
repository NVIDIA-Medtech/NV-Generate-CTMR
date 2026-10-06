---
name: infer_mask-only_via_diffusion_model
description: How to generate a synthetic mask from scratch using the v2 mask diffusion model (Path A). Covers the 19-d conditioning vector (14 anatomy + 5 demographics), RFlow scheduler, CFG guidance, and when this path is chosen vs the real-mask DB path. Trigger when the user asks "how do I generate a new mask with the diffusion model", "how does controllable_anatomy_size work", "how many inference steps for the mask", or wants to control organ/tumor sizes or patient demographics in a fully synthetic mask.
---

# Mask generation via diffusion model — v2 (Path A)

Path A runs the **v2 mask diffusion UNet** (`DiffusionModelUNetMaisiAdaGN`, AdaGN conditioning) to synthesise a brand-new mask conditioned on a 19-d vector: 14 anatomy size slots + 5 demographics slots. It is chosen automatically when `controllable_anatomy_size` is **non-empty** or `controllable_demographics` is **non-null/non-empty**.

## When Path A runs

```json
// config_infer.json — Path A trigger, only anatomy size
"controllable_anatomy_size": [["bone lesion", 0.5]],
"controllable_demographics": null
```

```json
// config_infer.json — Path A with anatomy size + demographics
"controllable_anatomy_size": [["bone lesion", 0.5]],
"controllable_demographics": [["age", 55]]
```

```json
// config_infer.json — Path A trigger, only demographics
"controllable_anatomy_size": [],
"controllable_demographics": [["sex", "M"]]
```

```json
// demographics can be any subset of fields is valid; examples:
"controllable_anatomy_size": [],
"controllable_demographics": [["age", 55], ["sex", "M"], ["bmi", 24.5]]  // all three
"controllable_demographics": [["age", 55]]                                // age only
"controllable_demographics": [["sex", "M"], ["bmi", 24.5]]               // sex + bmi
```

`controllable_demographics` accepts any non-empty subset of the supported fields — omitted fields are left unconditioned (`-1`). Set to `null` or `[]` to disable demographic conditioning entirely.

### Demographics field examples

```json
["age", 55]      // age in years (training range 19–87)
["sex", "M"]     // "M" or "F"
["bmi", 24.5]    // BMI in kg/m² (preferred over weight; training range 18–63)
["weight", 72]   // body weight in kg (training range 22–144; prefer bmi instead)
```

Path A is triggered when **either** `controllable_anatomy_size` or `controllable_demographics` is non-empty. Both may be empty to fall back to Path B (real-mask DB lookup); see [`infer_mask-only_via_real_aug`](infer_mask-only_via_real_aug.md).

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

A fixed 14-slot float vector; each slot is a normalised size in `[0, 1]` or `-1` (no preference).

> ⚠️ **At most ONE anatomy slot may be set.** The v2 model is single-target conditioned. Passing `[["liver", 0.5], ["pancreas", 0.7]]` raises a `ValueError` at startup — input validation rejects multiple entries before generation begins.

| Slot | Name | Notes |
|------|------|-------|
| 0 | liver | |
| 1 | spleen | |
| 2 | stomach | |
| 3 | pancreas | |
| 4 | colon | |
| 5 | left kidney | |
| 6 | right kidney | |
| 7 | lung | |
| 8 | gallbladder | |
| 9 | lung tumor | |
| 10 | pancreatic tumor | |
| 11 | hepatic tumor | ⚠️ Low recall — the model rarely generates this label reliably. Avoid using this slot; results are unpredictable. |
| 12 | colon cancer primaries | |
| 13 | bone lesion | |

### Demographics slots (14–18)

Optional patient demographics, normalized to `[0, 1]`. Any unspecified slot uses `-1`.

| Slot | Name | Units | Training range | Notes |
|------|------|-------|----------------|-------|
| 14 | age | years | 19–87 | supported |
| 15 | sex | `"M"` / `"F"` | — | supported |
| 16 | weight | kg | 22–144 | supported but prefer `bmi` |
| 17 | bmi | kg/m² | 18–63 | supported; preferred over `weight` |
| 18 | height | cm | — | slot exists but not trained; do not use |

Rules:

- Provide `weight` OR `bmi`, not both.
- Do not set `height` — the model was not trained with this field.
- Demographics in `config_infer.json` use original units; the pipeline normalizes internally.

## `output_size` and `spacing` — FOV matters

> ⚠️ **FOV (= `output_size × spacing`) is the #1 quality knob.** The v2 mask DM generates at **256³** with flexible spacing — choose a realistic anatomy FOV and derive `spacing = FOV / output_size`.

Derive spacing from a realistic anatomy FOV:

```text
spacing[i] = FOV[i] / output_size[i]
```

Example (for **paired inference** where `output_size = [512, 512, 768]`): chest-to-pelvis CT → FOV ≈ 410×410×768 mm → `spacing = [0.8, 0.8, 1.0]`. For mask-only runs keep `output_size = [256, 256, 256]` and pick `spacing` from the desired anatomy FOV.

## Key config knobs

| Key | Default | Notes |
|-----|---------|-------|
| `controllable_anatomy_size` | `[]` (empty = Path B) | A single `[organ_name, size]` pair (list-of-lists), e.g. `[["bone lesion", 0.5]]`. Non-empty triggers Path A. Only one entry is supported. |
| `controllable_demographics` | `null` | Optional list of `[name, value]` pairs in original units, or `null`. Example: `[["age", 55], ["sex", "M"], ["bmi", 24.5]]`. Valid names: `age` (yr), `sex` (`"M"`/`"F"`), `bmi` (kg/m²), `weight` (kg, prefer bmi). Do not set `height`. |
| `mask_generation_num_inference_steps` | `100` | RFlow steps. **Do not set to 1000** — the v2 model uses RFlow, not DDPM. |
| `mask_generation_cfg_guidance_scale` | `2.0` | CFG scale. `0.0` disables guidance (unconditioned). |
| `output_size` | `[256, 256, 256]` | The mask DM always generates at 256³. The result is resampled: (1) spacing → target `spacing`, (2) pad/crop → `output_size`. So `output_size` × `spacing` defines the physical FOV of the final mask. For mask-only runs keep `[256, 256, 256]`; for paired runs set to your desired output volume. |
| `spacing` | `[1.5, 1.5, 2.0]` | Voxel spacing in mm. The v2 model supports flexible spacing. |

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
