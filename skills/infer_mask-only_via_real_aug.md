---
name: infer_mask-only_via_real_aug
description: How to generate a mask by looking up a real training mask and applying augmentation (Path B). Covers body_region and anatomy_list filtering, the exact-match vs closest-match fallback, and the augmentation pipeline (body zoom + per-tumor elastic). Trigger when the user asks "how does the mask DB lookup work", "what does body_region filter", "how is the real mask augmented", "why did the pipeline resample a mask", or wants to reuse a training mask rather than synthesise from scratch.
---

# Mask generation via real mask + augmentation (Path B)

Path B retrieves a real training mask that matches the requested anatomy and applies random augmentation so the output is not a verbatim copy. It is chosen automatically when **both** `controllable_anatomy_size` and `controllable_demographics` in `config_infer.json` are **empty / null**. If either is non-empty, Path A (diffusion) runs instead.

## When Path B runs

```json
// config_infer.json — Path B trigger
"controllable_anatomy_size": [],
"controllable_demographics": null,
"body_region": ["chest", "abdomen"],
"anatomy_list": ["spleen", "right kidney", "left kidney"]
```

Path A (diffusion) runs when **either** `controllable_anatomy_size` or `controllable_demographics` is non-empty. Both must be empty/null for Path B. See [`infer_mask-only_via_diffusion_model`](infer_mask-only_via_diffusion_model.md).

## Workflow

```text
body_region + anatomy_list + spacing + output_size
        │
        ▼ find_masks() — exact match
  candidate_mask_files
        │ (empty?)
        ├─ No  → select_mask() → shuffle + always if_aug=True
        └─ Yes → find_closest_masks() → resample to output_size/spacing
                        │
                        ▼
              read_mask_information(mask_file)
                        │
                        ▼ if_aug=True
                 augmentation(mask, output_size)
                   ├─ augmentation_body()   → RandZoom(0.99–1.01)
                   └─ per-tumor elastic     → Rand3DElastic / RandAffine
                        │ retry up to 1000× until all requested organs present
                        ▼
                   [augmented mask]
                        │
                        ▼ image stage (ControlNet)
                   [synthetic image + mask pair]
```

## Filtering: `body_region` and `anatomy_list`

`find_masks()` returns candidate masks that satisfy **all** of:

- Contain every body region listed in `body_region` (e.g. `"chest"`, `"abdomen"`, `"pelvis"`).
- Contain every anatomy label in `anatomy_list` (organ names from `label_dict.json`, e.g. `"spleen"`, `"right kidney"`).
- If no tumor is in `anatomy_list`, the candidate must also be **tumor-free**.
- If `check_spacing_and_output_size=True` (exact match), spacing and output_size must also match.

`body_region` values are normalised internally — `"thorax"` and `"chest"` are equivalent.

## Fallback: closest-match + resample

When exact match finds fewer candidates than `num_img`:

1. `find_closest_masks()` searches the DB for the mask whose spacing and output size are nearest to the requested values (ignoring the exact anatomy/region filter).
2. The selected mask is resampled to the target `output_size` and `spacing` via `ensure_output_size_and_spacing()`.

A log line `"Resample mask file to get desired output size and spacing"` confirms this path was taken.

## Augmentation pipeline

Augmentation is **always applied** (`if_aug=True` for every selected mask). It calls `augmentation(mask, output_size)` in `scripts/augmentation.py`:

| Sub-step | Transform | Parameters |
|----------|-----------|-----------|
| Body zoom | `RandZoom` | min=0.99, max=1.01, mode=nearest, prob=1.0 |
| Lung tumor | `Rand3DElastic` | — |
| Pancreatic tumor | `Rand3DElastic` | — |
| Hepatic tumor | `Rand3DElastic` | — |
| Colon tumor | `Rand3DElastic` | — |
| Bone lesion | `RandAffine` | — |

After each augmentation attempt the pipeline verifies all requested organs are still present. It retries up to **1000 times** (`MAX_COUNT`) before raising an error — if this limit is hit, the anatomy filter or augmentation parameters are too restrictive for the selected mask.

## Key config knobs

| Key | Default | Notes |
|-----|---------|-------|
| `controllable_anatomy_size` | `[]` | Must be `[]` to trigger Path B. |
| `body_region` | `["chest", "abdomen"]` | Filters candidate masks by body coverage. |
| `anatomy_list` | `["spleen", "right kidney", "left kidney"]` | Organ names (from `label_dict.json`) that must be present in the candidate. **Path B only** — not used when Path A runs. Note: `"lung"` is a valid `controllable_anatomy_size` conditioning name but is **not** a valid `anatomy_list` entry (use the individual lobe names, e.g. `"left lung lower lobe"`). |
| `output_size` | `[512, 512, 512]` | Exact match filter; mismatches trigger closest-match + resample. |
| `spacing` | `[1.0, 1.0, 1.0]` | Exact match filter; mismatches trigger closest-match + resample. |
| `all_mask_files_json` | set in config | Path to `configs/all_mask_files_*.json` — the mask DB index. |

## Relevant scripts

| Script | Role |
|--------|------|
| `scripts/find_masks.py` | `find_masks()` — exact-match DB query by body_region, anatomy_list, spacing, output_size. |
| `scripts/sample.py` (`LDMSampler`) | `select_mask()` — shuffles candidates and sets `if_aug=True`; `find_closest_masks()` — fallback nearest-neighbor search; `read_mask_information()` — loads the selected mask. |
| `scripts/augmentation.py` | `augmentation()` — body zoom + per-tumor elastic; sub-functions per tumor type. |
| `configs/all_mask_files_*.json` | The mask DB index files listing all training masks with their metadata. |

## Related skills

- [`infer_mask-only`](infer_mask-only.md) — overview of both paths and output format.
- [`infer_mask-only_via_diffusion_model`](infer_mask-only_via_diffusion_model.md) — Path A: synthesise a mask from scratch with size control.
- [`infer_mask-image-paired`](infer_mask-image-paired.md) — run command and GPU-memory presets.
