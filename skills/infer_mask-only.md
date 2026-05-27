---
name: infer_mask-only
description: Explains the mask-generation stage of NV-Generate-CTMR — how to drive it via config_infer.json (Path A "diffusion from scratch" vs Path B "training-mask database"), the anatomy_size conditioning vector, and the output mask format. Trigger when the user asks "how do I control the mask shape", "what does controllable_anatomy_size do", "how does Path A / Path B differ", or wants to understand the mask stage of the paired pipeline.
---

# Mask-only generation (NV-Generate-CTMR)

This skill covers the **mask-generation stage** that runs inside the paired pipeline (`scripts.inference`). It is not a standalone CLI — masks come out of the same `python -m scripts.inference` invocation that generates the paired image (see [`infer_mask-image-paired`](infer_mask-image-paired.md) for the run command).

The mask stage produces a 3D MAISI-labeled volume that subsequently conditions the image LDM. **CT-only** — the mask DM was trained on CT masks, and there is no MR equivalent.

## Workflow

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

## Configuration: Path A vs Path B

The mask stage has two paths, dispatched by `controllable_anatomy_size` in `config_infer.json`:

| Path | Trigger | What happens |
|---|---|---|
| **Path A** — diffusion from scratch | `controllable_anatomy_size` non-empty | Mask DM samples a new mask conditioned on the anatomy_size vector. |
| **Path B** — training-mask DB lookup | `controllable_anatomy_size` empty | Look up a real training mask from `configs/all_mask_files_*.json` matching `body_region` + `anatomy_list` + `spacing` + `output_size`; apply light augmentation so the output isn't a verbatim copy. |

Knobs that drive these:

| Knob | Path | Effect |
|---|---|---|
| `controllable_anatomy_size` | A vs B switch | Non-empty list of `(organ_name, size)` tuples — at most 10 entries, at most 1 tumor — triggers Path A. Empty triggers Path B. |
| `body_region` | B | Filters the mask DB. Any subset of `["head", "chest", "thorax", "abdomen", "pelvis", "lower"]`. |
| `anatomy_list` | A and B | Required organs. Used by Path B's `find_masks` filter; also used by both paths as the post-process `filter_mask_with_organs` (only listed organs survive in the output). |
| `output_size`, `spacing` | A and B | Target shape and voxel spacing — see [`infer_mask-image-paired`](infer_mask-image-paired.md) for the GPU-memory presets table. |
| `mask_generation_num_inference_steps` | A | Always **1000** — the mask DM is DDPM regardless of the image-DM variant; lowering it silently degrades mask quality. |

## Input: the `anatomy_size` slot vector (Path A only)

When Path A runs, the user-specified `(organ_name, size)` tuples are turned into a 10-d vector with fixed slots:

| Index | Organ | Index | Tumor |
|---|---|---|---|
| 0 | gallbladder | 5 | lung tumor |
| 1 | liver | 6 | pancreatic tumor |
| 2 | stomach | 7 | hepatic tumor |
| 3 | pancreas | 8 | colon cancer primaries |
| 4 | colon | 9 | bone lesion |

Each slot value is either:

- A float in `[0, 1]` — desired size on a normalized scale, **or**
- `-1.0` — "no preference / don't care".

The pipeline snaps the user-specified vector to the closest entry in `configs/all_anatomy_size_conditions.json` (a database of size vectors from real training cases), then **overwrites** the user-specified slots with the user's exact values. This keeps the conditioning vector near the training distribution while honouring user intent.

## Output

A 3D integer NIfTI of MAISI labels with shape `(H, W, D)`. Contains MAISI organ labels (1..132 with gaps) and the body envelope `200`. Saved by the paired CLI as `sample_<timestamp>_label.nii.gz` alongside the paired image.

## Output-size and spacing constraints

The pretrained mask DM was trained at **256×256×256 × 1.5 mm isotropic** (Path A). Resampling to your requested `output_size` and `spacing` happens automatically; major upsampling degrades label boundaries, so stay close to 256³ × 1.5 mm when feasible. For Path B, mask candidates are drawn from a training-FOV distribution — the closer your requested FOV is to a mode of that distribution, the less reshaping is needed.

## Related scripts

| Script | Role |
|---|---|
| `scripts/sample_mask.py` | Mask-generation pipeline. Contains `ldm_conditional_sample_one_mask` (Path A) and the DB-lookup helpers `find_masks` / `find_closest_masks` (Path B). |
| `scripts/sample.py` (`LDMSampler`) | Orchestrator: chooses Path A or B based on `controllable_anatomy_size`, then chains the image stage. |
| `scripts/inference.py` | CLI entry point for the paired pipeline (mask stage + image stage together). |
| `scripts/utils.py` | Label utilities: `binarize_labels`, `remap_labels`, `general_mask_generation_post_process`. |

## Related skills

- [`infer_mask-image-paired`](infer_mask-image-paired.md) — the CLI that drives this stage end-to-end.
- [`infer_image-from-mask`](infer_image-from-mask.md) — what happens to the mask after this stage.
- [`infer_image-only`](infer_image-only.md) — image-only generation (no mask DM involved).
