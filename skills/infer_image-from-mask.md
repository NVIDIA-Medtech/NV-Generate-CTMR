---
name: infer_image-from-mask
description: How to synthesize a paired CT image from an existing 3D label mask using NV-Generate-CTMR's ControlNet pipeline. Trigger when the user asks "how do I generate a CT from my mask", "what command runs image-from-mask inference", "what mask format does NV-Generate-CTMR expect", "how do I add the body envelope", or wants help running the README §2.6 command. CT-only (no MR ControlNet exists in this repo).
---

# Image-from-mask inference (NV-Generate-CTMR)

This skill covers running NV-Generate-CTMR on a **mask you already have** to synthesize a paired CT image. The image LDM is conditioned on the mask via a ControlNet branch. **CT-only** — the released ControlNet checkpoints were trained on CT masks; no MR ControlNet exists.

If you want the model to *also* generate the mask, use the [`infer_mask-image-paired`](infer_mask-image-paired.md) skill instead. If you want image-only generation (no mask, including MR), use the [`infer_image-only`](infer_image-only.md) skill.

## Command to run

```bash
network="rflow"                       # or "ddpm"
generate_version="rflow-ct"           # or "ddpm-ct"

python -m scripts.download_model_data --version ${generate_version} --root_dir "./"

python -m scripts.infer_image_from_mask \
    -t ./configs/config_network_${network}.json \
    -i ./configs/config_infer.json \
    -e ./configs/environment_${generate_version}.json \
    --mask /path/to/your_mask.nii.gz \
    --random-seed 0
```

Arguments:

- `-t` — network architecture (`config_network_rflow.json` or `config_network_ddpm.json`).
- `-i` — inference hyper-parameters (`config_infer.json`, or one of the `config_infer_<XXg>_<dim>.json` presets — see [Configuration](#configuration) below).
- `-e` — environment paths (`environment_rflow-ct.json` or `environment_ddpm-ct.json`).
- `--mask` — your mask NIfTI (see [Input: the mask](#input-the-mask) for the required format).
- `--random-seed` — optional integer for deterministic sampling.

### End-to-end example: synthesize a CT from a MAISI-vocabulary mask

Concrete worked example, assuming `~/your_mask.nii.gz` already contains MAISI 132-class labels with `body=200` (see [Input: the mask](#input-the-mask) for how to produce one):

```bash
# 1. Download weights (one-time, ~5 GB).
python -m scripts.download_model_data --version rflow-ct --root_dir "./"

# 2. Pick a config_infer preset matching your GPU memory + target output_size.
#    Example for a 24 GB GPU targeting 512×512×128 output:
#    use configs/config_infer_24g_512x512x128.json as-is (no edits needed
#    for a default CT run). Key values it already sets:
#      "output_size": [512, 512, 128],
#      "spacing":     [0.75, 0.75, 4.0],
#      "modality":                            1,    # CT
#      "cfg_guidance_scale_tumor":            0.0,  # off, the correct default
#      "num_inference_steps":                 30,
#      "autoencoder_sliding_window_infer_size":    [80, 80, 32],
#      "autoencoder_sliding_window_infer_overlap": 0.4,
#      "autoencoder_tp_num_splits":                2

# 3. Run inference.
python -m scripts.infer_image_from_mask \
    -t ./configs/config_network_rflow.json \
    -i ./configs/config_infer_24g_512x512x128.json \
    -e ./configs/environment_rflow-ct.json \
    --mask ~/your_mask.nii.gz \
    --random-seed 0
```

**Expected output**: two NIfTIs under the `output_dir` set in `environment_rflow-ct.json` — `sample_<timestamp>_image.nii.gz` (the synthesized CT, HU range `[-1000, 1000]`) and `sample_<timestamp>_label.nii.gz` (your mask, resampled if needed).

## Input: the mask

### Format

A **1-channel integer NIfTI** (`.nii` or `.nii.gz`) whose voxel values are drawn from the **MAISI 132-class label vocabulary** plus the body envelope:

| Value | Meaning |
|---|---|
| `0` | background |
| `1..132` (with gaps — see [`configs/label_dict.json`](../configs/label_dict.json)) | organ / structure labels (e.g. `1`=liver, `3`=spleen, `4`=pancreas, `5`=right kidney, `14`=left kidney, `28..32`=lung lobes, `33..57`=vertebrae) |
| `200` | **body envelope** — every body voxel not labeled with a specific organ |

`200` is the critical outer-body label. The CT ControlNet expects it; `nv-segment-ct` does not produce it. **You must add it yourself** during preprocessing.

### Producing a valid mask from a CT image

#### Option A (recommended): `nv-segment-ct` + add body envelope

1. Run [`nv-segment`](https://github.com/NVIDIA-Medtech/NV-Segment-CTMR) (the NV-Segment-CTMR bundle, modality `CT_BODY`) on your CT image. It outputs a 1-channel NIfTI **already in the MAISI vocabulary** — no label remapping needed.
2. Add the body envelope (label `200`) via [`scripts.utils.add_body_envelope(seg_mask, ct_image)`](../scripts/utils.py).
3. Save as a 1-channel integer NIfTI.

#### Option B: another segmenter + remap + add body envelope

1. Run your segmenter (e.g. TotalSegmentator) on the CT image.
2. **Remap label IDs** to the MAISI 132-class space defined in [`configs/label_dict.json`](../configs/label_dict.json) by matching anatomical structure names. Structures not present in MAISI must be set to `0`. If you skip this step or get the mapping wrong, the generated CT will be unusable.
3. Add the body envelope (label `200`) as in Option A step 2.
4. Save as a 1-channel integer NIfTI.

### Common pitfall: the 0..124 AE-channel space is NOT the right space

The mask AE inside this repo internally produces a 125-channel softmax that gets `argmax`'d to integer labels in `[0, 124]`. Those values are remapped to the MAISI 132-class vocabulary via [`configs/label_dict_124_to_132.json`](../configs/label_dict_124_to_132.json) before the CT ControlNet sees them. So:

- ✅ Correct user-mask input: MAISI 132-class labels with `body=200`.
- ❌ Incorrect: feeding `0..124` AE-channel-space labels.

If you have a mask in `0..124` space (e.g. from intermediate steps of a custom mask-DM pipeline), apply `remap_labels(mask, configs/label_dict_124_to_132.json)` first before passing it here.

### Validation in the CLI

`scripts/infer_image_from_mask.py::validate_user_mask` will:

- Confirm the mask is 1-channel integer NIfTI.
- Warn (not error) if any voxel value is outside the MAISI 132-class vocabulary (`{0..132} ∪ {200}`).
- Auto-resample shape/spacing to a valid `(output_size, spacing)` target (with a warning) if needed.

If many voxel values fall outside the vocabulary you almost certainly forgot a remap step.

## Configuration

The config you pass to `-i` is `config_infer.json` (or one of the `config_infer_<XXg>_<dim>.json` presets). For the **per-knob walkthrough** — `modality`, `output_size`, `spacing = FOV / output_size`, AE sliding-window knobs (`_size` / `_overlap` / `_tp_num_splits`) keyed to GPU memory, `cfg_guidance_scale_tumor`, `num_inference_steps` — see the **"How to configure a run"** section in [`infer_mask-image-paired.md`](infer_mask-image-paired.md). The same presets apply here — this script consumes the same `config_infer.json`.

Quick reminder of the CT-specific knob most relevant to this skill:

- `cfg_guidance_scale_tumor` — classifier-free guidance scale on tumor presence. CFG runs the model twice per step (mask as-is vs mask with `remove_tumors()`) and amplifies the difference, strengthening tumor signal in the synthesized image. `0` (default) = off, correct whenever the mask has no tumors or you want unsteered output. `1..5` = stronger tumor enforcement (more artifact risk above 5). Doubles per-step compute when `> 0`. Distinct from `cfg_guidance_scale_modality` in `config_maisi_diff_model_*.json` (MR image-only path) — see [`infer_image-only`](infer_image-only.md). Legacy un-suffixed `cfg_guidance_scale` is still accepted for one release with a `DeprecationWarning`.

### Hard constraints on `output_size` + `spacing`

Validated by `check_input_ct` / `check_input_mr` (in `scripts/sample_mask.py`):

- `output_size[0] == output_size[1]`
- `output_size[0] ∈ {256, 384, 512}`
- `output_size[2] ∈ {128, 256, 384, 512, 640, 768}`
- `spacing[0] == spacing[1]`
- `spacing[0] ∈ [0.5, 3.0]` mm, `spacing[2] ∈ [0.5, 5.0]` mm
- FOV_xy ≥ 256 mm for head, ≥ 384 mm for abdomen / body

See [`infer_image-only`](infer_image-only.md) for the recommended `(dim, spacing)` per anatomical target.

## Output

Two files in the directory given by `output_dir` (from the environment config):

- `sample_<timestamp>_image.nii.gz` — synthetic CT volume.
- `sample_<timestamp>_label.nii.gz` — the (resampled) input mask returned for downstream filtering.

Image intensity ranges:

| Modality | Output type | Voxel value range |
|---|---|---|
| CT (modality code `1..7`) | int16 NIfTI | HU, clipped to `[-1000, 1000]` |
| MR (codes `8..32`) — n/a for this pipeline; provided for context | int16 NIfTI | `[0, +∞)` |

Background voxels (where the mask is `0`) are set to `a_min` (CT: `-1000` HU) via `crop_img_body_mask`, preserving the body silhouette.

## Related scripts

| Script | Role |
|---|---|
| `scripts/infer_image_from_mask.py` | CLI for this skill. Loads models, validates the user mask, calls `ldm_conditional_sample_one_image`, writes the NIfTIs. |
| `scripts/sample.py` (`LDMSampler.sample_one_pair`) | Library wrapper used by the paired pipeline. Re-exports the same core sampler. |
| `scripts/download_model_data.py` | Downloads ControlNet + image AE + image DM weights. Run once before inference. |
| `scripts/utils.py` (`add_body_envelope`, `binarize_labels`, `remap_labels`) | Mask preprocessing utilities — needed for Options A/B above and for converting 0..124 → 132-class. |
