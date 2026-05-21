---
name: download-models
description: How to download pretrained checkpoints (CT, MR, MR-Brain variants) and optional auxiliary data (mask database, anatomy-size-conditions JSON) for NV-Generate-CTMR inference. Trigger when the user asks "where are the checkpoints", "how do I download the model weights", "what does download_model_data.py do", or hits a missing-checkpoint error.
---

# Downloading pretrained models + auxiliary data

This skill covers `scripts/download_model_data.py` — the entry point for fetching everything you need before running inference.

## TL;DR

```bash
python -m scripts.download_model_data --version <VARIANT> --root_dir "./" [--model_only]
```

Where `<VARIANT>` is one of `rflow-ct`, `ddpm-ct`, `rflow-mr`, `rflow-mr-brain`.

Files land under `./models/` (weights) and `./datasets/` (optional auxiliary data).

## What gets downloaded per variant

Source: HuggingFace Hub via `huggingface_hub.hf_hub_download`. The script also pings each repo's `config.json` once so HuggingFace's download counter ticks.

### `rflow-ct` (CT, Rectified Flow — recommended for CT)

Always downloaded (`models/`):

- `autoencoder_v1.pt` (image AE) — from `nvidia/NV-Generate-CT`
- `mask_generation_autoencoder.pt` — from `nvidia/NV-Generate-CT`
- `mask_generation_diffusion_unet.pt` — from `nvidia/NV-Generate-CT`
- `diff_unet_3d_rflow-ct.pt` (image DM) — from `nvidia/NV-Generate-CT`
- `controlnet_3d_rflow-ct.pt` — from `nvidia/NV-Generate-CT`

If **`--model_only` is NOT set**, also downloads (`datasets/`):

- `all_anatomy_size_conditions.json` — anatomy-size database for `prepare_anatomy_size_condition` (Path A in mask-image paired inference)
- `all_masks_flexible_size_and_spacing_4000.zip` — training-mask database for `find_masks` (Path B)
- `candidate_masks_flexible_size_and_spacing_4000.json` — index for the mask DB

### `ddpm-ct` (CT, DDPM — slower but supports body_region input)

Same as `rflow-ct` but swaps:

- `diff_unet_3d_ddpm-ct.pt` + `controlnet_3d_ddpm-ct.pt` (instead of the rflow variants)

And uses `candidate_masks_flexible_size_and_spacing_3000.json` (smaller mask index).

### `rflow-mr-brain` (Brain MRI, Rectified Flow)

Only the image-DM stack (no mask DM, no ControlNet):

- `autoencoder_v1.pt` — from `nvidia/NV-Generate-CT` (yes, MR-Brain reuses the CT image AE)
- `diff_unet_3d_rflow-mr-brain_v0.pt` — from `nvidia/NV-Generate-MR-Brain`

### `rflow-mr` (Other MRI, Rectified Flow)

- `autoencoder_v2.pt` — from `nvidia/NV-Generate-MR`
- `diff_unet_3d_rflow-mr.pt` — from `nvidia/NV-Generate-MR`

## Output layout

```text
./
├── models/
│   ├── autoencoder_v1.pt                      # image AE (CT + MR-Brain)
│   ├── autoencoder_v2.pt                      # image AE (MR)
│   ├── mask_generation_autoencoder.pt         # mask AE (CT only)
│   ├── mask_generation_diffusion_unet.pt      # mask DM (CT only)
│   ├── diff_unet_3d_<variant>.pt              # image DM
│   └── controlnet_3d_<variant>.pt             # ControlNet (CT only)
└── datasets/
    ├── all_anatomy_size_conditions.json       # CT mask-generation database
    ├── all_masks_flexible_size_and_spacing_4000.zip
    └── candidate_masks_flexible_size_and_spacing_4000.json
```

The paths above are exactly what the `environment_<variant>.json` configs expect, so as long as you run `download_model_data` from the repo root with `--root_dir "./"`, no path edits are needed.

## When to use `--model_only`

- **Skip auxiliary data**: pass `--model_only` if you only intend to use `controllable_anatomy_size` (Path A, diffusion-generated masks). The mask database (`all_masks_flexible_size_and_spacing_4000.zip` etc.) is only needed for Path B (real-mask retrieval).
- **Full download (default)**: omit `--model_only` for the full paired-inference pipeline so both mask paths work.

## License gating

Some HuggingFace repos require you to accept their license terms before downloading. If you hit a 403, visit the repo page and accept terms:

- [nvidia/NV-Generate-CT](https://huggingface.co/nvidia/NV-Generate-CT) — NVIDIA Open Model License
- [nvidia/NV-Generate-MR](https://huggingface.co/nvidia/NV-Generate-MR) — NVIDIA Non-Commercial
- [nvidia/NV-Generate-MR-Brain](https://huggingface.co/nvidia/NV-Generate-MR-Brain) — NVIDIA Open Model License

After accepting, pass `--token YOUR_HF_TOKEN` or set `HF_TOKEN` in the environment.

## Failure modes and retries

| Symptom | Cause | Fix |
|---|---|---|
| `huggingface_hub.errors.GatedRepoError` | License not accepted | Visit repo page, accept terms, retry |
| `requests.exceptions.ConnectionError` | Network drop | Just re-run — `hf_hub_download` resumes from cache |
| Partial file (size mismatch) | Interrupted download | Delete the partial file in `./models/` or `./datasets/` and re-run |
| Wrong checkpoint shape at inference | Stale cached checkpoint after a model update | Re-run with `--overwrite` (if available) or manually delete the local file |

## Code reference

| Symbol | File |
|---|---|
| `download_model_data` | `scripts/download_model_data.py` |
| `fetch_to_hf_path_cmd` | `scripts/download_model_data.py` (the HF download loop with counter ping) |
| `ensure_hf_download_tracked` | `scripts/download_model_data.py` (pings `config.json` per repo) |

## Related skills

- `image-only-inference` — uses the image DM only (no ControlNet, no mask DM). Run with `--model_only`.
- `mask-image-paired-inference` — needs the full set (mask AE + mask DM + image DM + ControlNet). Run without `--model_only` if you'll use Path B.
- `mask-generation` / `image-from-mask` — algorithm details for the two stages.
