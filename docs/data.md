# Data preparation

This page is a **short guide** to training data for NV-Generate-CTMR. **Authoritative tables, dataset links, and JSON examples** live in the canonical document: **[`data/README.md`](../data/README.md)** (MAISI data preparation). Always check that file for the latest counts and policies.

**Disclaimer (from the data README):** We are not the hosts of the data. Read each dataset’s requirements and usage policies and credit the authors.

## Quick links (same structure as the data README)

| Topic | In [`data/README.md`](../data/README.md) |
|--------|-------------------------------------------|
| VAE — autoencoder_v1.pt | [§1.1](../data/README.md#11-autoencoder_v1pt) |
| VAE — autoencoder_v2.pt | [§1.2](../data/README.md#12-autoencoder_v2pt) |
| Diffusion — DDPM CT | [§2.1](../data/README.md#21-diff_unet_3d_ddpm-ctpt) |
| Diffusion — rectified-flow CT (+ HNSCC) | [§2.2](../data/README.md#22-diff_unet_3d_rflow-ctpt) |
| Diffusion — rectified-flow MR | [§2.3](../data/README.md#23-diff_unet_3d_rflow-mrpt) |
| ControlNet — DDPM CT (20 datasets) | [§3.1](../data/README.md#31-controlnet_3d_ddpm-ctpt) |
| ControlNet — rflow CT (+ HNSCC) | [§3.2](../data/README.md#32-controlnet_3d_rflow-ctpt) |
| ControlNet — finetune example (C4KC-KiTS) | [§3.3](../data/README.md#33-example-to-finetune-on-a-new-dataset) |
| Questions / bugs / reference | [§4](../data/README.md#4-questions-and-bugs), [Reference](../data/README.md#reference) |

## Summary of training data (high level)

### Foundation autoencoder (VAE)

- **autoencoder_v1.pt** — **37,243** CT train / **1,963** CT val; **17,887** MRI train / **940** MRI val (chest, abdomen, head/neck, brain, etc.). License: research and commercial-friendly; see [NV-Generate-CT](https://huggingface.co/nvidia/NV-Generate-CT). [Details & per-dataset table → §1.1](../data/README.md#11-autoencoder_v1pt)
- **autoencoder_v2.pt** — Adds **eight** more sources (CT and MR) on top of v1; combined totals **39,831** CT / **2,380** val and **20,024** MRI / **1,270** val. Research-only (not cleared for commercial use); see [NV-Generate-MR](https://huggingface.co/nvidia/NV-Generate-MR). [Details & table → §1.2](../data/README.md#12-autoencoder_v2pt)

### Latent diffusion (U-Net) training

- **diff_unet_3d_ddpm-ct.pt** — **10,277** CT volumes across **24** datasets. [Per-dataset counts → §2.1](../data/README.md#21-diff_unet_3d_ddpm-ctpt)
- **diff_unet_3d_rflow-ct.pt** — Same CT mix plus **HNSCC** (**1,225** volumes). [Table → §2.2](../data/README.md#22-diff_unet_3d_rflow-ctpt)
- **diff_unet_3d_rflow-mr.pt** — **16,291** distinct utilized MR images from **17** datasets (after excluding volumes with fewer than **48** slices). The README breaks out modality columns (T1w, T2w, FLAIR, etc.) and “original” vs training counts. [Full table → §2.3](../data/README.md#23-diff_unet_3d_rflow-mrpt)

### ControlNet (CT, paired image / mask)

- **controlnet_3d_ddpm-ct.pt** — **6,330** CT volumes (**5,058** train / **1,272** val) over **20** datasets. [Table → §3.1](../data/README.md#31-controlnet_3d_ddpm-ctpt)
- **controlnet_3d_rflow-ct.pt** — Adds **HNSCC** (**1,225** volumes) on top of the DDPM ControlNet mix. [Table → §3.2](../data/README.md#32-controlnet_3d_rflow-ctpt)
- **Finetuning on a new site** — Example **C4KC-KiTS** subset, downloads, folder layout, JSON schema, and preprocessing notes (embeddings, VISTA pseudo labels, resampling to multiples of 128). [Full walkthrough → §3.3](../data/README.md#33-example-to-finetune-on-a-new-dataset)

## Bring your own data for ControlNet Training

Training volumes are expected as **NIfTI** (`.nii.gz`). For ControlNet-style training with latent embeddings, the README’s finetune section is in [example JSON block in §3.3](../data/README.md#33-example-to-finetune-on-a-new-dataset).

## Where to go next

- **Every dataset name, volume count, and citation link:** [`data/README.md`](../data/README.md)
- **Training and inference tutorials:** see the main repository [README](../README.md) and the other Markdown files in this `docs/` directory.
