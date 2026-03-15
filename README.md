# NV-Generate-CTMR

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![HuggingFace CT](https://img.shields.io/badge/HuggingFace-NV--Generate--CT-yellow.svg)](https://huggingface.co/nvidia/NV-Generate-CT)
[![HuggingFace MR](https://img.shields.io/badge/HuggingFace-NV--Generate--MR-yellow.svg)](https://huggingface.co/nvidia/NV-Generate-MR)
[![HuggingFace MR-Brain](https://img.shields.io/badge/HuggingFace-NV--Generate--MR--Brain-yellow.svg)](https://huggingface.co/nvidia/NV-Generate-MR-Brain)
[![arXiv MAISI-v1](https://img.shields.io/badge/arXiv-2409.11169-red.svg)](https://arxiv.org/abs/2409.11169)
[![arXiv MAISI-v2](https://img.shields.io/badge/arXiv-2508.05772-red.svg)](https://arxiv.org/abs/2508.05772)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)

3D Latent Diffusion Models (LDM) for generating large CT and MRI images with corresponding segmentation masks. Supports variable volume size and voxel spacing with precise control of organ/tumor size.

| | |
|:---:|:---:|
| ![MR example](assets/MR_example.png) | ![Generated CT and segmentation](assets/typical-generated-ct-image-corresponding-segmentation-condition.gif) |
| *Generated MR T2w prostate and T1w brain image* | *Generated CT image/mask pair* |

## Overview

NV-Generate-CTMR generates high-resolution synthetic 3D medical volumes using latent diffusion models built on the MAISI (Medical AI for Synthetic Imaging) framework. It produces CT images with paired segmentation masks and MRI volumes across multiple contrasts — enabling synthetic training data generation, data augmentation for rare pathologies, and privacy-preserving data sharing.

Key capabilities:

- **CT generation** with paired 132-class segmentation masks, supporting volumes up to 512x512x768 voxels with controllable organ and tumor size
- **MRI generation** across T1, T2, FLAIR, and additional contrasts for brain, abdomen, breast, and prostate anatomy
- **Brain MRI synthesis** with cross-sequence ControlNet for generating matched multi-contrast brain volumes (T1w, T2w, FLAIR, SWI)
- **Variable resolution** with configurable volume size and voxel spacing for each generation

**[Live Demo](https://build.nvidia.com/nvidia/maisi)** (no GPU required)

## News

- **[March 2026]** — Released NV-Generate-MR-Brain for brain MRI synthesis across T1w, T2w, FLAIR, and SWI contrasts with cross-sequence ControlNet
- **[October 2025]** — Released rectified flow models `rflow-mr` for fast high-resolution 3D MR image generation. Upgraded previous MAISI repo to this NV-Generate-CTMR repo.
- **[March 2025]** — Released rectified flow models `rflow-ct` for **fast** high-resolution 3D CT image generation and paired CT image/mask synthesis. `rflow-ct` is **33x faster** than `ddpm-ct` and generates better quality images for the head region and small output volumes.
- **[August 2024]** — Initial release `ddpm-ct` supporting 3D latent diffusion (DDPM) for CT image generation and paired CT image/mask synthesis.

## Model Variants

| | `ddpm-ct` | `rflow-ct` | `rflow-mr` | `rflow-mr-brain` |
|---|---|---|---|---|
| **Modality** | CT | CT | MRI | MRI (Brain) |
| **Model Weights** | [NV-Generate-CT](https://huggingface.co/nvidia/NV-Generate-CT) | [NV-Generate-CT](https://huggingface.co/nvidia/NV-Generate-CT) | [NV-Generate-MR](https://huggingface.co/nvidia/NV-Generate-MR) | [NV-Generate-MR-Brain](https://huggingface.co/nvidia/NV-Generate-MR-Brain) |
| **Architecture** | MAISI-v1 (DDPM) | MAISI-v2 (Rectified Flow) | MAISI-v2 (Rectified Flow) | MAISI-v2 + ControlNet |
| **Paper** | [MAISI-v1](https://arxiv.org/abs/2409.11169) | [MAISI-v2](https://arxiv.org/abs/2508.05772) | [MAISI-v2](https://arxiv.org/abs/2508.05772) | [MAISI-v2](https://arxiv.org/abs/2508.05772) |
| **Inference Steps** | 1000 | 30 | 30 | 30 |
| **Max Volume** | 512x512x768 | 512x512x768 | 512x512x128 | 512x512x128 |
| **Use Case** | CT image/mask pair generation | CT image/mask pair generation | MR image-only generation | Brain multi-contrast synthesis |
| **License** | [NVIDIA Open Model](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) | [NVIDIA Open Model](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) | [NVIDIA Non-Commercial](https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf) | [NVIDIA Open Model](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) |

**Recommendations**: Use `rflow-ct` for CT projects (ready for whole-body inference). Use `rflow-mr-brain` for brain MRI with matched multi-contrast volumes. Use `rflow-mr` for other MRI anatomies (fine-tune on your own data).

## Quick Start

Requires Python 3.11+ and an NVIDIA GPU with at least 16GB VRAM.

### Installation

```bash
pip install -r requirements.txt
```

### CT Paired Image/Mask Generation

```bash
export MONAI_DATA_DIRECTORY="./temp_work_dir"
network="rflow"
generate_version="rflow-ct"
python -m scripts.inference -t ./configs/config_network_${network}.json -i ./configs/config_infer.json -e ./configs/environment_${generate_version}.json --random-seed 0 --version ${generate_version}
```

See also: [inference_tutorial.ipynb](inference_tutorial.ipynb)

### CT Image-Only Generation

```bash
network="rflow"
generate_version="rflow-ct"
python -m scripts.download_model_data --version ${generate_version} --root_dir "./" --model_only
python -m scripts.diff_model_infer -t ./configs/config_network_${network}.json -e ./configs/environment_maisi_diff_model_${generate_version}.json -c ./configs/config_maisi_diff_model_${generate_version}.json
```

### MR Image Generation

Change `"modality"` in [configs/config_maisi_diff_model_rflow-mr.json](configs/config_maisi_diff_model_rflow-mr.json) according to [configs/modality_mapping.json](configs/modality_mapping.json) to control the output MR contrast. Supported contrasts: T1/T2 brain, FLAIR skull-stripped brain, T2 prostate, T1 breast, T1/T2 abdomen.

```bash
network="rflow"
generate_version="rflow-mr"
python -m scripts.download_model_data --version ${generate_version} --root_dir "./" --model_only
python -m scripts.diff_model_infer -t ./configs/config_network_${network}.json -e ./configs/environment_maisi_diff_model_${generate_version}.json -c ./configs/config_maisi_diff_model_${generate_version}.json
```

See also: [inference_diff_unet_tutorial.ipynb](inference_diff_unet_tutorial.ipynb)

## Documentation

| Guide | Description |
|-------|-------------|
| [Setup](docs/setup.md) | Full installation guide, dependencies, model weight download |
| [Inference](docs/inference.md) | Detailed inference parameters, spacing tables, GPU memory usage |
| [Training](docs/training.md) | VAE, Diffusion Model, and ControlNet training guides |
| [Data Preparation](docs/data.md) | Dataset formats and preparation steps |
| [Evaluation](docs/evaluation.md) | FID evaluation tool and benchmark results |
| [Troubleshooting](docs/troubleshooting.md) | Common issues and solutions |
| [Applications](docs/applications.md) | Community adaptations (MR-to-CT synthesis) |
| [Inference Tutorial](inference_tutorial.ipynb) | Quick start CT paired generation (notebook) |
| [Diffusion Inference](inference_diff_unet_tutorial.ipynb) | CT/MR image-only generation (notebook) |
| [Training Tutorials](train_vae_tutorial.ipynb) | VAE, diffusion, and ControlNet training |

## Performance

On the unseen [autoPET 2023](https://www.nature.com/articles/s41597-022-01718-3) benchmark:

| Model | FID Score | Inference Steps | Speed vs ddpm-ct |
|-------|----------|-----------------|------------------|
| `rflow-ct` | **5.124** | 30 | **33x faster** |
| `ddpm-ct` | 6.083 | 1000 | baseline |

Detailed GPU memory usage and inference timing in [docs/inference.md](docs/inference.md).

## License

| Component | License |
|-----------|---------|
| Source code | [Apache 2.0](LICENSE) |
| NV-Generate-CT weights | [NVIDIA Open Model](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) |
| NV-Generate-MR weights | [NVIDIA Non-Commercial](https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf) |
| NV-Generate-MR-Brain weights | [NVIDIA Open Model](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) |

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

## Citation

```bibtex
@inproceedings{chen2025maisi,
  title={MAISI: Medical AI for Synthetic Imaging},
  author={Chen, Pengfei and others},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2025},
  url={https://arxiv.org/abs/2409.11169}
}
```

```bibtex
@article{chen2025maisiv2,
  title={MAISI-v2: Accelerated 3D High-Resolution Medical Image Synthesis with Rectified Flow},
  author={Chen, Pengfei and others},
  journal={arXiv preprint arXiv:2508.05772},
  year={2025},
  url={https://arxiv.org/abs/2508.05772}
}
```

## Resources

- [NV-Generate-CT on HuggingFace](https://huggingface.co/nvidia/NV-Generate-CT) -- CT model weights and model card
- [NV-Generate-MR on HuggingFace](https://huggingface.co/nvidia/NV-Generate-MR) -- MR model weights and model card
- [NV-Generate-MR-Brain on HuggingFace](https://huggingface.co/nvidia/NV-Generate-MR-Brain) -- Brain MRI model weights and model card
- [MAISI Live Demo](https://build.nvidia.com/nvidia/maisi) -- Try online without GPU
- [MAISI-v1 Paper (WACV 2025)](https://arxiv.org/pdf/2409.11169)
- [MAISI-v2 Paper](https://arxiv.org/pdf/2508.05772)
- Built with [MONAI](https://monai.io/) -- Medical Open Network for AI