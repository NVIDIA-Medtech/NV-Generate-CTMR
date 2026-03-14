# Setup and Installation

## Prerequisites

- Python 3.11+
- NVIDIA GPU with at least 16GB VRAM (see [GPU requirements](#gpu-requirements) below)
- CUDA 11.8+ or CUDA 12.x

## Installation

### Using pip

```bash
git clone https://github.com/nvidia-medtech/NV-Generate-CTMR.git
cd NV-Generate-CTMR
pip install -r requirements.txt
```

### Key Dependencies

| Package | Version | Notes |
|---------|---------|-------|
| `torch` | >=2.1.0 | PyTorch with CUDA support |
| `monai` | >=1.5.0 | Required for MR support; CT works with >=1.3.2 |
| `numpy` | >=1.24.0 | Numerical computing |
| `scipy` | >=1.10.0 | Scientific computing |
| `scikit-image` | >=0.20.0 | Image processing utilities |
| `nibabel` | >=5.0.0 | NIfTI file I/O |
| `matplotlib` | >=3.7.0 | Visualization |
| `einops` | >=0.7.0 | Tensor operations for model architecture |
| `huggingface_hub` | >=0.20.0 | Model weight download |
| `tqdm` | >=4.65.0 | Progress bars |
| `fire` | >=0.5.0 | CLI for FID evaluation script |
| `tensorboard` | >=2.14.0 | Training logging |

### MONAI Version Requirements

- **For CT generation only (`ddpm-ct`, `rflow-ct`)**: `monai>=1.3.2`
- **For MR generation (`rflow-mr`)**: `monai>=1.5.0`

The `requirements.txt` specifies `monai>=1.5.0` to cover both CT and MR use cases.

## GPU Requirements

GPU requirement depends on the size of the images you want to generate:

- **512x512x128**: minimum 16GB GPU memory for both training and inference
- **512x512x512**: minimum 40GB for training, 24GB for inference

See [docs/inference.md](inference.md) for detailed GPU memory usage tables by output size and configuration.

## Downloading Model Weights

Model weights are automatically downloaded from HuggingFace when running inference. You can also download them manually:

```bash
# Download CT models (rflow-ct)
python -m scripts.download_model_data --version rflow-ct --root_dir "./" --model_only

# Download MR models (rflow-mr)
python -m scripts.download_model_data --version rflow-mr --root_dir "./" --model_only

# Download legacy CT models (ddpm-ct)
python -m scripts.download_model_data --version ddpm-ct --root_dir "./" --model_only
```

Model weights are hosted on HuggingFace:

- [nvidia/NV-Generate-CT](https://huggingface.co/nvidia/NV-Generate-CT) -- CT image generation (ddpm-ct, rflow-ct)
- [nvidia/NV-Generate-MR](https://huggingface.co/nvidia/NV-Generate-MR) -- MR image generation (rflow-mr)

## Pre-commit Hooks

For development, install pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

This will automatically run Ruff linting and formatting checks before each commit.
