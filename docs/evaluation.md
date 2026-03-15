# Evaluation: FID Score

## Overview

We provide the `compute_fid_2-5d_ct.py` script that calculates the Frechet Inception Distance (FID) between two 3D medical datasets (e.g., **real** vs. **synthetic** images). It uses a **2.5D** feature-extraction approach across three orthogonal planes (XY, YZ, ZX) and leverages **distributed GPU processing** (via PyTorch's `torch.distributed` and NCCL) for efficient, large-scale computations.

## Key Features

- **Distributed Processing**
  Scales to multiple GPUs and larger datasets by splitting the workload across devices.

- **2.5D Feature Extraction**
  Uses a slice-based technique, applying a 2D model across all slices in each dimension.

- **Flexible Preprocessing**
  Supports optional center-cropping, padding, and resampling to target shapes or voxel spacings.

## Usage Example

Suppose your **real** dataset root is `path/to/real_images`, and you have a `real_filelist.txt` that lists filenames line by line, such as:

```text
case001.nii.gz
case002.nii.gz
case003.nii.gz
```

You also have a **synthetic** dataset in `path/to/synth_images` with a corresponding `synth_filelist.txt`. You can run the script as follows:

```bash
torchrun --nproc_per_node=2 compute_fid_2-5d_ct.py \
  --model_name "radimagenet_resnet50" \
  --real_dataset_root "path/to/real_images" \
  --real_filelist "path/to/real_filelist.txt" \
  --real_features_dir "datasetA" \
  --synth_dataset_root "path/to/synth_images" \
  --synth_filelist "path/to/synth_filelist.txt" \
  --synth_features_dir "datasetB" \
  --enable_center_slices_ratio 0.4 \
  --enable_padding True \
  --enable_center_cropping True \
  --enable_resampling_spacing "1.0x1.0x1.0" \
  --ignore_existing True \
  --num_images 100 \
  --output_root "./features/features-512x512x512" \
  --target_shape "512x512x512"
```

This command will:

1. Launch a distributed run with 2 GPUs.
2. Load each `.nii.gz` file from your specified `real_filelist` and `synth_filelist`.
3. Apply 2.5D feature extraction across the XY, YZ, and ZX planes.
4. Compute FID to compare **real** vs. **synthetic** feature distributions.

For more details, see the in-code docstring in [`../scripts/compute_fid_2-5d_ct.py`](../scripts/compute_fid_2-5d_ct.py) or consult our documentation for a deeper dive into function arguments and the underlying implementation.

## Results and Evaluation

We retrained several state-of-the-art diffusion model-based methods using our dataset. The results in the table and figure below show that our method outperforms previous methods on an unseen dataset ([autoPET 2023](https://www.nature.com/articles/s41597-022-01718-3)). Our method shows superior performance to previous methods based on all [Frechet Inception Distance (FID)](https://papers.nips.cc/paper/2017/hash/8a1d694707eb0fefe65871369074926d-Abstract.html) scores on different 2D planes. Here we compared the generated images with real images of size 512 x 512 x 512 and spacing 1.0 x 1.0 x 1.0 mm3.

| Method | FID (XY Plane) | FID (YZ Plane) | FID (ZX Plane) | FID (Average) |
|--------|:-:|:-:|:-:|:-:|
| [DDPM](https://proceedings.neurips.cc/paper_files/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf) | 18.524 | 23.696 | 25.604 | 22.608 |
| [LDM](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf) | 16.853 | 10.191 | 10.093 | 12.379 |
| [HA-GAN](https://ieeexplore.ieee.org/document/9770375) | 17.432 | 10.266 | 13.572 | 13.757 |
| MAISI (`ddpm-ct`) | 3.301 | 5.838 | 9.109 | 6.083 |
| MAISI (`rflow-ct`) | 2.685 | 4.723 | 7.963 | 5.124 |

**Table 1.** Comparison of Frechet Inception Distance scores between our foundation model and retrained baseline methods
using the unseen public dataset [autoPET 2023](https://www.nature.com/articles/s41597-022-01718-3) as the reference.

![Generated image examples](https://developer-blogs.nvidia.com/wp-content/uploads/2024/06/generated-medical-image-method-comparison-1.png)

**Figure 1.** Qualitative comparison of generated images between baseline methods
(retrained using our large-scale dataset) and our method. The MAISI here refers to `ddpm-ct`.

| Dataset | Model | LPIPS | SSIM | PSNR | GPU |
|---|---|---|---|---|---|
| MSD Task07 | MAISI-v1 VAE | **0.038** | **0.978** | **37.266** | **0h** |
| | Dedicated VAE | 0.047 | 0.971 | 34.750 | 619h |
| MSD Task08 | MAISI-v1 VAE | 0.046 | 0.970 | 36.559 | **0h** |
| | Dedicated VAE | **0.041** | **0.973** | **37.110** | 669h |
| Brats18 | MAISI-v1 VAE | **0.026** | **0.977** | **39.003** | **0h** |
| | Dedicated VAE | 0.030 | 0.975 | 38.971 | 672h |

**Table 2:** Performance comparison of the `MAISI-v1 VAE` model on out-of-distribution datasets (i.e., unseen during MAISI-v1 VAE training) versus `Dedicated VAE` models (i.e., train from scratch on in-distribution data). The "GPU" column shows additional GPU hours for training with one 32G V100 GPU. MAISI VAE model achieved comparable results without additional GPU resource expenditure on unseen datasets.
