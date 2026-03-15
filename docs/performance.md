# Performance: accuracy, speed, and GPU memory usage

This document describes inference time cost, GPU memory usage during inference and training, and how to tune parameters to fit your hardware.

## Inference Time Cost and GPU Memory Usage

| `output_size` | Peak Memory | VAE Time + DM Time (`maisi3d-ddpm`) | VAE Time + DM Time (`maisi3d-rflow`) | latent size | `autoencoder_sliding_window_infer_size` | `autoencoder_tp_num_splits` | VAE Time | DM Time (`maisi3d-ddpm`) | DM Time (`maisi3d-rflow`) |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| [256x256x128](../configs/config_infer_16g_256x256x128.json) | 15.0G | 58s | 3s | 4x64x64x32 | >=[64,64,32], not used | 2 | 1s | 57s | 2s |
| [256x256x256](../configs/config_infer_16g_256x256x256.json) | 15.4G | 86s | 8s | 4x64x64x64 | [48,48,64], 4 patches | 4 | 5s | 81s | 3s |
| [512x512x128](../configs/config_infer_16g_512x512x128.json) | 15.7G | 146s | 13s | 4x128x128x32 | [64,64,32], 9 patches | 2 | 8s | 138s | 5s |
| | | | | | | | | | |
| [256x256x256](../configs/config_infer_24g_256x256x256.json) | 22.7G | 83s | 5s | 4x64x64x64 | >=[64,64,64], not used | 4 | 2s | 81s | 3s |
| [512x512x128](../configs/config_infer_24g_512x512x128.json) | 21.0G | 144s | 11s | 4x128x128x32 | [80,80,32], 4 patches | 2 | 6s | 138s | 5s |
| [512x512x512](../configs/config_infer_24g_512x512x512.json) | 22.8G | 598s | 48s | 4x128x128x128 | [64,64,48], 36 patches | 2 | 29s | 569s | 19s |
| | | | | | | | | | |
| [512x512x512](../configs/config_infer_32g_512x512x512.json) | 28.4G | 599s | 49s | 4x128x128x128 | [80,80,48], 16 patches | 4 | 30s | 569s | 19s |
| | | | | | | | | | |
| [512x512x512](../configs/config_infer_80g_512x512x512.json) | 45.3G | 601s | 51s | 4x128x128x128 | [80,80,80], 8 patches | 2 | 32s | 569s | 19s |
| [512x512x768](../configs/config_infer_80g_512x512x768.json) | 49.7G | 961s | 87s | 4x128x128x192 | [80,80,96], 12 patches | 4 | 57s | 904s | 30s |

**Table:** Inference Time Cost and GPU Memory Usage. `DM Time` refers to the time required for diffusion model inference. `VAE Time` refers to the time required for VAE decoder inference. The total inference time is the sum of `DM Time` and `VAE Time`. The experiment was conducted on an A100 80G GPU.

During inference, the peak GPU memory usage occurs during the VAE's decoding of latent features.
To reduce GPU memory usage, we can either increase `autoencoder_tp_num_splits` or reduce `autoencoder_sliding_window_infer_size`.
Increasing `autoencoder_tp_num_splits` has a smaller impact on the generated image quality, while reducing `autoencoder_sliding_window_infer_size` may introduce stitching artifacts and has a larger impact on the generated image quality.

When `autoencoder_sliding_window_infer_size` is equal to or larger than the latent feature size, the sliding window will not be used, and the time and memory costs remain the same.

## Training GPU Memory Usage

The VAE is trained on patches and can be trained using a 16G GPU if the patch size is set to a small value, such as [64, 64, 64]. Users can adjust the patch size to fit the available GPU memory. For the released model, we initially trained the autoencoder on 16G V100 GPUs with a small patch size of [64, 64, 64], and then continued training on 32G V100 GPUs with a larger patch size of [128, 128, 128].

The DM and ControlNet are trained on whole images rather than patches. The GPU memory usage during training depends on the size of the input images. There is no big difference on memory usage between `maisi3d-ddpm` and `maisi3d-rflow`.

| image size | latent size | Peak Memory |
|---|:---|:-:|
| 256x256x128 | 4x64x64x32 | 5G |
| 256x256x256 | 4x64x64x64 | 8G |
| 512x512x128 | 4x128x128x32 | 12G |
| 512x512x256 | 4x128x128x64 | 21G |
| 512x512x512 | 4x128x128x128 | 39G |
| 512x512x768 | 4x128x128x192 | 58G |
