# Training Guide

## Prerequisites

- GPU with sufficient VRAM (see [GPU memory requirements](#training-gpu-memory-usage) below)
- Training data prepared according to [data preparation](data.md)
- Model weights downloaded (see [setup guide](setup.md))

Training data preparation can be found in [../data/README.md](../data/README.md).

## 3D Autoencoder (VAE) Training

Please refer to [train_vae_tutorial.ipynb](../train_vae_tutorial.ipynb) for the tutorial for MAISI VAE model training.

The information for the training hyperparameters and data processing parameters, like learning rate and patch size, are stored in [../configs/config_maisi_vae_train.json](../configs/config_maisi_vae_train.json). The provided configuration works for 16G V100 GPU. Please feel free to tune the parameters for your datasets and device.

### Dataset Preprocessing Parameters

- `"random_aug"`: bool, whether to add random data augmentation for training data.
- `"spacing_type"`: choose from `"original"` (no resampling involved), `"fixed"` (all images resampled to same voxel size), and `"rand_zoom"` (images randomly zoomed, valid when `"random_aug"` is True).
- `"spacing"`: None or list of three floats. If `"spacing_type"` is `"fixed"`, all the images will be interpolated to the voxel size of `"spacing"`.
- `"select_channel"`: int, if multi-channel MRI, which channel it will select.

### Training Configuration Parameters

- `"batch_size"`: training batch size. Please consider increasing it if GPU memory is larger than 16G.
- `"patch_size"`: training patch size. For the released model, we first trained the autoencoder with small patch size [64,64,64], then continued training with patch size of [128,128,128].
- `"val_patch_size"`: Size of validation patches. If None, will use the whole volume for validation. If given, will central crop a patch for validation.
- `"val_sliding_window_patch_size"`: if the validation patch is too large, will use sliding window inference. Please consider increasing it if GPU memory is larger than 16G.
- `"val_batch_size"`: validation batch size.
- `"perceptual_weight"`: perceptual loss weight.
- `"kl_weight"`: KL loss weight, important hyper-parameter. If too large, decoder cannot recon good results from latent space. If too small, latent space will not be regularized enough for the diffusion model.
- `"adv_weight"`: adversarial loss weight.
- `"recon_loss"`: choose from 'l1' and 'l2'.
- `"val_interval"`: int, do validation every `"val_interval"` epochs.
- `"cache"`: float between 0 and 1, dataloader cache, choose small value if CPU memory is small.
- `"n_epochs"`: int, number of epochs to train. Please adjust it based on the size of your datasets. We used 280 epochs for the released model on 58k data.

## 3D Latent Diffusion Training

Please refer to [train_diff_unet_tutorial.ipynb](../train_diff_unet_tutorial.ipynb) for the tutorial for MAISI diffusion model training.

```bash
export NUM_GPUS_PER_NODE=8
network="rflow"
generate_version="rflow-ct"
torchrun \
    --nproc_per_node=${NUM_GPUS_PER_NODE} \
    --nnodes=1 \
    --master_addr=localhost --master_port=1234 \
    -m scripts.diff_model_train -t ./configs/config_network_${network}.json -c ./configs/config_maisi_diff_model_${generate_version}.json -e ./configs/environment_maisi_diff_model_${generate_version}.json -g ${NUM_GPUS_PER_NODE}
```

To run the diffusion model training script with MAISI Rectified flow for MRI, please run the code above with:

```bash
network="rflow"
generate_version="rflow-mr"
```

To run the diffusion model training script with MAISI DDPM for CT, please run the code above with:

```bash
network="ddpm"
generate_version="ddpm-ct"
```

## 3D ControlNet Training

Please refer to [train_controlnet_tutorial.ipynb](../train_controlnet_tutorial.ipynb) for the tutorial for MAISI controlnet model training.

We provide a [training config](../configs/config_maisi_controlnet_train.json) executing finetuning for pretrained ControlNet with a new class (i.e., Kidney Tumor).
When finetuning with other new class names, please update the `weighted_loss_label` in training config
and [label_dict.json](../configs/label_dict.json) accordingly. There are 8 dummy labels as deletable placeholders in default `label_dict.json` that can be used for finetuning. Users may apply any placeholder labels for fine-tuning purpose. If there are more than 8 new labels needed in finetuning, users can freely define numeric label indices less than 256. The current ControlNet implementation can support up to 256 labels (0~255).
Preprocessed dataset for ControlNet training and more details about data preparation can be found in the [README](../data/README.md).

### Training Configuration

The training was performed with the following:

- GPU: at least 60GB GPU memory for 512 x 512 x 512 volume
- Actual Model Input (the size of 3D image feature in latent space) for the latent diffusion model: 128 x 128 x 128 for 512 x 512 x 512 volume
- AMP: True

### Execute Training

To train with a single GPU, please run:

```bash
network="rflow"
generate_version="rflow-ct"
python -m scripts.train_controlnet -t ./configs/config_network_${network}.json -c ./configs/config_maisi_diff_model_${generate_version}.json -e ./configs/environment_maisi_diff_model_${generate_version}.json -g 1
```

To run the ControlNet model training script with MAISI Rectified flow for MRI, please run the code above with:

```bash
network="rflow"
generate_version="rflow-mr"
```

To run the ControlNet model training script with MAISI DDPM for CT, please run the code above with:

```bash
network="ddpm"
generate_version="ddpm-ct"
```

The training script also enables multi-GPU training. For instance, if you are using eight GPUs, you can run the training script with the following command:

```bash
export NUM_GPUS_PER_NODE=8
network="rflow"
generate_version="rflow-ct"
torchrun \
    --nproc_per_node=${NUM_GPUS_PER_NODE} \
    --nnodes=1 \
    --master_addr=localhost --master_port=1234 \
    -m scripts.train_controlnet -t ./configs/config_network_${network}.json -c ./configs/config_maisi_controlnet_train_${generate_version}.json -e ./configs/environment_maisi_controlnet_train_${generate_version}.json -g ${NUM_GPUS_PER_NODE}
```

Please also check [train_controlnet_tutorial.ipynb](../train_controlnet_tutorial.ipynb) for more details about data preparation and training parameters.

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
