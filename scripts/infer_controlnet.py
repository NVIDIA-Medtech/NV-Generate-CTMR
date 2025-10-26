# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import logging
import os
import sys
from datetime import datetime

import torch
import torch.distributed as dist
import monai
from monai.data import MetaTensor, decollate_batch
from monai.networks.utils import copy_model_state
from monai.transforms import SaveImage
from monai.utils import RankFilter

from .sample import ldm_conditional_sample_one_image
from .utils import define_instance, prepare_maisi_controlnet_json_dataloader, setup_ddp
from .diff_model_setting import load_config


@torch.inference_mode()
def infer_controlnet(
    env_config_path: str, model_config_path: str, model_def_path: str, num_gpus: int
) -> None:

    # Step 0: configuration
    logger = logging.getLogger("maisi.controlnet.infer")
    # whether to use distributed data parallel
    use_ddp = num_gpus > 1
    if use_ddp:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = setup_ddp(rank, world_size)
        logger.addFilter(RankFilter())
    else:
        rank = 0
        world_size = 1
        device = torch.device(f"cuda:{rank}")

    torch.cuda.set_device(device)
    logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
    logger.info(f"World_size: {world_size}")

    args = load_config(env_config_path, model_config_path, model_def_path)

    

    # Step 2: define AE, diffusion model and controlnet
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    checkpoint_autoencoder = torch.load(args.trained_autoencoder_path)
    if "unet_state_dict" in checkpoint_autoencoder.keys():
        checkpoint_autoencoder = checkpoint_autoencoder["unet_state_dict"]
    autoencoder.load_state_dict(checkpoint_autoencoder)
    logger.info(f"Load trained VAE model from {args.trained_autoencoder_path}.")

    unet = define_instance(args, "diffusion_unet_def").to(device)
    checkpoint_diffusion_unet = torch.load(args.trained_diffusion_path, weights_only=False)
    unet.load_state_dict(checkpoint_diffusion_unet["unet_state_dict"], strict=False)
    scale_factor = checkpoint_diffusion_unet["scale_factor"].to(device)
    include_body_region = unet.include_top_region_index_input
    include_modality = unet.num_class_embeds is not None
    logger.info(f"Load trained diffusion model from {args.trained_diffusion_path}.")

    controlnet = define_instance(args, "controlnet_def").to(device)
    checkpoint_controlnet = torch.load(args.trained_controlnet_path, weights_only=False)
    monai.networks.utils.copy_model_state(controlnet, unet.state_dict())
    controlnet.load_state_dict(checkpoint_controlnet["controlnet_state_dict"], strict=False)
    logger.info(f"Load trained controlnet model from {args.trained_controlnet_path}.")

    noise_scheduler = define_instance(args, "noise_scheduler")

    # set data loader
    if include_modality:
        if args.modality_mapping_path is not None:
            if not os.path.exists(args.modality_mapping_path):
                raise ValueError(f"Please check if {args.modality_mapping_path} exist.")
        else:
            raise ValueError(f"'modality_mapping_path' in {env_config_path} cannot be null")
        with open(args.modality_mapping_path, "r") as f:
            args.modality_mapping = json.load(f)
    else:
        args.modality_mapping = None

    # Step 1: set data loader
    _, val_loader = prepare_maisi_controlnet_json_dataloader(
        json_data_list=args.json_data_list,
        data_base_dir=args.data_base_dir,
        rank=rank,
        world_size=world_size,
        batch_size=args.controlnet_train["batch_size"],
        cache_rate=args.controlnet_train["cache_rate"],
        fold=args.controlnet_train["fold"],
        modality_mapping=args.modality_mapping
    )

    # Step 3: inference
    autoencoder.eval()
    controlnet.eval()
    unet.eval()

    for batch in val_loader:
        # get label mask
        labels = batch["label"].to(device)
        # get corresponding conditions
        if include_body_region:
            top_region_index_tensor = batch["top_region_index"].to(device)
            bottom_region_index_tensor = batch["bottom_region_index"].to(device)
        else:
            top_region_index_tensor = None
            bottom_region_index_tensor = None
        spacing_tensor = batch["spacing"].to(device)
        modality_tensor = args.controlnet_infer["modality"] * torch.ones((len(labels),), dtype=torch.long).to(device)
        out_spacing = tuple((batch["spacing"].squeeze().numpy() / 100).tolist())
        # get target dimension
        dim = batch["dim"]
        output_size = (dim[0].item(), dim[1].item(), dim[2].item())
        latent_shape = (args.latent_channels, output_size[0] // 4, output_size[1] // 4, output_size[2] // 4)

        # generate a single synthetic image using a latent diffusion model with controlnet.
        synthetic_images, _ = ldm_conditional_sample_one_image(
            autoencoder=autoencoder,
            diffusion_unet=unet,
            controlnet=controlnet,
            noise_scheduler=noise_scheduler,
            scale_factor=scale_factor,
            device=device,
            combine_label_or=labels,
            top_region_index_tensor=top_region_index_tensor,
            bottom_region_index_tensor=bottom_region_index_tensor,
            spacing_tensor=spacing_tensor,
            modality_tensor=modality_tensor,
            latent_shape=latent_shape,
            output_size=output_size,
            noise_factor=1.0,
            num_inference_steps=args.controlnet_infer["num_inference_steps"],
            autoencoder_sliding_window_infer_size=args.controlnet_infer["autoencoder_sliding_window_infer_size"],
            autoencoder_sliding_window_infer_overlap=args.controlnet_infer["autoencoder_sliding_window_infer_overlap"],
        )
        # save image/label pairs
        labels = decollate_batch(batch)[0]["label"]
        output_postfix = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        labels.meta["filename_or_obj"] = "sample.nii.gz"
        synthetic_images = MetaTensor(synthetic_images.squeeze(0), meta=labels.meta)
        img_saver = SaveImage(
            output_dir=args.output_dir,
            output_postfix=output_postfix + "_image",
            separate_folder=False,
        )
        img_saver(synthetic_images)
        label_saver = SaveImage(
            output_dir=args.output_dir,
            output_postfix=output_postfix + "_label",
            separate_folder=False,
        )
        label_saver(labels)
    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ControlNet Model Training")
    parser.add_argument(
        "--env_config_path",
        type=str,
        default="./configs/environment_maisi_diff_model.json",
        help="Path to environment configuration file",
    )
    parser.add_argument(
        "--model_config_path",
        type=str,
        default="./configs/config_maisi_diff_model.json",
        help="Path to model training/inference configuration",
    )
    parser.add_argument(
        "--model_def_path", type=str, default="./configs/config_maisi.json", help="Path to model definition file"
    )
    parser.add_argument(
        "-g",
        "--num_gpus", 
        type=int, 
        default=1, 
        help="Number of GPUs to use for training"
    )

    args = parser.parse_args()
    infer_controlnet(args.env_config_path, args.model_config_path, args.model_def_path, args.num_gpus)
