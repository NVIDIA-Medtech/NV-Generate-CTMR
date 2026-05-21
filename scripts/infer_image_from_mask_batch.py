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
from datetime import datetime

import torch
import torch.distributed as dist
from monai.data import MetaTensor, decollate_batch
from monai.transforms import SaveImage
from monai.utils import RankFilter

from .diff_model_setting import load_config
from .infer_image_from_mask import ldm_conditional_sample_one_image
from .utils import prepare_maisi_controlnet_json_dataloader, setup_ddp
from .utils_infer import load_image_models


@torch.inference_mode()
def infer_image_from_mask_batch(env_config_path: str, model_config_path: str, model_def_path: str, num_gpus: int) -> None:
    """
    Batch image-from-mask inference driven by a JSON manifest.

    Reads ``args.json_data_list`` (the manifest), which must list cases each
    with at least:

        - ``label``: path to a NIfTI mask in the MAISI 132-class vocabulary
        - ``dim``: ``[H, W, D]`` target output size
        - ``spacing``: ``[sx, sy, sz]`` (mm)
        - ``top_region_index``, ``bottom_region_index``: one-hot 4-lists
          (required only when the image DM was trained with
          ``include_top_region_index_input=True`` — i.e. ``ddpm-ct``)

    For each batch, calls :func:`ldm_conditional_sample_one_image` and saves
    the paired image + label as NIfTI under ``args.output_dir``.

    For single-mask inference, use ``python -m scripts.infer_image_from_mask
    --mask <file>`` instead (no manifest needed).
    """
    # Step 0: configuration
    logger = logging.getLogger("maisi.image_from_mask_batch.infer")
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

    # Step 2: load image-side networks (AE + DM + ControlNet) via the shared helper
    autoencoder, unet, controlnet, scale_factor, noise_scheduler = load_image_models(args, device)
    include_body_region = unet.include_top_region_index_input
    include_modality = unet.num_class_embeds is not None
    logger.info("Loaded image AE + DM + ControlNet via utils_infer.load_image_models.")

    # set data loader
    if include_modality:
        if args.modality_mapping_path is not None:
            if not os.path.exists(args.modality_mapping_path):
                raise ValueError(f"Please check if {args.modality_mapping_path} exist.")
        else:
            raise ValueError(f"'modality_mapping_path' in {env_config_path} cannot be null")
        with open(args.modality_mapping_path) as f:
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
        modality_mapping=args.modality_mapping,
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
    parser = argparse.ArgumentParser(
        prog="python -m scripts.infer_image_from_mask_batch",
        description=(
            "Batch image-from-mask inference: read a JSON manifest of mask files and "
            "generate a paired CT/MR image for each one via the ControlNet-conditioned "
            "image LDM. For single-mask inference use scripts.infer_image_from_mask."
        ),
    )
    parser.add_argument(
        "-t",
        "--config-file",
        required=True,
        help="Config json file that stores network hyper-parameters (e.g. ./configs/config_network_rflow.json).",
    )
    parser.add_argument(
        "-e",
        "--environment-file",
        required=True,
        help="Environment json file that stores environment paths (e.g. ./configs/environment_rflow-ct.json).",
    )
    parser.add_argument(
        "-i",
        "--inference-file",
        required=True,
        help="Config json file that stores inference hyper-parameters (e.g. ./configs/config_maisi_diff_model_rflow-ct.json).",
    )
    parser.add_argument("-g", "--num-gpus", type=int, default=1, help="Number of GPUs to use")

    args = parser.parse_args()
    infer_image_from_mask_batch(args.environment_file, args.inference_file, args.config_file, args.num_gpus)
