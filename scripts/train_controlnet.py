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
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from monai.networks.utils import copy_model_state
from monai.utils import RankFilter
from monai.transforms.utils_morphological_ops import dilate
from monai.networks.schedulers import RFlowScheduler
from monai.networks.schedulers.ddpm import DDPMPredictionType
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from .utils import binarize_labels, define_instance, prepare_maisi_controlnet_json_dataloader, setup_ddp
from .diff_model_setting import initialize_distributed, load_config, setup_logging
from .augmentation import remove_tumors

def remove_roi(labels):
    """
    Remove ROI voxels from a label tensor. 
    Users need to define their own function of remove_roi.
    Here we use scripts.augmentation.remove_tumors as default

    Args:
        labels (torch.Tensor): Segmentation tensor. Shape is
            [B, 1, X, Y, Z]. Dtype is usually integer/long.

    Returns:
        torch.Tensor: Labels with ROI content removed. Same shape and
        device as `labels`.
    """
    labels_roi_free = []
    for b in range(labels.shape[0]):
        labels_roi_free_b = remove_tumors(labels[b,...])
        labels_roi_free.append(labels_roi_free_b)
    labels_roi_free = torch.cat(labels_roi_free,dim=0)
    return labels_roi_free
    
def compute_region_contrasive_loss(
    model_output,model_output_roi_free,model_gt,
    roi_contrastive,roi_contrastive_bg,
    max_region_contrasive_loss=2,
    loss_contrastive = torch.nn.L1Loss(reduction = 'none')
):
    """
    Compute region-wise contrastive losses between the model output with and
    without ROIs, promoting differences inside ROI and similarity outside ROI.

    The loss has two parts:
      1) `loss_region_contrasive`: encourages the model output to differ from
         its ROI-free counterpart *inside* the ROI (foreground). Implemented as
         a (negative) masked L1 reduced by the foreground voxel count and then
         clipped by a ReLU window around `max_region_contrasive_loss`.
      2) `loss_region_bg`: encourages *similarity* in the background
         (outside ROI) between the ROI-free output and the original output,
         implemented as masked L1 reduced by background voxel count.

    Args:
        model_output (torch.Tensor):
            Network output with ROI present. Shape [B, C, X, Y, Z].
        model_output_roi_free (torch.Tensor):
            Network output for ROI-removed labels (same shape/device).
        roi_contrastive (torch.Tensor):
            Foreground ROI mask (1 inside ROI, 0 outside). Can be bool or
            integer; will be resized to `model_output.shape[2:]` using
            nearest-neighbor and multiplied as weights.
            Expected shape broadcastable to [B, C, X, Y, Z].
        roi_contrastive_bg (torch.Tensor):
            Background mask (1 outside ROI, 0 inside). Will be resized to
            `inputs.shape[2:]` (see Notes) and repeated over channels to match
            [B, C, X, Y, Z].
        max_region_contrasive_loss (float, optional):
            Upper-window parameter used to bound the foreground loss via
            `relu(loss + max) - max`. Defaults to 2.
        loss_contrastive (torch.nn.modules.loss._Loss, optional):
            Elementwise regression loss with `reduction='none'` (e.g., L1).
            Defaults to `torch.nn.L1Loss(reduction='none')`.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - loss_region_contrasive (scalar tensor): foreground contrastive loss.
            - loss_region_bg (scalar tensor): background similarity loss.
    """
    if roi_contrastive.shape[1]!=1 or roi_contrastive_bg.shape[1]!=1:
        raise ValueError(f"Assert roi_contrastive.shape[1]==1 or roi_contrastive_bg.shape[1]==1, yet got {roi_contrastive.shape} and {roi_contrastive_bg.shape}.")
        
    roi_contrastive = F.interpolate(roi_contrastive, size=model_output.shape[2:], mode="nearest")  
    roi_contrastive = roi_contrastive.repeat(1, model_output.shape[1], 1, 1, 1)    
    loss_region_contrasive = -(loss_contrastive(model_output, model_output_roi_free)*roi_contrastive).sum()/(torch.sum(roi_contrastive>0)+1e-5)
    loss_region_contrasive = F.relu(loss_region_contrasive+max_region_contrasive_loss)-max_region_contrasive_loss # we do not need it to be extreme           
    
    roi_contrastive_bg = F.interpolate(roi_contrastive_bg, size=model_output.shape[2:], mode="nearest").to(torch.long)
    roi_contrastive_bg = roi_contrastive_bg.repeat(1, model_output.shape[1], 1, 1, 1)    
    loss_region_bg = (loss_contrastive(model_output_roi_free, model_gt)*roi_contrastive_bg).sum()/(torch.sum(roi_contrastive_bg>0)+1e-5)
    return loss_region_contrasive, loss_region_bg

def compute_model_output(
    images,labels,noise,timesteps,noise_scheduler,
    controlnet,unet,
    spacing_tensor,
    modality_tensor=None,
    top_region_index_tensor=None,
    bottom_region_index_tensor=None,
    return_controlnet_blocks=False
):
    """
    Run ControlNet + U-Net to obtain the denoising network output (and optionally
    the ControlNet intermediate blocks) for a given noisy latent and conditions.

    Pipeline:
      1) Binarize labels to build ControlNet condition.
      2) Add noise to `images` at `timesteps` via the scheduler.
      3) Pass noisy latent and conditions to ControlNet to get down/mid features.
      4) Pass everything to U-Net (with spacing, optional modality & body-region
         tokens) to produce `model_output`.

    Args:
        images (torch.Tensor):
            Input latent/image tensor to be noised. Shape [B, C, X, Y, Z].
        labels (torch.Tensor or monai.data.MetaTensor):
            Segmentation labels used to create ControlNet condition.
        noise (torch.Tensor):
            Noise tensor aligned with `images`.
        timesteps (torch.Tensor or Any):
            Diffusion timesteps for the scheduler and networks.
        noise_scheduler:
            Object exposing `add_noise(original_samples, noise, timesteps)`.
        controlnet (torch.nn.Module):
            Control network returning `(down_block_res_samples, mid_block_res_sample)`.
        unet (torch.nn.Module):
            Denoising network that accepts additional residuals from ControlNet.
        spacing_tensor (torch.Tensor):
            Per-sample spacing or resolution encoding; passed into U-Net.
        modality_tensor (torch.Tensor, optional):
            Class labels or modality codes for conditional generation (e.g., MRI/CT).
        top_region_index_tensor (torch.Tensor, optional):
            Region index tensor (top bound) for body-region-aware conditioning.
        bottom_region_index_tensor (torch.Tensor, optional):
            Region index tensor (bottom bound) for body-region-aware conditioning.
        return_controlnet_blocks (bool, optional):
            If True, also return `(down_block_res_samples, mid_block_res_sample)`.
            Defaults to False.

    Returns:
        Tuple[torch.Tensor, Optional[Any], Optional[Any]]:
            - model_output (torch.Tensor): U-Net output with shape [B, C, X, Y, Z].
            - down_block_res_samples (optional): ControlNet down-block features if requested, else None.
            - mid_block_res_sample (optional): ControlNet mid-block feature if requested, else None.
    """
    # generate random noise
    include_modality = ( modality_tensor is not None )
    include_body_region = ( top_region_index_tensor is not None) and (bottom_region_index_tensor is not None)

    # use binary encoding to encode segmentation mask
    controlnet_cond = binarize_labels(labels.as_tensor().to(torch.long)).float()

    # create noisy latent
    noisy_latent = noise_scheduler.add_noise(original_samples=images, noise=noise, timesteps=timesteps)

    # get controlnet output
    # Create a dictionary to store the inputs
    controlnet_inputs = {
        "x": noisy_latent,
        "timesteps": timesteps,
        "controlnet_cond": controlnet_cond,
    }
    if include_modality:
        controlnet_inputs.update(
            {
                "class_labels": modality_tensor,
            }
        )
    down_block_res_samples, mid_block_res_sample = controlnet(**controlnet_inputs)

    # get diffusion network output
    # Create a dictionary to store the inputs
    unet_inputs = {
        "x": noisy_latent,
        "timesteps": timesteps,
        "spacing_tensor": spacing_tensor,
        "down_block_additional_residuals": down_block_res_samples,
        "mid_block_additional_residual": mid_block_res_sample,
    }
    # Add extra arguments if include_body_region is True
    if include_body_region:
        unet_inputs.update(
            {
                "top_region_index_tensor": top_region_index_tensor,
                "bottom_region_index_tensor": bottom_region_index_tensor,
            }
        )
    if include_modality:
        unet_inputs.update(
            {
                "class_labels": modality_tensor,
            }
        )
    model_output = unet(**unet_inputs)
    if return_controlnet_blocks:
        return model_output, down_block_res_samples, mid_block_res_sample
    else:
        return model_output, None, None

def train_controlnet(
    env_config_path: str, model_config_path: str, model_def_path: str, num_gpus: int
) -> None:
    # Step 0: configuration
    logger = logging.getLogger("maisi.controlnet.training")
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
    if "use_region_contrasive_loss" not in args.controlnet_train.keys():
        args.use_region_contrasive_loss = False
    else:
        args.use_region_contrasive_loss = args.controlnet_train["use_region_contrasive_loss"]
        for k in ["region_contrasive_loss_delta", "region_contrasive_loss_weight"]:
            if k not in args.controlnet_train.keys():
                raise ValueError(f"Since 'use_region_contrasive_loss' is in 'controlnet_train' of {model_config_path}, we need 'region_contrasive_loss_delta' and 'region_contrasive_loss_weight' also be in it.")
    
    logger.info(f"use_region_contrasive_loss: {args.use_region_contrasive_loss}")
    if args.use_region_contrasive_loss:
        logger.warning(f"User sets 'use_region_contrasive_loss' as true in {model_config_path}.")
        logger.warning("********************")
        logger.warning(
            "Please check remove_roi() in train_controlnet.py to ensure ROI is removed as intended; "
            "default logic will not match your requirement."
        )
        logger.warning("********************")


    # initialize tensorboard writer
    if rank == 0:
        tensorboard_path = os.path.join(args.tfevent_path, args.exp_name)
        Path(tensorboard_path).mkdir(parents=True, exist_ok=True)
        tensorboard_writer = SummaryWriter(tensorboard_path)


    # Step 2: define diffusion model and controlnet
    # define diffusion Model
    unet = define_instance(args, "diffusion_unet_def").to(device)
    include_body_region = unet.include_top_region_index_input
    include_modality = unet.num_class_embeds is not None

    # load trained diffusion model
    if args.trained_diffusion_path is not None:
        if not os.path.exists(args.trained_diffusion_path):
            raise ValueError(f"Please download the trained diffusion unet checkpoint to {args.trained_diffusion_path}.")
        diffusion_model_ckpt = torch.load(args.trained_diffusion_path, map_location=device, weights_only=False)
        unet.load_state_dict(diffusion_model_ckpt["unet_state_dict"], strict=False)
        # load scale factor from diffusion model checkpoint
        scale_factor = diffusion_model_ckpt["scale_factor"]
        logger.info(f"Load trained diffusion model from {args.trained_diffusion_path}.")
        logger.info(f"loaded scale_factor from diffusion model ckpt -> {scale_factor}.")
    else:
        raise ValueError(f"'trained_diffusion_path' in {env_config_path} cannot be null.")

    # define ControlNet
    controlnet = define_instance(args, "controlnet_def").to(device)
    # copy weights from the DM to the controlnet
    copy_model_state(controlnet, unet.state_dict())
    # load trained controlnet model if it is provided
    if args.existing_ckpt_filepath is not None:
        if not os.path.exists(args.existing_ckpt_filepath):
            raise ValueError("Please download the trained ControlNet checkpoint.")
        controlnet.load_state_dict(
            torch.load(args.existing_ckpt_filepath, map_location=device, weights_only=False)["controlnet_state_dict"]
        )
        logger.info(f"load trained controlnet model from {args.existing_ckpt_filepath}")
    else:
        logger.info("train controlnet model from scratch.")
    # we freeze the parameters of the diffusion model.
    for p in unet.parameters():
        p.requires_grad = False

    noise_scheduler = define_instance(args, "noise_scheduler")

    if use_ddp:
        controlnet = DDP(controlnet, device_ids=[device], output_device=rank, find_unused_parameters=True)

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

    train_loader, _ = prepare_maisi_controlnet_json_dataloader(
        json_data_list=args.json_data_list,
        data_base_dir=args.data_base_dir,
        rank=rank,
        world_size=world_size,
        batch_size=args.controlnet_train["batch_size"],
        cache_rate=args.controlnet_train["cache_rate"],
        fold=args.controlnet_train["fold"],
        modality_mapping = args.modality_mapping
    )

    # Step 3: training config
    weighted_loss = args.controlnet_train["weighted_loss"]
    weighted_loss_label = args.controlnet_train["weighted_loss_label"]
    optimizer = torch.optim.AdamW(params=controlnet.parameters(), lr=args.controlnet_train["lr"])
    total_steps = (args.controlnet_train["n_epochs"] * len(train_loader.dataset)) / args.controlnet_train["batch_size"]
    logger.info(f"total number of training steps: {total_steps}.")

    lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_steps, power=2.0)

    # Step 4: training
    n_epochs = args.controlnet_train["n_epochs"]
    scaler = GradScaler("cuda")
    total_step = 0
    best_loss = 1e4

    if weighted_loss > 1.0:
        logger.info(f"apply weighted loss = {weighted_loss} on labels: {weighted_loss_label}")

    controlnet.train()
    unet.eval()
    prev_time = time.time()
    for epoch in range(n_epochs):
        epoch_loss_ = 0
        for step, batch in enumerate(train_loader):
            # get image embedding and label mask and scale image embedding by the provided scale_factor
            images = batch["image"].to(device) * scale_factor
            labels = batch["label"].to(device)
            if labels.shape[1] != 1:
                raise ValueError(f"We expect labels with shape [B,1,X,Y,Z], yet got {labels.shape}")
            # get corresponding conditions
            spacing_tensor = batch["spacing"].to(device)
            top_region_index_tensor = None
            bottom_region_index_tensor = None
            modality_tensor = None
            if include_body_region:
                top_region_index_tensor = batch["top_region_index"].to(device)
                bottom_region_index_tensor = batch["bottom_region_index"].to(device)
            # We trained with only CT in this version
            if include_modality:
                modality_tensor = batch["modality"].to(device)

            optimizer.zero_grad(set_to_none=True)

            if args.use_region_contrasive_loss:
                labels_roi_free = remove_roi(labels)

            with autocast("cuda", enabled=True):
                # randomly sample noise
                noise_shape = list(images.shape)
                noise = torch.randn(noise_shape, dtype=images.dtype).to(device)
                # randomly sample timesteps
                if isinstance(noise_scheduler, RFlowScheduler):
                    timesteps = noise_scheduler.sample_timesteps(images)
                else:
                    timesteps = torch.randint(
                        0, noise_scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                    ).long()
                (
                    model_output, 
                    model_block1_output, 
                    model_block2_output
                ) = compute_model_output(
                    images,labels,noise,timesteps,noise_scheduler,
                    controlnet,unet,
                    spacing_tensor,
                    modality_tensor,
                    top_region_index_tensor,
                    bottom_region_index_tensor,
                    return_controlnet_blocks=False
                )
                if args.use_region_contrasive_loss:
                    (
                        model_output_roi_free,
                        model_block1_output_roi_free,
                        model_block2_output_roi_free,
                    ) = compute_model_output(
                        images,labels_roi_free,noise,timesteps,noise_scheduler,
                        controlnet,unet,
                        spacing_tensor,
                        modality_tensor,
                        top_region_index_tensor,
                        bottom_region_index_tensor,
                        return_controlnet_blocks=False
                    )
                

                if noise_scheduler.prediction_type == DDPMPredictionType.EPSILON:
                    # predict noise
                    model_gt = noise
                elif noise_scheduler.prediction_type == DDPMPredictionType.SAMPLE:
                    # predict sample
                    model_gt = images
                elif noise_scheduler.prediction_type == DDPMPredictionType.V_PREDICTION:
                    # predict velocity
                    model_gt = images - noise
                else:
                    raise ValueError(
                        "noise scheduler prediction type has to be chosen from ",
                        f"[{DDPMPredictionType.EPSILON},{DDPMPredictionType.SAMPLE},{DDPMPredictionType.V_PREDICTION}]",
                    )
    
                if weighted_loss > 1.0:
                    weights = torch.ones_like(images).to(images.device)
                    roi = torch.zeros([noise_shape[0]] + [1] + noise_shape[2:]).to(images.device)
                    interpolate_label = F.interpolate(labels, size=images.shape[2:], mode="nearest")
                    # assign larger weights for ROI (tumor)
                    for label in weighted_loss_label:
                        roi[interpolate_label == label] = 1
                    weights[roi.repeat(1, images.shape[1], 1, 1, 1) == 1] = weighted_loss
                    loss = (F.l1_loss(model_output.float(), model_gt.float(), reduction="none") * weights).mean()
                else:
                    loss = F.l1_loss(model_output.float(), model_gt.float())
    
    
                if args.use_region_contrasive_loss:
                    roi_contrastive = (labels_roi_free != labels).to(torch.uint8)  # 0/1 mask
                    roi_contrastive_bg = 1 - dilate(roi_contrastive, filter_size=3).to(torch.uint8)
                    loss_region_contrasive, loss_region_bg = compute_region_contrasive_loss(
                        model_output,model_output_roi_free,model_gt,
                        roi_contrastive,roi_contrastive_bg,
                        max_region_contrasive_loss=args.controlnet_train["region_contrasive_loss_delta"],
                        loss_contrastive = torch.nn.L1Loss(reduction = 'none')
                    )
                    final_loss_region_contrasive = loss_region_contrasive + loss_region_bg
                    logger.info(f"loss_region_contrasive: {loss_region_contrasive}")
                    logger.info(f"loss_region_bg: {loss_region_bg}")
                    loss += args.controlnet_train["region_contrasive_loss_weight"]*final_loss_region_contrasive

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            total_step += 1

            if rank == 0:
                # write train loss for each batch into tensorboard
                tensorboard_writer.add_scalar(
                    "train/train_controlnet_loss_iter", loss.detach().cpu().item(), total_step
                )
                batches_done = step + 1
                batches_left = len(train_loader) - batches_done
                time_left = timedelta(seconds=batches_left * (time.time() - prev_time))
                prev_time = time.time()
                logger.info(
                    "\r[Epoch %d/%d] [Batch %d/%d] [LR: %.8f] [loss: %.4f] ETA: %s "
                    % (
                        epoch + 1,
                        n_epochs,
                        step + 1,
                        len(train_loader),
                        lr_scheduler.get_last_lr()[0],
                        loss.detach().cpu().item(),
                        time_left,
                    )
                )
            epoch_loss_ += loss.detach()

        epoch_loss = epoch_loss_ / (step + 1)

        if use_ddp:
            dist.barrier()
            dist.all_reduce(epoch_loss, op=torch.distributed.ReduceOp.AVG)

        if rank == 0:
            tensorboard_writer.add_scalar("train/train_controlnet_loss_epoch", epoch_loss.cpu().item(), total_step)
            # save controlnet only on master GPU (rank 0)
            controlnet_state_dict = controlnet.module.state_dict() if world_size > 1 else controlnet.state_dict()
            torch.save(
                {
                    "epoch": epoch + 1,
                    "loss": epoch_loss,
                    "controlnet_state_dict": controlnet_state_dict,
                },
                f"{args.model_dir}/{args.exp_name}_current.pt",
            )
            logger.info(f"Save trained model to {args.model_dir}/{args.exp_name}_current.pt")

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                logger.info(f"best loss -> {best_loss}.")
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "loss": best_loss,
                        "controlnet_state_dict": controlnet_state_dict,
                    },
                    f"{args.model_dir}/{args.exp_name}_best.pt",
                )
                logger.info(f"Save trained model to {args.model_dir}/{args.exp_name}_best.pt")

        torch.cuda.empty_cache()
    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ControlNet Model Training")
    parser.add_argument(
        "-e",
        "--env_config_path",
        type=str,
        default="./configs/environment_maisi_diff_model.json",
        help="Path to environment configuration file",
    )
    parser.add_argument(
        "-c",
        "--model_config_path",
        type=str,
        default="./configs/config_maisi_diff_model.json",
        help="Path to model training/inference configuration",
    )
    parser.add_argument(
        "-t",
        "--model_def_path", 
        type=str, 
        default="./configs/config_maisi.json", 
        help="Path to model definition file"
    )
    parser.add_argument(
        "-g",
        "--num_gpus", 
        type=int, 
        default=1, 
        help="Number of GPUs to use for training"
    )

    args = parser.parse_args()
    train_controlnet(args.env_config_path, args.model_config_path, args.model_def_path, args.num_gpus)

