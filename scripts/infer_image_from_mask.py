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

"""
Image-from-mask inference module.

Given a 3D body-region label mask (typically from
``sample_mask.ldm_conditional_sample_one_mask`` or a real training mask),
generates a CT/MR image using a ControlNet-conditioned image latent diffusion
model. See ``skills/image-from-mask.md`` for the algorithm walkthrough.

Re-imports ``ReconModel`` and ``initialize_noise_latents`` from
``sample_mask`` since both pipelines share these helpers.
"""

import gc
import logging
import time
import warnings

import torch
from monai.inferers.inferer import SlidingWindowInferer
from monai.networks.schedulers import DDPMScheduler, RFlowScheduler
from tqdm import tqdm

from .augmentation import remove_tumors
from .sample_mask import ReconModel, initialize_noise_latents
from .utils import binarize_labels, dynamic_infer


def ldm_conditional_sample_one_image(
    autoencoder,
    diffusion_unet,
    controlnet,
    noise_scheduler,
    scale_factor,
    device,
    combine_label_or,
    spacing_tensor,
    latent_shape,
    output_size,
    noise_factor,
    top_region_index_tensor=None,
    bottom_region_index_tensor=None,
    modality_tensor=None,
    num_inference_steps=1000,
    autoencoder_sliding_window_infer_size=[96, 96, 96],
    autoencoder_sliding_window_infer_overlap=0.6667,
    cfg_guidance_scale=0,
):
    """
    Generate a single synthetic image using a latent diffusion model with controlnet.

    Args:
        autoencoder (nn.Module): The autoencoder model.
        diffusion_unet (nn.Module): The diffusion U-Net model.
        controlnet (nn.Module): The controlnet model.
        noise_scheduler: The noise scheduler for the diffusion process.
        scale_factor (float): Scaling factor for the latent space.
        device (torch.device): The device to run the computation on.
        combine_label_or (torch.Tensor): The combined label tensor.
        spacing_tensor (torch.Tensor): Tensor specifying the spacing.
        latent_shape (tuple): The shape of the latent space.
        output_size (tuple): The desired output size of the image.
        noise_factor (float): Factor to scale the initial noise.
        top_region_index_tensor (torch.Tensor): Tensor specifying the top region index. Defaults to None.
        bottom_region_index_tensor (torch.Tensor): Tensor specifying the bottom region index. Defaults to None.
        modality_tensor (torch.Tensor): Int Tensor specifying the modality.
        num_inference_steps (int): Number of inference steps for the diffusion process.
        autoencoder_sliding_window_infer_size (list, optional): Size of the sliding window for inference. Defaults to [96, 96, 96].
        autoencoder_sliding_window_infer_overlap (float, optional): Overlap ratio for sliding window inference. Defaults to 0.6667.
        cfg_guidance_scale: float, classifier-free guidance

    Returns:
        tuple: A tuple containing the synthetic image and its corresponding label.
    """
    if modality_tensor <= 7:
        # CT image intensity range
        a_min = -1000
        a_max = 1000
        # autoencoder output intensity range
        b_min = 0.0
        b_max = 1
    else:
        # MRI image intensity range
        a_min = 0
        a_max = 1000
        # autoencoder output intensity range
        b_min = 0.0
        b_max = 1

    include_body_region = diffusion_unet.include_top_region_index_input
    include_modality = diffusion_unet.num_class_embeds is not None

    recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(device)

    with torch.no_grad(), torch.amp.autocast("cuda"):
        logging.info("---- Start generating latent features... ----")
        start_time = time.time()
        # generate segmentation mask
        combine_label = combine_label_or.to(device)
        if output_size[0] != combine_label.shape[2] or output_size[1] != combine_label.shape[3] or output_size[2] != combine_label.shape[4]:
            logging.info(
                "output_size is not a desired value. Need to interpolate the mask to match with output_size. The result image will be very low quality."
            )
            combine_label = torch.nn.functional.interpolate(combine_label, size=output_size, mode="nearest")

        controlnet_cond_vis = binarize_labels(combine_label.as_tensor().long()).half()

        # Generate random noise
        latents = initialize_noise_latents(latent_shape, device) * noise_factor

        # synthesize latents
        if isinstance(noise_scheduler, RFlowScheduler):
            noise_scheduler.set_timesteps(
                num_inference_steps=num_inference_steps,
                input_img_size_numel=torch.prod(torch.tensor(latents.shape[-3:])),
            )
        else:
            noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps)

        if isinstance(noise_scheduler, DDPMScheduler) and num_inference_steps < noise_scheduler.num_train_timesteps:
            warnings.warn(
                "**************************************************************\n"
                "* WARNING: Image noise_scheduler is a DDPMScheduler.\n"
                "* We expect num_inference_steps = noise_scheduler.num_train_timesteps"
                f" = {noise_scheduler.num_train_timesteps}.\n"
                f"* Yet got num_inference_steps = {num_inference_steps}.\n"
                "* The generated image quality is not guaranteed.\n"
                "**************************************************************"
            )

        all_timesteps = noise_scheduler.timesteps
        all_next_timesteps = torch.cat((all_timesteps[1:], torch.tensor([0], dtype=all_timesteps.dtype)))
        progress_bar = tqdm(
            zip(all_timesteps, all_next_timesteps),
            total=min(len(all_timesteps), len(all_next_timesteps)),
        )
        if cfg_guidance_scale > 0:
            combine_label_no_tumor = torch.nn.functional.interpolate(
                remove_tumors(combine_label.squeeze(0)).unsqueeze(0).float(), size=output_size, mode="nearest"
            ).to(combine_label.dtype)
            controlnet_cond_vis_no_tumor = binarize_labels(combine_label_no_tumor.as_tensor().long()).half()
            del combine_label_no_tumor
        for t, next_t in progress_bar:
            # get controlnet output
            # Create a dictionary to store the inputs
            controlnet_inputs = {
                "x": latents,
                "timesteps": torch.Tensor((t,)).to(device),
                "controlnet_cond": controlnet_cond_vis,
            }
            if include_modality:
                controlnet_inputs.update(
                    {
                        "class_labels": modality_tensor,
                    }
                )
            if cfg_guidance_scale > 0:
                for k in controlnet_inputs.keys():
                    if k == "class_labels":
                        controlnet_inputs[k] = torch.cat([modality_tensor, torch.zeros_like(modality_tensor)])
                    elif k == "controlnet_cond":
                        controlnet_inputs[k] = torch.cat([controlnet_cond_vis, controlnet_cond_vis_no_tumor])
                    else:
                        controlnet_inputs[k] = torch.cat([controlnet_inputs[k]] * 2)

            down_block_res_samples, mid_block_res_sample = controlnet(**controlnet_inputs)

            # get diffusion network output
            # Create a dictionary to store the inputs
            unet_inputs = {
                "x": latents,
                "timesteps": torch.Tensor((t,)).to(device),
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
            if cfg_guidance_scale > 0:
                for k in unet_inputs.keys():
                    if k in ["down_block_additional_residuals", "mid_block_additional_residual"]:
                        pass
                    elif k != "class_labels":
                        unet_inputs[k] = torch.cat([unet_inputs[k]] * 2)
                    else:
                        unet_inputs[k] = torch.cat([unet_inputs[k], torch.zeros_like(modality_tensor)])
            if cfg_guidance_scale == 0:
                model_output = diffusion_unet(**unet_inputs)
            else:
                model_t, model_uncond = diffusion_unet(**unet_inputs).chunk(2)
                model_output = model_uncond + cfg_guidance_scale * (model_t - model_uncond)

            if not isinstance(noise_scheduler, RFlowScheduler):
                latents, _ = noise_scheduler.step(model_output, t, latents)  # type: ignore
            else:
                latents, _ = noise_scheduler.step(model_output, t, latents, next_t)  # type: ignore
        end_time = time.time()
        logging.info(f"---- DM/ControlNet Latent features generation time: {end_time - start_time} seconds ----")
        del (
            unet_inputs,
            controlnet_inputs,
            model_output,
            controlnet_cond_vis,
            down_block_res_samples,
            mid_block_res_sample,
        )
        gc.collect()
        torch.cuda.empty_cache()

        # decode latents to synthesized images
        logging.info("---- Start decoding latent features into images... ----")
        start_time = time.time()

        inferer = SlidingWindowInferer(
            roi_size=autoencoder_sliding_window_infer_size,
            sw_batch_size=1,
            progress=True,
            mode="gaussian",
            overlap=autoencoder_sliding_window_infer_overlap,
            sw_device=device,
            device=torch.device("cpu"),
        )
        synthetic_images = dynamic_infer(inferer, recon_model, latents)
        if modality_tensor <= 7:
            synthetic_images = torch.clip(synthetic_images, b_min, b_max).cpu()
        else:
            synthetic_images = torch.clip(synthetic_images, b_min, None).cpu()
        end_time = time.time()
        logging.info(f"---- Image VAE decoding time: {end_time - start_time} seconds ----")

        ## post processing:
        # project output to [0, 1]
        synthetic_images = (synthetic_images - b_min) / (b_max - b_min)
        # project output to [-1000, 1000]
        synthetic_images = synthetic_images * (a_max - a_min) + a_min
        # regularize background intensities
        synthetic_images = crop_img_body_mask(synthetic_images, combine_label, a_min=a_min)
        torch.cuda.empty_cache()

    return synthetic_images, combine_label


def crop_img_body_mask(synthetic_images, combine_label, a_min=-1000):
    """
    Crop the synthetic image using a body mask.

    Args:
        synthetic_images (torch.Tensor): The synthetic images.
        combine_label (torch.Tensor): The body mask.

    Returns:
        torch.Tensor: The cropped synthetic images.
    """
    synthetic_images[combine_label == 0] = a_min
    return synthetic_images
