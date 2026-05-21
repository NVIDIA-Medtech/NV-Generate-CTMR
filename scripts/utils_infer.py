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
Shared inference helpers reused across:

- ``scripts/sample_mask.py``           — mask DDPM (anatomy_size → mask)
- ``scripts/infer_image_from_mask.py`` — mask → CT/MR via ControlNet image DM

The core function ``run_controlnet_conditioned_image_dm`` is conditioning-
modality-agnostic (it takes a pre-prepared ControlNet conditioning tensor),
so future ControlNet variants trained on other conditioning modalities can
add their own wrapper module and reuse this core.

What lives here:

- ``ReconModel``                          — wraps an autoencoder for scale-corrected decode
- ``initialize_noise_latents``            — fp16 random-noise latent generator
- ``run_controlnet_conditioned_image_dm`` — modality-agnostic core: timestep loop +
                                            ControlNet + image DM + sliding-window AE decode +
                                            HU range mapping. Caller pre-prepares the
                                            ControlNet conditioning tensor.
- ``load_image_models``                   — image AE + image DM + ControlNet + scheduler
- ``load_mask_models``                    — mask AE + mask DM + mask scheduler
- ``load_paired_inference_models``        — convenience: both bundles for LDMSampler
- ``build_conditioning_tensors``          — packs spacing / region / modality tensors

What is NOT here (lives elsewhere):

- Mask-specific helpers (``binarize_labels``, ``remap_labels``,
  ``general_mask_generation_post_process``) → ``scripts/utils.py``
- Mask wrapper + CLI                      → ``scripts/infer_image_from_mask.py``
"""

from __future__ import annotations

import gc
import logging
import time
import warnings

import monai
import torch
from monai.inferers.inferer import SlidingWindowInferer
from monai.networks.schedulers import DDPMScheduler, RFlowScheduler
from tqdm import tqdm

from .utils import dynamic_infer, get_body_region_index_from_mask


class ReconModel(torch.nn.Module):
    """
    A PyTorch module for reconstructing images from latent representations.

    Attributes:
        autoencoder: The autoencoder model used for decoding.
        scale_factor: Scaling factor applied to the input before decoding.
    """

    def __init__(self, autoencoder, scale_factor):
        super().__init__()
        self.autoencoder = autoencoder
        self.scale_factor = scale_factor

    def forward(self, z):
        """
        Decode the input latent representation to an image.

        Args:
            z (torch.Tensor): The input latent representation.

        Returns:
            torch.Tensor: The reconstructed image.
        """
        recon_pt_nda = self.autoencoder.decode_stage_2_outputs(z / self.scale_factor)
        return recon_pt_nda


def initialize_noise_latents(latent_shape, device):
    """
    Initialize random noise latents for image generation with float16.

    Args:
        latent_shape (tuple): The shape of the latent space.
        device (torch.device): The device to create the tensor on.

    Returns:
        torch.Tensor: Initialized noise latents.
    """
    return (
        torch.randn(
            [
                1,
            ]
            + list(latent_shape)
        )
        .half()
        .to(device)
    )


def run_controlnet_conditioned_image_dm(
    autoencoder,
    diffusion_unet,
    controlnet,
    noise_scheduler,
    scale_factor,
    device,
    controlnet_cond_tensor,
    spacing_tensor,
    latent_shape,
    output_size,
    noise_factor,
    top_region_index_tensor=None,
    bottom_region_index_tensor=None,
    modality_tensor=None,
    num_inference_steps=1000,
    autoencoder_sliding_window_infer_size=(96, 96, 96),
    autoencoder_sliding_window_infer_overlap=0.6667,
    cfg_guidance_scale=0.0,
    controlnet_uncond_tensor=None,
):
    """
    Run the ControlNet-conditioned image-DM denoising loop + AE decode.

    **Conditioning-modality-agnostic** — the caller supplies a pre-prepared
    ``controlnet_cond_tensor`` (and optionally an ``controlnet_uncond_tensor``
    for classifier-free guidance). The mask-conditioned wrapper lives in
    ``scripts/infer_image_from_mask.py``; future ControlNet variants trained
    on other conditioning modalities (e.g. image) can add their own wrapper
    and reuse this core.

    Args:
        autoencoder, diffusion_unet, controlnet: networks (image AE, image DM,
            ControlNet). Already moved to ``device`` and ``.eval()``'d.
        noise_scheduler: RFlow or DDPM scheduler matching the trained DM.
        scale_factor (float|Tensor): latent normalization factor.
        device (torch.device): inference device.
        controlnet_cond_tensor (Tensor): pre-prepared ControlNet conditioning
            of shape ``(B, C_cond, H_out, W_out, D_out)`` — already half().
            ``C_cond`` must match the ControlNet's
            ``conditioning_embedding_in_channels`` (default 8 for the
            mask-conditioned variant via ``binarize_labels``).
        spacing_tensor (Tensor): ``(B, 3)`` per-axis spacing × 100.
        latent_shape (tuple): ``(C_latent, H_lat, W_lat, D_lat)``.
        output_size (tuple): ``(H_out, W_out, D_out)``.
        noise_factor (float): multiplier on initial noise (typically 1.0).
        top_region_index_tensor, bottom_region_index_tensor (Tensor|None):
            one-hot 4-vectors. Required iff the diffusion UNet was trained
            with body-region conditioning (``ddpm-ct``).
        modality_tensor (Tensor): integer modality code per sample
            (CT=1, MR variants 8..32). Drives the HU-range selection AND the
            class-label input to the DM (if it supports modality).
        num_inference_steps (int): RFlow → 30; **DDPM → 1000 (required)**.
        autoencoder_sliding_window_infer_size, _overlap: AE-decode tiling.
        cfg_guidance_scale (float): classifier-free guidance scale. ``0``
            disables CFG. When > 0, ``controlnet_uncond_tensor`` is required.
        controlnet_uncond_tensor (Tensor|None): pre-prepared unconditional
            counterpart of ``controlnet_cond_tensor`` — same shape. Caller
            decides what "unconditional" means for the conditioning modality
            (mask conditioning: tumor-free mask; image conditioning: TBD).

    Returns:
        Tensor: synthetic image in HU/MR intensity range. Shape ``(B, 1,
        H_out, W_out, D_out)`` on CPU. **No background-mask cleanup is
        applied here** — that's modality-specific and lives in the wrapper.
    """
    if cfg_guidance_scale > 0 and controlnet_uncond_tensor is None:
        raise ValueError(
            "cfg_guidance_scale > 0 requires controlnet_uncond_tensor "
            "(caller must supply a modality-appropriate unconditional "
            "ControlNet conditioning tensor)."
        )

    if modality_tensor is not None and modality_tensor <= 7:
        # CT image intensity range
        a_min, a_max = -1000, 1000
    else:
        # MRI image intensity range
        a_min, a_max = 0, 1000
    # autoencoder output intensity range
    b_min, b_max = 0.0, 1.0

    include_body_region = diffusion_unet.include_top_region_index_input
    include_modality = diffusion_unet.num_class_embeds is not None

    recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(device)

    with torch.no_grad(), torch.amp.autocast("cuda"):
        logging.info("---- Start generating latent features... ----")
        start_time = time.time()

        latents = initialize_noise_latents(latent_shape, device) * noise_factor

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

        for t, next_t in progress_bar:
            # ControlNet forward
            controlnet_inputs = {
                "x": latents,
                "timesteps": torch.Tensor((t,)).to(device),
                "controlnet_cond": controlnet_cond_tensor,
            }
            if include_modality:
                controlnet_inputs["class_labels"] = modality_tensor
            if cfg_guidance_scale > 0:
                for k in list(controlnet_inputs.keys()):
                    if k == "class_labels":
                        controlnet_inputs[k] = torch.cat([modality_tensor, torch.zeros_like(modality_tensor)])
                    elif k == "controlnet_cond":
                        controlnet_inputs[k] = torch.cat([controlnet_cond_tensor, controlnet_uncond_tensor])
                    else:
                        controlnet_inputs[k] = torch.cat([controlnet_inputs[k]] * 2)

            down_block_res_samples, mid_block_res_sample = controlnet(**controlnet_inputs)

            # Diffusion UNet forward
            unet_inputs = {
                "x": latents,
                "timesteps": torch.Tensor((t,)).to(device),
                "spacing_tensor": spacing_tensor,
                "down_block_additional_residuals": down_block_res_samples,
                "mid_block_additional_residual": mid_block_res_sample,
            }
            if include_body_region:
                unet_inputs.update(
                    {
                        "top_region_index_tensor": top_region_index_tensor,
                        "bottom_region_index_tensor": bottom_region_index_tensor,
                    }
                )
            if include_modality:
                unet_inputs["class_labels"] = modality_tensor
            if cfg_guidance_scale > 0:
                for k in list(unet_inputs.keys()):
                    if k in ("down_block_additional_residuals", "mid_block_additional_residual"):
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

        del unet_inputs, controlnet_inputs, model_output, down_block_res_samples, mid_block_res_sample
        gc.collect()
        torch.cuda.empty_cache()

        # Sliding-window AE decode
        logging.info("---- Start decoding latent features into images... ----")
        start_time = time.time()
        inferer = SlidingWindowInferer(
            roi_size=list(autoencoder_sliding_window_infer_size),
            sw_batch_size=1,
            progress=True,
            mode="gaussian",
            overlap=autoencoder_sliding_window_infer_overlap,
            sw_device=device,
            device=torch.device("cpu"),
        )
        synthetic_images = dynamic_infer(inferer, recon_model, latents)
        # modality_tensor can be scalar (single mask) or shape (B,) (batch infer).
        # Use the first element so a >1-batch boolean tensor doesn't blow up in `if`.
        # All batch items share the same modality in our call sites.
        if modality_tensor is not None and int(modality_tensor.flatten()[0]) <= 7:
            synthetic_images = torch.clip(synthetic_images, b_min, b_max).cpu()
        else:
            synthetic_images = torch.clip(synthetic_images, b_min, None).cpu()
        end_time = time.time()
        logging.info(f"---- Image VAE decoding time: {end_time - start_time} seconds ----")

        # HU range mapping (modality-agnostic post-process)
        synthetic_images = (synthetic_images - b_min) / (b_max - b_min)
        synthetic_images = synthetic_images * (a_max - a_min) + a_min
        torch.cuda.empty_cache()

    return synthetic_images


def load_image_models(args, device: torch.device):
    """
    Load **image-side** networks (image AE + image DM + ControlNet) + the
    image noise scheduler from disk.

    Args:
        args: a config namespace already populated by ``load_config``. Must
            contain the keys: ``trained_autoencoder_path``,
            ``trained_diffusion_path``, ``trained_controlnet_path``,
            plus the network defs (``autoencoder_def``, ``diffusion_unet_def``,
            ``controlnet_def``, ``noise_scheduler``).
        device: target device.

    Returns:
        ``(autoencoder, diffusion_unet, controlnet, scale_factor, noise_scheduler)``.
        All networks are moved to ``device`` and set to ``.eval()`` mode.
    """
    from .utils import define_instance

    autoencoder = define_instance(args, "autoencoder_def").to(device)
    ckpt = torch.load(args.trained_autoencoder_path, weights_only=False)
    if "unet_state_dict" in ckpt:
        ckpt = ckpt["unet_state_dict"]
    autoencoder.load_state_dict(ckpt)

    diffusion_unet = define_instance(args, "diffusion_unet_def").to(device)
    ckpt_dm = torch.load(args.trained_diffusion_path, weights_only=False)
    diffusion_unet.load_state_dict(ckpt_dm["unet_state_dict"], strict=False)
    scale_factor = ckpt_dm["scale_factor"].to(device)

    controlnet = define_instance(args, "controlnet_def").to(device)
    ckpt_cn = torch.load(args.trained_controlnet_path, weights_only=False)
    monai.networks.utils.copy_model_state(controlnet, diffusion_unet.state_dict())
    controlnet.load_state_dict(ckpt_cn["controlnet_state_dict"], strict=False)

    noise_scheduler = define_instance(args, "noise_scheduler")

    autoencoder.eval()
    diffusion_unet.eval()
    controlnet.eval()
    return autoencoder, diffusion_unet, controlnet, scale_factor, noise_scheduler


def load_mask_models(args, device: torch.device):
    """
    Load **mask-side** networks (mask AE + mask DM) + the mask noise scheduler.

    Used by the paired-inference path (``LDMSampler`` Path A, where a mask is
    generated from ``controllable_anatomy_size``). The image-only and
    image-from-mask CLIs don't need these.

    Args:
        args: a config namespace already populated by ``load_config``. Must
            contain ``trained_mask_generation_autoencoder_path``,
            ``trained_mask_generation_diffusion_path``,
            ``mask_generation_autoencoder_def``,
            ``mask_generation_diffusion_def``,
            ``mask_generation_noise_scheduler``.
        device: target device.

    Returns:
        ``(mask_autoencoder, mask_diffusion_unet, mask_scale_factor, mask_noise_scheduler)``.
        Networks are moved to ``device`` and set to ``.eval()``.
    """
    from .utils import define_instance

    mask_autoencoder = define_instance(args, "mask_generation_autoencoder_def").to(device)
    ckpt_mae = torch.load(args.trained_mask_generation_autoencoder_path, weights_only=True)
    mask_autoencoder.load_state_dict(ckpt_mae)

    mask_diffusion_unet = define_instance(args, "mask_generation_diffusion_def").to(device)
    ckpt_mdm = torch.load(args.trained_mask_generation_diffusion_path, weights_only=True)
    mask_diffusion_unet.load_state_dict(ckpt_mdm["unet_state_dict"])
    mask_scale_factor = ckpt_mdm["scale_factor"]

    mask_noise_scheduler = define_instance(args, "mask_generation_noise_scheduler")

    mask_autoencoder.eval()
    mask_diffusion_unet.eval()
    return mask_autoencoder, mask_diffusion_unet, mask_scale_factor, mask_noise_scheduler


def load_paired_inference_models(args, device: torch.device) -> dict:
    """
    Load **all** networks needed for the paired image+mask inference path
    (``LDMSampler.sample_multiple_images``). Convenience wrapper that calls
    both ``load_image_models`` and ``load_mask_models``.

    Returns a dict with keys matching the names ``LDMSampler.__init__``
    expects, so callers can do ``LDMSampler(**load_paired_inference_models(args, device), ...other args...)``
    (though usually you'll pull individual entries out explicitly).
    """
    autoencoder, diffusion_unet, controlnet, scale_factor, noise_scheduler = load_image_models(args, device)
    mask_autoencoder, mask_diffusion_unet, mask_scale_factor, mask_noise_scheduler = load_mask_models(args, device)
    return {
        "autoencoder": autoencoder,
        "diffusion_unet": diffusion_unet,
        "controlnet": controlnet,
        "scale_factor": scale_factor,
        "noise_scheduler": noise_scheduler,
        "mask_generation_autoencoder": mask_autoencoder,
        "mask_generation_diffusion_unet": mask_diffusion_unet,
        "mask_generation_scale_factor": mask_scale_factor,
        "mask_generation_noise_scheduler": mask_noise_scheduler,
    }


def build_conditioning_tensors(
    label_or_image: torch.Tensor,
    spacing: tuple,
    modality: int,
    include_body_region: bool,
    device: torch.device,
):
    """
    Pack the per-case auxiliary tensors that ``run_controlnet_conditioned_image_dm``
    expects (spacing, body-region indices, modality).

    For mask conditioning, ``label_or_image`` is the label NIfTI and
    ``get_body_region_index_from_mask`` is used to derive the body-region
    indices. For image conditioning, the body-region indices probably need
    to come from a paired mask or be set to None (depends on whether the
    image-conditioned DM was trained with body-region inputs).
    """
    spacing_tensor = torch.FloatTensor(list(spacing)).unsqueeze(0).half().to(device) * 1e2

    if include_body_region:
        top_idx, bottom_idx = get_body_region_index_from_mask(label_or_image.squeeze(0))
        top_region_index_tensor = torch.FloatTensor(top_idx).unsqueeze(0).half().to(device) * 1e2
        bottom_region_index_tensor = torch.FloatTensor(bottom_idx).unsqueeze(0).half().to(device) * 1e2
    else:
        top_region_index_tensor = None
        bottom_region_index_tensor = None

    modality_tensor = modality * torch.ones((1,), dtype=torch.long).to(device)
    return spacing_tensor, top_region_index_tensor, bottom_region_index_tensor, modality_tensor
