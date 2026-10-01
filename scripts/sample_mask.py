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
Mask generation module (v2: AdaGN, 19-d conditioning, RFlow + CFG).

Generates a 3D body-region label mask from scratch using a rectified-flow
latent diffusion model conditioned on a 19-d vector:
  slots 0-8  : organ sizes  (liver, spleen, stomach, pancreas, colon,
                              left kidney, right kidney, lung, gallbladder)
  slots 9-13 : tumor sizes  (lung tumor, pancreatic tumor, hepatic tumor,
                              colon cancer primaries, bone lesion)
  slots 14-18: demographics (age, sex, weight, bmi, height) — normalized

Unspecified slots use the sentinel -1 (no conditioning).  CFG (classifier-free
guidance) is applied at every step: conditioning = uncond + scale*(cond - uncond).

Also hosts the shared helpers ``ReconModel`` and ``initialize_noise_latents``
that the image-from-mask module re-imports, and the input validation
functions ``check_input_ct`` / ``check_input_mr`` that gate the mask-pipeline
inputs (``output_size`` / ``spacing`` / ``controllable_anatomy_size``).
"""

import json
import logging

import torch
from monai.inferers.inferer import SlidingWindowInferer

from .utils import (
    dynamic_infer,
    general_mask_generation_post_process,
    remap_labels,
)

# ReconModel + initialize_noise_latents are shared with the image-from-mask
# pipeline (and any future conditioning-modality wrapper), so they live in
# utils_infer. Re-export them from this module's namespace for backward
# compatibility with callers that imported them from scripts.sample_mask
# (or via the scripts.sample shim).
from .utils_infer import ReconModel, initialize_noise_latents, move_models  # noqa: F401

# Fixed demographics normalization ranges (lo=0, hi listed below). Must match
# the training demographics_transform.py DEFAULT_RANGES / build_manifest.py NORM_RANGE.
_DEMOG_MAX = {"age": 120.0, "weight": 200.0, "bmi": 75.0, "height": 200.0}

# v2 anatomy conditioning slot order (14 slots: 9 organs + 5 tumors).
# Matches ANATOMY_ORGANS in the training diff_model_train.py.
ANATOMY_SIZE_IDX = {
    "liver": 0,
    "spleen": 1,
    "stomach": 2,
    "pancreas": 3,
    "colon": 4,
    "left kidney": 5,
    "right kidney": 6,
    "lung": 7,
    "gallbladder": 8,
    "lung tumor": 9,
    "pancreatic tumor": 10,
    "hepatic tumor": 11,
    "colon cancer primaries": 12,
    "bone lesion": 13,
}
N_ANATOMY = 14  # 9 organs + 5 tumors
N_DEMOG = 5     # age, sex, weight, bmi, height
N_COND = N_ANATOMY + N_DEMOG  # 19


def ldm_conditional_sample_one_mask(
    autoencoder,
    diffusion_unet,
    noise_scheduler,
    scale_factor,
    conditioning,
    device,
    latent_shape,
    label_dict_remap_json,
    num_inference_steps=100,
    cfg_guidance_scale=2.0,
    spacing=None,
    autoencoder_sliding_window_infer_size=[96, 96, 96],
    autoencoder_sliding_window_infer_overlap=0.6667,
    low_vram=False,
):
    """
    Generate a single synthetic mask using the v2 AdaGN latent diffusion model.

    Args:
        autoencoder: mask AE model.
        diffusion_unet: mask DM (DiffusionModelUNetMaisiAdaGN).
        noise_scheduler: RFlowScheduler instance.
        scale_factor (float): AE latent scale factor.
        conditioning (list): 19-d conditioning vector [anatomy(14) | demographics(5)].
            Unspecified slots should be -1.
        device: target device.
        latent_shape (tuple): shape of the mask latent, e.g. (4, 64, 64, 64).
        label_dict_remap_json (str): path to label remapping JSON.
        num_inference_steps (int): RFlow denoising steps. Default 100.
        cfg_guidance_scale (float): classifier-free guidance scale. 0 disables CFG.
        spacing (list|None): voxel spacing in mm, e.g. [1.5, 1.5, 1.5]. Used for
            the UNet's spacing embedding (SPACING_SCALE = 1e2 applied internally).
            Defaults to [1.5, 1.5, 1.5].
        autoencoder_sliding_window_infer_size: AE decode sliding window size.
        autoencoder_sliding_window_infer_overlap: AE decode sliding window overlap.

    Returns:
        torch.Tensor: generated mask with MAISI 132-class labels + body=200.
    """
    if low_vram:
        move_models((diffusion_unet,), device)

    if spacing is None:
        spacing = [1.5, 1.5, 1.5]

    recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(device)

    # Build conditioning tensors — shape (1, 1, 19) for cross-attention
    cond = torch.tensor(conditioning, dtype=torch.float32, device=device).view(1, 1, N_COND).half()
    uncond = torch.full((1, 1, N_COND), -1.0, dtype=torch.float16, device=device)

    # Spacing tensor (scaled by 1e2 to match training normalisation); shape (1, 3)
    spacing_tensor = torch.tensor(spacing, dtype=torch.float32, device=device).view(1, 3).half() * 1e2

    with torch.no_grad(), torch.amp.autocast("cuda"):
        latents = initialize_noise_latents(latent_shape, device)

        # latent spatial size for RFlow timestep transform (64³ for the 256³ mask model)
        lat_numel = int(latent_shape[1]) * int(latent_shape[2]) * int(latent_shape[3])
        noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps, input_img_size_numel=lat_numel)

        timesteps = noise_scheduler.timesteps
        for i, t in enumerate(timesteps):
            next_t = timesteps[i + 1] if i + 1 < len(timesteps) else None
            t_batch = t.unsqueeze(0).to(device)
            if cfg_guidance_scale > 0:
                x_in = torch.cat([latents, latents])
                t_in = t_batch.expand(2)
                c_in = torch.cat([cond, uncond])
                sp_in = torch.cat([spacing_tensor, spacing_tensor])
                mo_all = diffusion_unet(x=x_in, timesteps=t_in, context=c_in, spacing_tensor=sp_in)
                mo_c, mo_u = mo_all.chunk(2)
                mo = mo_u + cfg_guidance_scale * (mo_c - mo_u)
            else:
                mo = diffusion_unet(x=latents, timesteps=t_batch, context=cond, spacing_tensor=spacing_tensor)
            out = noise_scheduler.step(mo, t, latents, next_timestep=next_t)
            latents = out[0] if isinstance(out, (tuple, list)) else out

        inferer = SlidingWindowInferer(
            roi_size=autoencoder_sliding_window_infer_size,
            sw_batch_size=1,
            progress=True,
            mode="gaussian",
            overlap=autoencoder_sliding_window_infer_overlap,
            sw_device=device,
            device=torch.device("cpu"),
        )
        if low_vram:
            move_models((diffusion_unet,), torch.device("cpu"))
            move_models((autoencoder,), device)
        if isinstance(scale_factor, torch.Tensor):
            scale_factor = scale_factor.to(device)
        recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(device)
        synthetic_mask = dynamic_infer(inferer, recon_model, latents)
        synthetic_mask = torch.softmax(synthetic_mask, dim=1)
        synthetic_mask = torch.argmax(synthetic_mask, dim=1, keepdim=True)
        synthetic_mask = remap_labels(synthetic_mask, label_dict_remap_json)

        data = synthetic_mask.squeeze().cpu().detach().numpy()

        # Tumor slot indices in v2: 9=lung tumor, 10=pancreatic tumor, 11=hepatic tumor,
        # 12=colon cancer primaries, 13=bone lesion (MAISI labels: 23, 24, 26, 27, 128)
        tumor_maisi_labels = [23, 24, 26, 27, 128]
        target_tumor_label = None
        cond_cpu = cond.squeeze().float().cpu()
        for idx, maisi_label in zip(range(9, 14), tumor_maisi_labels):
            if cond_cpu[idx].item() != -1.0:
                target_tumor_label = maisi_label

        logging.info(f"target_tumor_label for postprocess: {target_tumor_label}")
        data = general_mask_generation_post_process(data, target_tumor_label=target_tumor_label, device=device)
        synthetic_mask = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).to(device)

        if low_vram:
            move_models((autoencoder,), torch.device("cpu"))

    return synthetic_mask


def filter_mask_with_organs(combine_label, anatomy_list):
    """
    Filter a mask to only include specified organs.

    Args:
        combine_label (torch.Tensor): The input mask.
        anatomy_list (list): List of organ labels to keep.

    Returns:
        torch.Tensor: The filtered mask.
    """
    combine_label = combine_label.long()
    for i in range(len(anatomy_list)):
        organ = anatomy_list[i]
        combine_label[combine_label == organ] = -(i + 1)
    combine_label[combine_label > 0] = 0
    combine_label = -combine_label
    return combine_label


def check_input_ct(
    body_region,
    anatomy_list,
    label_dict_json,
    output_size,
    spacing,
    controllable_anatomy_size=[],
    controllable_demographics=None,
):
    """
    Validate input parameters for CT mask generation (v2 model).

    Args:
        body_region (list): Body regions for Path B (mask DB lookup).
        anatomy_list (list): Required anatomy label IDs.
        label_dict_json (str): Path to the label dictionary JSON.
        output_size (tuple): Output volume shape.
        spacing (tuple): Voxel spacing in mm.
        controllable_anatomy_size (list): At most ONE [organ_name, size] pair
            that triggers Path A (diffusion from scratch). Empty → Path B.
        controllable_demographics (list|None): Optional list of [name, value]
            demographics in original units: age (yr), weight (kg), height (cm),
            bmi (kg/m²), sex ("M"/"F"). None or [] → no demographic conditioning.
    """
    if output_size[0] != output_size[1]:
        raise ValueError(f"The first two components of output_size need to be equal, yet got {output_size}.")
    if (output_size[0] not in [256, 384, 512]) or (output_size[2] not in [128, 256, 384, 512, 640, 768]):
        raise ValueError(
            f"output_size[0] must be in [256, 384, 512] and output_size[2] in [128, 256, 384, 512, 640, 768], "
            f"got {output_size}."
        )
    if spacing[0] != spacing[1]:
        raise ValueError(f"The first two components of spacing need to be equal, yet got {spacing}.")
    if spacing[0] < 0.5 or spacing[0] > 3.0 or spacing[2] < 0.5 or spacing[2] > 5.0:
        raise ValueError(
            f"spacing[0] must be in [0.5, 3.0] mm and spacing[2] in [0.5, 5.0] mm, got {spacing}."
        )
    if output_size[0] * spacing[0] < 256:
        FOV = [output_size[axis] * spacing[axis] for axis in range(3)]  # noqa: N806
        raise ValueError(
            f"spacing ({spacing} mm) × output_size ({output_size}) gives FOV {FOV} mm. "
            "Recommend FOV ≥ 256 mm in x/y (≥ 384 mm for abdomen)."
        )

    # Validate controllable_demographics (optional; None/[] = no demographic conditioning).
    # weight and bmi must not both be given — they come from disjoint training datasets.
    if controllable_demographics:
        available_demographics = ["age", "sex", "weight", "bmi", "height"]
        demographics_max = {"age": 120.0, "weight": 200.0, "bmi": 75.0, "height": 200.0}
        demographics_observed = {"age": (19, 87), "weight": (22, 144), "bmi": (18, 63), "height": (160, 190)}
        seen_demographics = []
        for demographics_pair in controllable_demographics:
            name, value = demographics_pair[0], demographics_pair[1]
            if name not in available_demographics:
                raise ValueError(
                    f"controllable_demographics name must be one of {available_demographics}, got {name!r}."
                )
            if name in seen_demographics:
                raise ValueError(f"Duplicate controllable_demographics field: {name!r}.")
            if name == "sex":
                if str(value).upper() not in ("M", "F"):
                    raise ValueError(f"controllable_demographics 'sex' must be 'M' or 'F', got {value!r}.")
            else:
                if value < 0 or value > demographics_max[name]:
                    units = {"age": "yr", "weight": "kg", "bmi": "kg/m²", "height": "cm"}[name]
                    raise ValueError(
                        f"controllable_demographics '{name}'={value} is outside [0, {demographics_max[name]:g}] {units}."
                    )
                obs_lo, obs_hi = demographics_observed[name]
                if value < obs_lo or value > obs_hi:
                    logging.warning(
                        f"controllable_demographics '{name}'={value} is outside the training range "
                        f"[{obs_lo}, {obs_hi}]; the model may extrapolate."
                    )
            seen_demographics.append(name)
        if "weight" in seen_demographics and "bmi" in seen_demographics:
            raise ValueError(
                "Provide only ONE of 'weight' or 'bmi' in controllable_demographics — "
                "they co-occur in <0.4% of training data."
            )
        if "height" in seen_demographics:
            logging.warning(
                "'height' is present in only ~0.4% of training data; "
                "its conditioning effect is unreliable. Prefer age/sex/weight/bmi."
            )
        if controllable_anatomy_size and len(seen_demographics) >= 3:
            logging.warning(
                f"controllable_anatomy_size set together with {len(seen_demographics)} demographics "
                f"({seen_demographics}): this is heavily constrained and out-of-distribution. "
                "Generation quality is not guaranteed."
            )

    if controllable_anatomy_size is None:
        logging.info("`controllable_anatomy_size` is not provided.")
        return

    # v2: single-target model — at most ONE [organ_name, size] pair accepted.
    if len(controllable_anatomy_size) > 1:
        raise ValueError(
            f"controllable_anatomy_size accepts at most ONE entry (one organ OR one tumor) "
            f"for the v2 mask model, got {len(controllable_anatomy_size)}: {controllable_anatomy_size}."
        )

    available_controllable_organ = [
        "liver", "spleen", "stomach", "pancreas", "colon",
        "left kidney", "right kidney", "lung", "gallbladder",
    ]
    available_controllable_tumor = [
        "lung tumor", "pancreatic tumor", "hepatic tumor",
        "colon cancer primaries", "bone lesion",
    ]
    available_controllable_anatomy = available_controllable_organ + available_controllable_tumor

    for pair in controllable_anatomy_size:
        if pair[0] not in available_controllable_anatomy:
            raise ValueError(
                f"controllable_anatomy must be one of {available_controllable_anatomy}, got {pair[0]!r}."
            )
        size = pair[1]
        if size != -1 and (size < 0 or size > 1.0):
            raise ValueError(
                f"Controllable size must be in [0, 1] or -1, got {size}."
            )

    if len(controllable_anatomy_size) > 0:
        logging.info(
            f"`controllable_anatomy_size` is set: Path A (diffusion) with {controllable_anatomy_size}. "
            "body_region and anatomy_list will be ignored."
        )
    else:
        logging.info(
            f"`controllable_anatomy_size` is empty: Path B (real mask DB) with "
            f"body_region={body_region}, anatomy_list={anatomy_list}."
        )
        available_body_region = ["head", "chest", "thorax", "abdomen", "pelvis", "lower"]
        for region in body_region:
            if region not in available_body_region:
                raise ValueError(
                    f"body_region components must be in {available_body_region}, got {region!r}."
                )
        with open(label_dict_json) as f:
            label_dict = json.load(f)
        for anatomy in anatomy_list:
            if anatomy not in label_dict.keys():
                raise ValueError(
                    f"anatomy_list components must be in label_dict keys, got {anatomy!r}."
                )
    logging.info(f"Output: spacing={spacing} mm, size={output_size}.")


def check_input_mr(
    body_region,
    anatomy_list,
    label_dict_json,
    output_size,
    spacing,
    controllable_anatomy_size=[("pancreas", 0.5)],
):
    """
    Validate input parameters for MR image generation.

    Args:
        body_region (list): List of body regions.
        anatomy_list (list): List of anatomical structures.
        label_dict_json (str): Path to the label dictionary JSON file.
        output_size (tuple): Desired output size of the image.
        spacing (tuple): Desired voxel spacing.
        controllable_anatomy_size (list): List of tuples specifying controllable anatomy sizes.

    Raises:
        ValueError: If any input parameter is invalid.
    """
    if output_size[0] != output_size[1] and output_size[0] != output_size[2] and output_size[2] != output_size[1]:
        raise ValueError(f"At least two components of output_size need to be equal, yet got {output_size}.")
    if output_size[2] == 128:
        if output_size[0] != output_size[1]:
            raise ValueError(f"Two first components of output_size need to be equal when the third size is 128, yet got {output_size}.")
        if output_size[0] not in [128, 256, 384, 512]:
            raise ValueError(f"The output_size[0] have to be chosen from [128, 256, 384, 512] when output_size[2]=128, yet got {output_size}.")
    elif output_size[2] == 256:
        if (
            (output_size[0] == 128 and output_size[1] == 256)
            or (output_size[0] == 256 and output_size[1] == 128)
            or (output_size[0] == 256 and output_size[1] == 256)
        ):
            pass
        else:
            raise ValueError(
                f"The output_size can only be [128,256,256] or [256,128,256], or [256,256,256] when output_size[2]=256, yet got {output_size}."
            )
    else:
        raise ValueError(f"The output_size[2] have to be chosen from [128, 256], yet got {output_size}.")

    if any(s < 0.4 for s in spacing) or any(s > 5.0 for s in spacing):
        raise ValueError(f"spacing have to be between 0.4 and 5.0 mm, yet got {spacing}.")

    with open(label_dict_json) as f:
        label_dict = json.load(f)
    for anatomy in anatomy_list:
        if anatomy not in label_dict.keys():
            raise ValueError(f"The components in anatomy_list have to be chosen from {label_dict.keys()}, yet got {anatomy}.")
    logging.info(f"The generate results will have voxel size to be {spacing}mm, volume size to be {output_size}.")
