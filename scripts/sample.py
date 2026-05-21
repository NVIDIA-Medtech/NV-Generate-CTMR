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
LDM sampler orchestrator + backward-compatible re-exports.

The previously-monolithic ``sample.py`` has been split into two focused
modules so that "how mask generation works" and "how image-from-mask
inference works" are each explained in one place:

- ``scripts.sample_mask`` — mask-generation pipeline (see ``skills/mask-generation.md``).
- ``scripts.infer_image_from_mask`` — image-from-mask pipeline (see ``skills/image-from-mask.md``).

This file re-exports the public symbols from those modules so existing
imports (``from scripts.sample import ...``) keep working, and hosts the
``LDMSampler`` orchestrator class that wires both pipelines together with
mask-database lookup, quality checks, and output-size enforcement.
"""

import json
import logging
import os
import random
import time
from datetime import datetime

import monai
import torch
from monai.data import MetaTensor
from monai.transforms import Compose, SaveImage
from monai.utils import set_determinism

from .augmentation import augmentation
from .find_masks import find_masks
from .quality_check import is_outlier
from .utils import get_body_region_index_from_mask

# Backward-compat re-exports — existing callers ``from scripts.sample import X``
# keep working. ``X`` now physically lives in sample_mask / infer_image_from_mask.
from .sample_mask import (  # noqa: F401  (re-exported)
    ReconModel,
    check_input_ct,
    check_input_mr,
    filter_mask_with_organs,
    initialize_noise_latents,
    ldm_conditional_sample_one_mask,
)
from .infer_image_from_mask import (  # noqa: F401  (re-exported)
    crop_img_body_mask,
    ldm_conditional_sample_one_image,
)


class LDMSampler:
    """
    A sampler class for generating synthetic medical images and masks using latent diffusion models.

    Attributes:
        Various attributes related to model configuration, input parameters, and generation settings.
    """

    def __init__(
        self,
        body_region,
        anatomy_list,
        all_mask_files_json,
        all_anatomy_size_conditions_json,
        all_mask_files_base_dir,
        label_dict_json,
        label_dict_remap_json,
        autoencoder,
        diffusion_unet,
        controlnet,
        noise_scheduler,
        scale_factor,
        mask_generation_autoencoder,
        mask_generation_diffusion_unet,
        mask_generation_scale_factor,
        mask_generation_noise_scheduler,
        device,
        latent_shape,
        mask_generation_latent_shape,
        output_size,
        output_dir,
        controllable_anatomy_size,
        image_output_ext=".nii.gz",
        label_output_ext=".nii.gz",
        real_img_median_statistics="./configs/image_median_statistics_ct.json",
        spacing=[1, 1, 1],
        modality=1,
        num_inference_steps=None,
        mask_generation_num_inference_steps=None,
        random_seed=None,
        autoencoder_sliding_window_infer_size=[96, 96, 96],
        autoencoder_sliding_window_infer_overlap=0.6667,
        cfg_guidance_scale=0.0,
    ) -> None:
        """
        Initialize the LDMSampler with various parameters and models.

        Args:
            Various parameters related to model configuration, input settings, and output specifications.
        """
        self.random_seed = random_seed
        if random_seed is not None:
            set_determinism(seed=random_seed)

        with open(label_dict_json) as f:
            label_dict = json.load(f)
        self.all_anatomy_size_conditions_json = all_anatomy_size_conditions_json

        # initialize variables
        self.body_region = body_region
        self.anatomy_list = [label_dict[organ] for organ in anatomy_list]
        self.all_mask_files_json = all_mask_files_json
        self.data_root = all_mask_files_base_dir
        self.label_dict_remap_json = label_dict_remap_json
        self.autoencoder = autoencoder
        self.diffusion_unet = diffusion_unet
        self.controlnet = controlnet
        self.noise_scheduler = noise_scheduler
        self.scale_factor = scale_factor
        self.mask_generation_autoencoder = mask_generation_autoencoder
        self.mask_generation_diffusion_unet = mask_generation_diffusion_unet
        self.mask_generation_scale_factor = mask_generation_scale_factor
        self.mask_generation_noise_scheduler = mask_generation_noise_scheduler
        self.device = device
        self.latent_shape = latent_shape
        self.mask_generation_latent_shape = mask_generation_latent_shape
        self.output_size = output_size
        self.output_dir = output_dir
        self.noise_factor = 1.0
        self.cfg_guidance_scale = cfg_guidance_scale
        self.controllable_anatomy_size = controllable_anatomy_size
        if len(self.controllable_anatomy_size):
            logging.info("controllable_anatomy_size is given, mask generation is triggered!")
            # overwrite the anatomy_list by given organs in self.controllable_anatomy_size
            self.anatomy_list = [label_dict[organ_and_size[0]] for organ_and_size in self.controllable_anatomy_size]
        self.image_output_ext = image_output_ext
        self.label_output_ext = label_output_ext
        # Set the default value for number of inference steps to 1000
        self.num_inference_steps = num_inference_steps if num_inference_steps is not None else 1000
        self.mask_generation_num_inference_steps = mask_generation_num_inference_steps if mask_generation_num_inference_steps is not None else 1000

        if any(size % 16 != 0 for size in autoencoder_sliding_window_infer_size):
            raise ValueError(f"autoencoder_sliding_window_infer_size must be divisible by 16.\n Got {autoencoder_sliding_window_infer_size}")
        if not (0 <= autoencoder_sliding_window_infer_overlap <= 1):
            raise ValueError(
                f"Value of autoencoder_sliding_window_infer_overlap must be between 0 and 1.\n Got {autoencoder_sliding_window_infer_overlap}"
            )
        self.autoencoder_sliding_window_infer_size = autoencoder_sliding_window_infer_size
        self.autoencoder_sliding_window_infer_overlap = autoencoder_sliding_window_infer_overlap

        # quality check args
        self.max_try_time = 2  # if not pass quality check, will try self.max_try_time times
        with open(real_img_median_statistics) as json_file:
            self.median_statistics = json.load(json_file)
        self.label_int_dict = {
            "liver": [1],
            "spleen": [3],
            "pancreas": [4],
            "kidney": [5, 14],
            "lung": [28, 29, 30, 31, 31],
            "brain": [22],
            "hepatic tumor": [26],
            "bone lesion": [128],
            "lung tumor": [23],
            "colon cancer primaries": [27],
            "pancreatic tumor": [24],
            "bone": list(range(33, 57)) + list(range(63, 98)) + [120, 122, 127],
        }

        # networks
        self.autoencoder.eval()
        self.diffusion_unet.eval()
        self.controlnet.eval()
        self.mask_generation_autoencoder.eval()
        self.mask_generation_diffusion_unet.eval()

        self.spacing = spacing
        self.modality_tensor = modality * torch.ones((1,), dtype=torch.long).to(device)
        self.include_body_region = self.diffusion_unet.include_top_region_index_input
        self.include_modality = self.diffusion_unet.num_class_embeds is not None

        val_transforms_list = [
            monai.transforms.LoadImaged(keys=["pseudo_label"]),
            monai.transforms.EnsureChannelFirstd(keys=["pseudo_label"]),
            monai.transforms.Orientationd(keys=["pseudo_label"], axcodes="RAS"),
            monai.transforms.EnsureTyped(keys=["pseudo_label"], dtype=torch.long),
            monai.transforms.Lambdad(keys="spacing", func=lambda x: torch.FloatTensor(x)),
            monai.transforms.Lambdad(keys="spacing", func=lambda x: x * 1e2),
        ]
        if self.include_body_region:
            val_transforms_list += [
                monai.transforms.Lambdad(keys="top_region_index", func=lambda x: torch.FloatTensor(x)),
                monai.transforms.Lambdad(keys="bottom_region_index", func=lambda x: torch.FloatTensor(x)),
                monai.transforms.Lambdad(keys="top_region_index", func=lambda x: x * 1e2),
                monai.transforms.Lambdad(keys="bottom_region_index", func=lambda x: x * 1e2),
            ]

        self.val_transforms = Compose(val_transforms_list)
        logging.info("LDM sampler initialized.")

    def sample_multiple_images(self, num_img):
        """
        Generate multiple synthetic images and masks.

        Args:
            num_img (int): Number of images to generate.
        """
        modality_tensor = self.modality_tensor
        output_filenames = []
        if len(self.controllable_anatomy_size) > 0:
            # we will use mask generation instead of finding candidate masks
            # create a dummy selected_mask_files for placeholder
            selected_mask_files = list(range(num_img))
            # prerpare organ size conditions
            anatomy_size_condition = self.prepare_anatomy_size_condition(self.controllable_anatomy_size)
        else:
            need_resample = False
            # find candidate mask and save to candidate_mask_files
            candidate_mask_files = find_masks(
                self.body_region,
                self.anatomy_list,
                self.spacing,
                self.output_size,
                True,
                self.all_mask_files_json,
                self.data_root,
            )
            if len(candidate_mask_files) < num_img:
                # if we cannot find enough masks based on the exact match of anatomy list, spacing, and output size,
                # then we will try to find the closest mask in terms of  spacing, and output size.
                logging.info("Resample mask file to get desired output size and spacing")
                candidate_mask_files = self.find_closest_masks(num_img)
                need_resample = True

            selected_mask_files = self.select_mask(candidate_mask_files, num_img)
            logging.info(f"Images will be generated based on {selected_mask_files}.")
            if len(selected_mask_files) < num_img:
                raise ValueError(
                    f"len(selected_mask_files) ({len(selected_mask_files)}) < num_img ({num_img}). "
                    "This should not happen. Please revisit function select_mask(self, candidate_mask_files, num_img)."
                )

        num_generated_img = 0
        for index_s in range(len(selected_mask_files)):
            item = selected_mask_files[index_s]
            if num_generated_img >= num_img:
                break
            logging.info("---- Start preparing masks... ----")
            start_time = time.time()
            if len(self.controllable_anatomy_size) > 0:
                # generate a synthetic mask
                (
                    combine_label_or,
                    top_region_index_tensor,
                    bottom_region_index_tensor,
                    spacing_tensor,
                ) = self.prepare_one_mask_and_meta_info(anatomy_size_condition)
            else:
                # read in mask file
                mask_file = item["mask_file"]
                if_aug = item["if_aug"]
                (
                    combine_label_or,
                    top_region_index_tensor,
                    bottom_region_index_tensor,
                    spacing_tensor,
                ) = self.read_mask_information(mask_file)
                if need_resample:
                    combine_label_or = self.ensure_output_size_and_spacing(combine_label_or)
                # mask augmentation
                if if_aug:
                    combine_label_or = augmentation(combine_label_or, self.output_size, self.random_seed)
            end_time = time.time()
            logging.info(f"---- Mask preparation time: {end_time - start_time} seconds ----")
            torch.cuda.empty_cache()
            # start generation
            synthetic_images, synthetic_labels = self.sample_one_pair(
                combine_label_or,
                top_region_index_tensor,
                bottom_region_index_tensor,
                spacing_tensor,
                modality_tensor,
            )
            # synthetic image quality check
            pass_quality_check = self.quality_check_ct(
                synthetic_images.cpu().detach().numpy(),
                combine_label_or.cpu().detach().numpy(),
                perform_quality_check=(modality_tensor <= 7 and modality_tensor >= 1),
            )
            if pass_quality_check or (num_img - num_generated_img) >= (len(selected_mask_files) - index_s):
                if not pass_quality_check:
                    logging.info(
                        "Generated image/label pair did not pass quality check, but will still save them. "
                        "Please consider changing spacing and output_size to facilitate a more realistic setting."
                    )
                num_generated_img = num_generated_img + 1
                # save image/label pairs
                output_postfix = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                synthetic_labels.meta["filename_or_obj"] = "sample.nii.gz"
                synthetic_images = MetaTensor(synthetic_images, meta=synthetic_labels.meta)
                img_saver = SaveImage(
                    output_dir=self.output_dir,
                    output_postfix=output_postfix + "_image",
                    output_ext=self.image_output_ext,
                    separate_folder=False,
                )
                img_saver(synthetic_images[0])
                synthetic_images_filename = os.path.join(self.output_dir, "sample_" + output_postfix + "_image" + self.image_output_ext)
                # filter out the organs that are not in anatomy_list
                synthetic_labels = filter_mask_with_organs(synthetic_labels, self.anatomy_list)
                label_saver = SaveImage(
                    output_dir=self.output_dir,
                    output_postfix=output_postfix + "_label",
                    output_ext=self.label_output_ext,
                    separate_folder=False,
                )
                label_saver(synthetic_labels[0])
                synthetic_labels_filename = os.path.join(self.output_dir, "sample_" + output_postfix + "_label" + self.label_output_ext)
                output_filenames.append([synthetic_images_filename, synthetic_labels_filename])
            else:
                logging.info("Generated image/label pair did not pass quality check, will re-generate another pair.")
        return output_filenames

    def select_mask(self, candidate_mask_files, num_img):
        """
        Select mask files for image generation.

        Args:
            candidate_mask_files (list): List of candidate mask files.
            num_img (int): Number of images to generate.

        Returns:
            list: Selected mask files with augmentation flags.
        """
        selected_mask_files = []
        random.shuffle(candidate_mask_files)

        for n in range(len(candidate_mask_files)):
            mask_file = candidate_mask_files[n % len(candidate_mask_files)]
            selected_mask_files.append({"mask_file": mask_file, "if_aug": True})
        return selected_mask_files

    def sample_one_pair(
        self,
        combine_label_or_aug,
        top_region_index_tensor,
        bottom_region_index_tensor,
        spacing_tensor,
        modality_tensor,
    ):
        """
        Generate a single pair of synthetic image and mask.

        Args:
            combine_label_or_aug (torch.Tensor): Combined label tensor or augmented label.
            top_region_index_tensor (torch.Tensor): Tensor specifying the top region index.
            bottom_region_index_tensor (torch.Tensor): Tensor specifying the bottom region index.
            spacing_tensor (torch.Tensor): Tensor specifying the spacing.
            modality_tensor (torch.Tensor): Int Tensor specifying the modality.

        Returns:
            tuple: A tuple containing the synthetic image and its corresponding label.
        """
        # generate image/label pairs
        synthetic_images, synthetic_labels = ldm_conditional_sample_one_image(
            autoencoder=self.autoencoder,
            diffusion_unet=self.diffusion_unet,
            controlnet=self.controlnet,
            noise_scheduler=self.noise_scheduler,
            scale_factor=self.scale_factor,
            device=self.device,
            combine_label_or=combine_label_or_aug,
            top_region_index_tensor=top_region_index_tensor,
            bottom_region_index_tensor=bottom_region_index_tensor,
            spacing_tensor=spacing_tensor,
            modality_tensor=modality_tensor,
            latent_shape=self.latent_shape,
            output_size=self.output_size,
            noise_factor=self.noise_factor,
            num_inference_steps=self.num_inference_steps,
            autoencoder_sliding_window_infer_size=self.autoencoder_sliding_window_infer_size,
            autoencoder_sliding_window_infer_overlap=self.autoencoder_sliding_window_infer_overlap,
            cfg_guidance_scale=self.cfg_guidance_scale,
        )
        return synthetic_images, synthetic_labels

    def prepare_anatomy_size_condition(
        self,
        controllable_anatomy_size,
    ):
        """
        Prepare anatomy size conditions for mask generation.

        Args:
            controllable_anatomy_size (list): List of tuples specifying controllable anatomy sizes.

        Returns:
            list: Prepared anatomy size conditions.
        """
        anatomy_size_idx = {
            "gallbladder": 0,
            "liver": 1,
            "stomach": 2,
            "pancreas": 3,
            "colon": 4,
            "lung tumor": 5,
            "pancreatic tumor": 6,
            "hepatic tumor": 7,
            "colon cancer primaries": 8,
            "bone lesion": 9,
        }
        provide_anatomy_size = [None for _ in range(10)]
        logging.info(f"controllable_anatomy_size: {controllable_anatomy_size}")
        for element in controllable_anatomy_size:
            anatomy_name, anatomy_size = element
            provide_anatomy_size[anatomy_size_idx[anatomy_name]] = anatomy_size

        with open(self.all_anatomy_size_conditions_json) as f:
            all_anatomy_size_conditions = json.load(f)

        # loop through the database and find closest combinations
        candidate_list = []
        for anatomy_size in all_anatomy_size_conditions:
            size = anatomy_size["organ_size"]
            diff = 0
            for db_size, provide_size in zip(size, provide_anatomy_size):
                if provide_size is None:
                    continue
                diff += abs(provide_size - db_size)
            candidate_list.append((size, diff))
        candidate_condition = sorted(candidate_list, key=lambda x: x[1])[0][0]

        # overwrite the anatomy size provided by users
        for element in controllable_anatomy_size:
            anatomy_name, anatomy_size = element
            candidate_condition[anatomy_size_idx[anatomy_name]] = anatomy_size

        return candidate_condition

    def prepare_one_mask_and_meta_info(self, anatomy_size_condition):
        """
        Prepare a single mask and its associated meta information.

        Args:
            anatomy_size_condition (list): Anatomy size conditions.

        Returns:
            tuple: A tuple containing the prepared mask and associated tensors.
        """
        combine_label_or = self.sample_one_mask(anatomy_size=anatomy_size_condition)
        # TODO: current mask generation model only can generate 256^3 volumes with 1.5 mm spacing.
        affine = torch.zeros((4, 4))
        affine[0, 0] = 1.5
        affine[1, 1] = 1.5
        affine[2, 2] = 1.5
        affine[3, 3] = 1.0  # dummy
        combine_label_or = MetaTensor(combine_label_or, affine=affine)
        combine_label_or = self.ensure_output_size_and_spacing(combine_label_or)

        top_region_index, bottom_region_index = get_body_region_index_from_mask(combine_label_or)

        spacing_tensor = torch.FloatTensor(self.spacing).unsqueeze(0).half().to(self.device) * 1e2
        top_region_index_tensor = torch.FloatTensor(top_region_index).unsqueeze(0).half().to(self.device) * 1e2
        bottom_region_index_tensor = torch.FloatTensor(bottom_region_index).unsqueeze(0).half().to(self.device) * 1e2

        return combine_label_or, top_region_index_tensor, bottom_region_index_tensor, spacing_tensor

    def sample_one_mask(self, anatomy_size):
        """
        Generate a single synthetic mask.

        Args:
            anatomy_size (list): Anatomy size specifications.

        Returns:
            torch.Tensor: The generated synthetic mask.
        """
        # generate one synthetic mask
        synthetic_mask = ldm_conditional_sample_one_mask(
            self.mask_generation_autoencoder,
            self.mask_generation_diffusion_unet,
            self.mask_generation_noise_scheduler,
            self.mask_generation_scale_factor,
            anatomy_size,
            self.device,
            self.mask_generation_latent_shape,
            label_dict_remap_json=self.label_dict_remap_json,
            num_inference_steps=self.mask_generation_num_inference_steps,
            autoencoder_sliding_window_infer_size=self.autoencoder_sliding_window_infer_size,
            autoencoder_sliding_window_infer_overlap=self.autoencoder_sliding_window_infer_overlap,
        )
        return synthetic_mask

    def ensure_output_size_and_spacing(self, labels, check_contains_target_labels=True):
        """
        Ensure the output mask has the correct size and spacing.

        Args:
            labels (torch.Tensor): Input label tensor.
            check_contains_target_labels (bool): Whether to check if the resampled mask contains target labels.

        Returns:
            torch.Tensor: Resampled label tensor.

        Raises:
            ValueError: If the resampled mask doesn't contain required class labels.
        """
        current_spacing = [labels.affine[0, 0], labels.affine[1, 1], labels.affine[2, 2]]
        current_shape = list(labels.squeeze().shape)

        need_resample = False
        # check spacing
        for i, j in zip(current_spacing, self.spacing):
            if i != j:
                need_resample = True
        # check output size
        for i, j in zip(current_shape, self.output_size):
            if i != j:
                need_resample = True
        # resample to target size and spacing
        if need_resample:
            logging.info("Resampling mask to target shape and spacing")
            logging.info(f"Resize Spacing: {current_spacing} -> {self.spacing}")
            logging.info(f"Output size: {current_shape} -> {self.output_size}")
            spacing = monai.transforms.Spacing(pixdim=tuple(self.spacing), mode="nearest")
            pad_crop = monai.transforms.ResizeWithPadOrCrop(spatial_size=tuple(self.output_size))
            labels = pad_crop(spacing(labels.squeeze(0))).unsqueeze(0).to(labels.dtype)

            contained_labels = torch.unique(labels)
            if check_contains_target_labels:
                # check if the resampled mask still contains those target labels
                for anatomy_label in self.anatomy_list:
                    if anatomy_label not in contained_labels:
                        raise ValueError(
                            f"Resampled mask does not contain required class labels {anatomy_label}. Please tune spacing and output size."
                        )
        return labels

    def read_mask_information(self, mask_file):
        """
        Read mask information from a file.

        Args:
            mask_file (str): Path to the mask file.

        Returns:
            tuple: A tuple containing the mask tensor and associated information.
        """
        val_data = self.val_transforms(mask_file)

        for key in ["pseudo_label", "spacing", "top_region_index", "bottom_region_index"]:
            if isinstance(val_data[key], torch.Tensor):
                val_data[key] = val_data[key].unsqueeze(0).to(self.device)
            else:
                val_data[key] = None

        return (
            val_data["pseudo_label"],
            val_data["top_region_index"],
            val_data["bottom_region_index"],
            val_data["spacing"],
        )

    def find_closest_masks(self, num_img):
        """
        Find the closest matching masks from the database.

        Args:
            num_img (int): Number of images to generate.

        Returns:
            list: List of closest matching mask candidates.

        Raises:
            ValueError: If suitable candidates cannot be found.
        """
        # first check the database based on anatomy list
        candidates = find_masks(
            self.body_region,
            self.anatomy_list,
            self.spacing,
            self.output_size,
            False,
            self.all_mask_files_json,
            self.data_root,
        )

        if len(candidates) < num_img:
            raise ValueError(f"candidate masks are less than {num_img}).")

        # loop through the database and find closest combinations
        new_candidates = []
        for c in candidates:
            diff = 0
            include_c = True
            for axis in range(3):
                if abs(c["dim"][axis]) <= self.output_size[axis] - 128:
                    # we cannot upsample the mask too much
                    include_c = False
                    break
                # check diff in FOV, major metric
                diff += abs((abs(c["dim"][axis] * c["spacing"][axis]) - self.output_size[axis] * self.spacing[axis]) / 10)
                # check diff in dim
                diff += abs((abs(c["dim"][axis]) - self.output_size[axis]) / 100)
                # check diff in spacing
                diff += abs(abs(c["spacing"][axis]) - self.spacing[axis])
            if include_c:
                new_candidates.append((c, diff))

        # choose top-2*num_img candidates (at least 5)
        num_candidates = max(self.max_try_time * num_img, 5)
        new_candidates = sorted(new_candidates, key=lambda x: x[1])

        final_candidates = []
        # check top-2*num_img candidates and update spacing after resampling
        for c, _ in new_candidates:
            c = self.resample_mask_check_organ_list(c)
            if c is not None:
                final_candidates.append(c)
            if len(final_candidates) >= num_candidates:
                break
        if len(final_candidates) == 0:
            raise ValueError("Cannot find body region with given organ list.")
        return final_candidates

    def resample_mask_check_organ_list(self, mask):
        """
        Resample mask and check if the resampled mask contains the required organ list.

        Args:
            mask (dict): input mask.

        Returns:
            dict: resampled mask. If None, means the resampled mask does not contain the required organ list

        Raises:
            ValueError: If suitable candidates cannot be found.
        """

        image_loader = monai.transforms.LoadImage(image_only=True, ensure_channel_first=True)
        label = image_loader(mask["pseudo_label"])
        try:
            label = self.ensure_output_size_and_spacing(label.unsqueeze(0))
        except ValueError as e:
            if "Resampled mask does not contain required class labels" in str(e):
                return None
            else:
                raise e
        # get region_index after resample
        top_region_index, bottom_region_index = get_body_region_index_from_mask(label)
        mask["top_region_index"] = top_region_index
        mask["bottom_region_index"] = bottom_region_index
        mask["spacing"] = self.spacing
        mask["dim"] = self.output_size
        return mask

    def quality_check_ct(self, image_data, label_data, perform_quality_check=True):
        """
        Perform a quality check on the generated image.
        Args:
            image_data (np.ndarray): The generated image.
            label_data (np.ndarray): The corresponding whole body mask.
        Returns:
            bool: True if the image passes the quality check, False otherwise.
        """
        if not perform_quality_check:
            return True
        outlier_results = is_outlier(self.median_statistics, image_data, label_data, self.label_int_dict)
        for label, result in outlier_results.items():
            if result.get("is_outlier", False):
                logging.info(
                    f"Generated image quality check for label '{label}' failed: median value {result['median_value']} is outside the acceptable range ({result['low_thresh']} - {result['high_thresh']})."
                )
                return False
        return True
