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

import numpy as np
import torch
import torch.nn.functional as F
from monai.transforms import Rand3DElastic, RandAffine, RandZoom
from monai.utils import ensure_tuple_rep
from torch import Tensor

from .utils import dilate_one_img, erode_one_img


def erode3d(input_tensor, erosion=3):
    # Define the structuring element
    erosion = ensure_tuple_rep(erosion, 3)
    structuring_element = torch.ones(1, 1, erosion[0], erosion[1], erosion[2]).to(input_tensor.device)

    # Pad the input tensor to handle border pixels
    input_padded = F.pad(
        input_tensor.float().unsqueeze(0).unsqueeze(0),
        (erosion[0] // 2, erosion[0] // 2, erosion[1] // 2, erosion[1] // 2, erosion[2] // 2, erosion[2] // 2),
        mode="constant",
        value=1.0,
    )

    # Apply erosion operation
    output = F.conv3d(input_padded, structuring_element, padding=0)

    # Set output values based on the minimum value within the structuring element
    output = torch.where(output == torch.sum(structuring_element), 1.0, 0.0)

    return output.squeeze(0).squeeze(0)


def dilate3d(input_tensor, erosion=3):
    # Define the structuring element
    erosion = ensure_tuple_rep(erosion, 3)
    structuring_element = torch.ones(1, 1, erosion[0], erosion[1], erosion[2]).to(input_tensor.device)

    # Pad the input tensor to handle border pixels
    input_padded = F.pad(
        input_tensor.float().unsqueeze(0).unsqueeze(0),
        (erosion[0] // 2, erosion[0] // 2, erosion[1] // 2, erosion[1] // 2, erosion[2] // 2, erosion[2] // 2),
        mode="constant",
        value=1.0,
    )

    # Apply erosion operation
    output = F.conv3d(input_padded, structuring_element, padding=0)

    # Set output values based on the minimum value within the structuring element
    output = torch.where(output > 0, 1.0, 0.0)

    return output.squeeze(0).squeeze(0)


def augmentation_tumor_bone(pt_nda, output_size, random_seed=None):
    volume = pt_nda.squeeze(0)
    real_l_volume_ = torch.zeros_like(volume)
    real_l_volume_[volume == 128] = 1
    real_l_volume_ = real_l_volume_.to(torch.uint8)

    elastic = RandAffine(
        mode="nearest",
        prob=1.0,
        translate_range=(5, 5, 0),
        rotate_range=(0, 0, 0.1),
        scale_range=(0.15, 0.15, 0),
        padding_mode="zeros",
    )
    elastic.set_random_state(seed=random_seed)

    tumor_szie = torch.sum((real_l_volume_ > 0).float())
    ###########################
    # remove pred in pseudo_label in real lesion region
    volume[real_l_volume_ > 0] = 200
    ###########################
    if tumor_szie > 0:
        # get organ mask
        organ_mask = (
            torch.logical_and(33 <= volume, volume <= 56).float()
            + torch.logical_and(63 <= volume, volume <= 97).float()
            + (volume == 127).float()
            + (volume == 114).float()
            + real_l_volume_
        )
        organ_mask = (organ_mask > 0).float()
        cnt = 0
        while True:
            threshold = 0.8 if cnt < 40 else 0.75
            real_l_volume = real_l_volume_
            # random distor mask
            distored_mask = elastic((real_l_volume > 0).cuda(), spatial_size=tuple(output_size)).as_tensor()
            real_l_volume = distored_mask * organ_mask
            cnt += 1
            print(torch.sum(real_l_volume), "|", tumor_szie * threshold)
            if torch.sum(real_l_volume) >= tumor_szie * threshold:
                real_l_volume = dilate3d(real_l_volume.squeeze(0), erosion=5)
                real_l_volume = erode3d(real_l_volume, erosion=5).unsqueeze(0).to(torch.uint8)
                break
    else:
        real_l_volume = real_l_volume_

    volume[real_l_volume == 1] = 128

    pt_nda = volume.unsqueeze(0)
    return pt_nda


def augmentation_tumor_liver(pt_nda, output_size, random_seed=None):
    volume = pt_nda.squeeze(0)
    real_l_volume_ = torch.zeros_like(volume)
    real_l_volume_[volume == 1] = 1
    real_l_volume_[volume == 26] = 2
    real_l_volume_ = real_l_volume_.to(torch.uint8)

    elastic = Rand3DElastic(
        mode="nearest",
        prob=1.0,
        sigma_range=(5, 8),
        magnitude_range=(100, 200),
        translate_range=(10, 10, 10),
        rotate_range=(np.pi / 36, np.pi / 36, np.pi / 36),
        scale_range=(0.2, 0.2, 0.2),
        padding_mode="zeros",
    )
    elastic.set_random_state(seed=random_seed)

    tumor_szie = torch.sum(real_l_volume_ == 2)
    ###########################
    # remove pred  organ labels
    volume[volume == 1] = 0
    volume[volume == 26] = 0
    # before move tumor maks, full the original location by organ labels
    volume[real_l_volume_ == 1] = 1
    volume[real_l_volume_ == 2] = 1
    ###########################
    while True:
        real_l_volume = real_l_volume_
        # random distor mask
        real_l_volume = elastic((real_l_volume == 2).cuda(), spatial_size=tuple(output_size)).as_tensor()
        # get organ mask
        organ_mask = (real_l_volume_ == 1).float() + (real_l_volume_ == 2).float()

        organ_mask = dilate3d(organ_mask.squeeze(0), erosion=5)
        organ_mask = erode3d(organ_mask, erosion=5).unsqueeze(0)
        real_l_volume = real_l_volume * organ_mask
        print(torch.sum(real_l_volume), "|", tumor_szie * 0.80)
        if torch.sum(real_l_volume) >= tumor_szie * 0.80:
            real_l_volume = dilate3d(real_l_volume.squeeze(0), erosion=5)
            real_l_volume = erode3d(real_l_volume, erosion=5).unsqueeze(0)
            break

    volume[real_l_volume == 1] = 26

    pt_nda = volume.unsqueeze(0)
    return pt_nda


def augmentation_tumor_lung(pt_nda, output_size, random_seed=None):
    volume = pt_nda.squeeze(0)
    real_l_volume_ = torch.zeros_like(volume)
    real_l_volume_[volume == 23] = 1
    real_l_volume_ = real_l_volume_.to(torch.uint8)

    elastic = Rand3DElastic(
        mode="nearest",
        prob=1.0,
        sigma_range=(5, 8),
        magnitude_range=(100, 200),
        translate_range=(20, 20, 20),
        rotate_range=(np.pi / 36, np.pi / 36, np.pi),
        scale_range=(0.15, 0.15, 0.15),
        padding_mode="zeros",
    )
    elastic.set_random_state(seed=random_seed)

    tumor_szie = torch.sum(real_l_volume_)
    # before move lung tumor maks, full the original location by lung labels
    new_real_l_volume_ = dilate3d(real_l_volume_.squeeze(0), erosion=3)
    new_real_l_volume_ = new_real_l_volume_.unsqueeze(0)
    new_real_l_volume_[real_l_volume_ > 0] = 0
    new_real_l_volume_[volume < 28] = 0
    new_real_l_volume_[volume > 32] = 0
    tmp = volume[(volume * new_real_l_volume_).nonzero(as_tuple=True)].view(-1)

    mode = torch.mode(tmp, 0)[0].item()
    print(mode)
    assert 28 <= mode <= 32
    volume[real_l_volume_.bool()] = mode
    ###########################
    if tumor_szie > 0:
        # aug
        while True:
            real_l_volume = real_l_volume_.cpu().contiguous()
            # random distor mask
            real_l_volume = elastic(real_l_volume, spatial_size=tuple(output_size)).as_tensor().cuda()
            # get lung mask v2 (133 order)
            lung_mask = (
                (volume == 28).float()
                + (volume == 29).float()
                + (volume == 30).float()
                + (volume == 31).float()
                + (volume == 32).float()
            )

            lung_mask = dilate3d(lung_mask.squeeze(0), erosion=5)
            lung_mask = erode3d(lung_mask, erosion=5).unsqueeze(0)
            real_l_volume = real_l_volume * lung_mask
            print(torch.sum(real_l_volume), "|", tumor_szie * 0.85)
            if torch.sum(real_l_volume) >= tumor_szie * 0.85:
                real_l_volume = dilate3d(real_l_volume.squeeze(0), erosion=5)
                real_l_volume = erode3d(real_l_volume, erosion=5).unsqueeze(0).to(torch.uint8)
                break
    else:
        real_l_volume = real_l_volume_

    volume[real_l_volume == 1] = 23

    pt_nda = volume.unsqueeze(0)
    return pt_nda


def augmentation_tumor_pancreas(pt_nda, output_size, random_seed=None):
    volume = pt_nda.squeeze(0)
    real_l_volume_ = torch.zeros_like(volume)
    real_l_volume_[volume == 4] = 1
    real_l_volume_[volume == 24] = 2
    real_l_volume_ = real_l_volume_.to(torch.uint8)

    elastic = Rand3DElastic(
        mode="nearest",
        prob=1.0,
        sigma_range=(5, 8),
        magnitude_range=(100, 200),
        translate_range=(15, 15, 15),
        rotate_range=(np.pi / 36, np.pi / 36, np.pi / 36),
        scale_range=(0.1, 0.1, 0.1),
        padding_mode="zeros",
    )
    elastic.set_random_state(seed=random_seed)

    tumor_szie = torch.sum(real_l_volume_ == 2)
    ###########################
    # remove pred  organ labels
    volume[volume == 24] = 0
    volume[volume == 4] = 0
    # before move tumor maks, full the original location by organ labels
    volume[real_l_volume_ == 1] = 4
    volume[real_l_volume_ == 2] = 4
    ###########################
    while True:
        real_l_volume = real_l_volume_
        # random distor mask
        real_l_volume = elastic((real_l_volume == 2).cuda(), spatial_size=tuple(output_size)).as_tensor()
        # get organ mask
        organ_mask = (real_l_volume_ == 1).float() + (real_l_volume_ == 2).float()

        organ_mask = dilate3d(organ_mask.squeeze(0), erosion=5)
        organ_mask = erode3d(organ_mask, erosion=5).unsqueeze(0)
        real_l_volume = real_l_volume * organ_mask
        print(torch.sum(real_l_volume), "|", tumor_szie * 0.80)
        if torch.sum(real_l_volume) >= tumor_szie * 0.80:
            real_l_volume = dilate3d(real_l_volume.squeeze(0), erosion=5)
            real_l_volume = erode3d(real_l_volume, erosion=5).unsqueeze(0)
            break

    volume[real_l_volume == 1] = 24

    pt_nda = volume.unsqueeze(0)
    return pt_nda


def augmentation_tumor_colon(pt_nda, output_size, random_seed=None):
    volume = pt_nda.squeeze(0)
    real_l_volume_ = torch.zeros_like(volume)
    real_l_volume_[volume == 27] = 1
    real_l_volume_ = real_l_volume_.to(torch.uint8)

    elastic = Rand3DElastic(
        mode="nearest",
        prob=1.0,
        sigma_range=(5, 8),
        magnitude_range=(100, 200),
        translate_range=(5, 5, 5),
        rotate_range=(np.pi / 36, np.pi / 36, np.pi / 36),
        scale_range=(0.1, 0.1, 0.1),
        padding_mode="zeros",
    )
    elastic.set_random_state(seed=random_seed)

    tumor_szie = torch.sum(real_l_volume_)
    ###########################
    # before move tumor maks, full the original location by organ labels
    volume[real_l_volume_.bool()] = 62
    ###########################
    if tumor_szie > 0:
        # get organ mask
        organ_mask = (volume == 62).float()
        organ_mask = dilate3d(organ_mask.squeeze(0), erosion=5)
        organ_mask = erode3d(organ_mask, erosion=5).unsqueeze(0)
        #         cnt = 0
        cnt = 0
        while True:
            threshold = 0.8
            real_l_volume = real_l_volume_
            if cnt < 20:
                # random distor mask
                distored_mask = elastic((real_l_volume == 1).cuda(), spatial_size=tuple(output_size)).as_tensor()
                real_l_volume = distored_mask * organ_mask
            elif 20 <= cnt < 40:
                threshold = 0.75
            else:
                break

            real_l_volume = real_l_volume * organ_mask
            print(torch.sum(real_l_volume), "|", tumor_szie * threshold)
            cnt += 1
            if torch.sum(real_l_volume) >= tumor_szie * threshold:
                real_l_volume = dilate3d(real_l_volume.squeeze(0), erosion=5)
                real_l_volume = erode3d(real_l_volume, erosion=5).unsqueeze(0).to(torch.uint8)
                break
    else:
        real_l_volume = real_l_volume_
    #     break
    volume[real_l_volume == 1] = 27

    pt_nda = volume.unsqueeze(0)
    return pt_nda


def augmentation_body(pt_nda, random_seed=None):
    volume = pt_nda.squeeze(0)

    zoom = RandZoom(min_zoom=0.99, max_zoom=1.01, mode="nearest", align_corners=None, prob=1.0)
    zoom.set_random_state(seed=random_seed)

    volume = zoom(volume)

    pt_nda = volume.unsqueeze(0)
    return pt_nda

def augmentation_tumor_only(tumor_mask_: Tensor, organ_mask: Tensor, aug_transform, spatial_size: tuple[int, int, int] | int | None = None, tumor_label: int = 1, min_tumor_size_ratio=0.8) -> Tensor:
    """
    tumor augmentation.

    Args:
        tumor_mask: input 3D tumor mask, [1,H,W,D] torch tensor.  
        organ_mask: input 3D tumor mask, [1,H,W,D] torch tensor, binary mask.  
        aug_transform: tumor augmentation transform       
        spatial_size: output image spatial size, used in random transform.
                      If not defined, will use (H,W,D). If some components are non-positive values,
                      the transform will use the corresponding components of whole_mask size.
                      For example, spatial_size=(128, 128, -1) will be adapted to (128, 128, 64)
                      if the third spatial dimension size of whole_mask is 64.
        tumor_label: e.g., it should be 2 if label 1 is organ, label 2 is tumor
        min_tumor_size_ratio: min tumor size after aug

    Return:
        augmented mask, with shape of spatial_size and data type as whole_mask.
        

    Example:

        .. code-block:: python

            # define a tumor mask
            tumor_mask = torch.zeros([1,128,128,128])
            tumor_mask[0, 90:110, 90:110, 90:110]=1
            tumor_mask[0, 97:103, 97:103, 97:103]=2
    """
    # Initialize binary tumor mask
    device = tumor_mask_.device

    tumor_region_binary_mask = torch.isin(tumor_mask_, torch.tensor(tumor_label, device=tumor_mask_.device)).long()
    tumor_size = torch.sum(tumor_region_binary_mask)
    ###########################
    if tumor_size > 0:
        count = 0
        # get organ mask
        organ_mask = dilate_one_img(organ_mask.squeeze(0), filter_size=5, pad_value=1.0)
        organ_mask = erode_one_img(organ_mask, filter_size=5, pad_value=1.0).unsqueeze(0)
        while True:
            tumor_mask = tumor_mask_
            # apply random augmentation to tumor region only, excluding organ basis
            augmented_mask = aug_transform(tumor_mask_*tumor_region_binary_mask, spatial_size=spatial_size).as_tensor()
            # generate final tumor mask
            count += 1
            tumor_mask = finalize_tumor_mask(augmented_mask, organ_mask, tumor_size * min_tumor_size_ratio)
            if tumor_mask is not None:
                break
            if count > MAX_COUNT:
                raise ValueError("Please check if tumor is inside organ.")
    else:
        tumor_mask = tumor_mask_

    return tumor_mask

def finalize_tumor_mask(augmented_mask: Tensor, organ_mask: Tensor, threshold_tumor_size: float):
    """
    Try to generate the final tumor mask by combining the augmented tumor mask and organ mask.
    Need to make sure tumor is inside of organ and is larger than threshold_tumor_size.

    Args:
        augmented_mask: input 3D binary tumor mask, [1,H,W,D] torch tensor.
        organ_mask: input 3D binary organ mask, [1,H,W,D] torch tensor.
        threshold_tumor_size: threshold tumor size, float

    Return:
        tumor_mask, [H,W,D] torch tensor; or None if the size did not qualify
    """
    tumor_mask = augmented_mask * organ_mask # might not be binary for multi-type tumor map
    if torch.sum(tumor_mask) >= threshold_tumor_size:
        label_list = torch.unique(tumor_mask.long())
        if len(label_list)==2:            
            tumor_mask = dilate_one_img(tumor_mask.squeeze(0), filter_size=5, pad_value=1.0)
            tumor_mask = erode_one_img(tumor_mask, filter_size=5, pad_value=1.0).unsqueeze(0).to(torch.uint8)
            tumor_mask[tumor_mask>0] = torch.max(label_list)
        return tumor_mask
    else:
        return None
        
def augmentation(pt_nda, output_size, random_seed=None):
    label_list = torch.unique(pt_nda)
    label_list = list(label_list.cpu().numpy())

    if 128 in label_list:
        print("augmenting bone lesion/tumor")
        pt_nda = augmentation_tumor_bone(pt_nda, output_size, random_seed)
    elif 26 in label_list:
        print("augmenting liver tumor")
        pt_nda = augmentation_tumor_liver(pt_nda, output_size, random_seed)
    elif 23 in label_list:
        print("augmenting lung tumor")
        pt_nda = augmentation_tumor_lung(pt_nda, output_size, random_seed)
    elif 24 in label_list:
        print("augmenting pancreas tumor")
        pt_nda = augmentation_tumor_pancreas(pt_nda, output_size, random_seed)
    elif 27 in label_list:
        print("augmenting colon tumor")
        pt_nda = augmentation_tumor_colon(pt_nda, output_size, random_seed)
    elif 401 in label_list or 402 in label_list or 403 in label_list:
        print("augmenting brats tumor")
        tumor_label = [401,402,403]
        elastic_tumor = Rand3DElastic(
            mode="nearest",
            prob=1.0,
            sigma_range=(5, 8),
            magnitude_range=(100, 200),
            translate_range=(3, 3, 3),
            rotate_range=(np.pi / 90, np.pi / 90, np.pi / 90),
            scale_range=(0.2, 0.2, 0.2),
            padding_mode="zeros",
        )
        elastic_tumor.set_random_state(seed=random_seed)
        volume = pt_nda.squeeze(0)
        organ_mask_ = (volume>0).long()
        tumor_mask_orig = torch.isin(volume, torch.tensor(tumor_label, device=volume.device)).long()
        tumor_mask = augmentation_tumor_only(volume, organ_mask_, elastic_tumor, output_size, tumor_label, 0.8).long()
        volume[tumor_mask_orig>0] = 22
        m = tumor_mask > 0
        volume[m] = tumor_mask[m]
        pt_nda = volume.unsqueeze(0)
    else:
        print("augmenting body")
        pt_nda = augmentation_body(pt_nda, random_seed)

    return pt_nda


def remove_tumors(orig_labels,pseudo_labels=None):
    """
    Replace tumor-related class ids with organ ids or pseudo labels.

    The function:
    - Maps tumor labels (e.g. hepatic, pancreatic, colon) to their organ counterparts.
    - Replaces ambiguous lesion regions with pseudo labels.
    
    Args:
        orig_labels (torch.Tensor): Original ground-truth labels.
        pseudo_labels (torch.Tensor): Pseudo labels used for replacement for tumor region, same size with orig_labels.

    Returns:
        torch.Tensor: Modified labels with tumor regions reassigned.
    """
    # hepatic tumor->liver, pancreatic tumor->pancreas, colon primaries timor->colon, kidney cyst-> kidney
    if len(orig_labels.shape) not in [3,4]:
        raise ValueError(f"input has to be 3D/4D, [1,X,Y,Z] or [1,X,Y]. Yet got {orig_labels.shape}.")
    x = remap_labels(
        orig_labels, 
        {26: 1, 24: 4, 27: 62, 116:14, 117:5}
    ) 

    if pseudo_labels!= None:
        # replace with pseudo_labels, for lung tumor, bone lesion, brain tumors 
        for lesion_id in [23,128,401,402,403,176]:  
            mask = (x == lesion_id)
            if mask.any():
                x[mask] = pseudo_labels[mask]
    else:    
        # Replace lesion-like classes by pseudo labels (explicit and ordered)
        # replace lung tumor with majority vote of its immediate neighborhood
        x = remove_tumors_majority_vote( 
            (x == 23),
            x,
            organ_label_lists=(28, 29, 30, 31, 32),
        )
        # replace brain tumors->brain
        x = remap_labels(
            x, 
            {401:22, 402:22, 403:22, 176:22}
        )
    return x

def remove_tumors_majority_vote(
    tumor_mask_: Tensor,
    volume: Tensor,
    organ_label_lists=(28, 29, 30, 31, 32),
):
    """
        Replace tumor voxels in a segmentation volume with the majority organ label 
        from their immediate neighborhood.

        Steps:
        1. Dilate the tumor mask with a small kernel to get a surrounding "ring."
        2. Remove the original tumor voxels from that dilation (keeping only the ring).
        3. Restrict the ring to voxels whose labels are in `organ_label_lists`.
        4. Extract labels from that ring and compute the majority vote (mode).
        - If the ring is empty, fall back to the most frequent organ label 
            in the entire volume.
        5. Overwrite the original tumor region in `volume` with the chosen organ label.

        Args:
            tumor_mask_ (Tensor): Binary (0/1) tumor mask of shape [1, D, H, W] or [1, ...].
            volume (Tensor): Segmentation label map of shape [D, H, W] or [1, D, H, W],
                            containing integer organ labels.
            organ_label_lists (iterable): Labels considered valid "organ" labels 
                                        to fill the tumor region (default [28–32]).

        Returns:
            Tensor: A copy of `volume` where tumor voxels are replaced with the 
                    majority-vote organ label.

        Notes:
            - `volume` is expected to contain integer labels, not raw intensities.
            - `dilate_one_img` should be a morphological dilation that accepts
            a single-channel binary tensor and returns the dilated mask.
        """
    # 1) Build a ring: dilate tumor, then remove original tumor voxels
    dil = dilate_one_img(tumor_mask_.squeeze(0), filter_size=3, pad_value=1.0).unsqueeze(0)
    ring = dil.bool() & (~tumor_mask_.bool())

    # 2) Keep only organ labels in the ring
    organ_set = torch.tensor(organ_label_lists, device=volume.device, dtype=volume.dtype)
    ring = ring & torch.isin(volume, organ_set)

    # 3) Majority vote from the ring (fallback if empty)
    tmp = volume[ring]
    if tmp.numel() == 0:
        # Fallback: pick the most common organ label in the whole volume
        counts = torch.stack([(volume == lbl).sum() for lbl in organ_set])
        mode = organ_set[counts.argmax()].item()
    else:
        mode = torch.mode(tmp, 0)[0].item()

    # 4) Fill original tumor with the chosen organ label
    out = volume.clone()
    out[tumor_mask_.bool()] = mode
    return out

def remap_labels(x, mapping: dict[int,int]):
    """
    Remap integer labels in a tensor.

    Args:
        x (torch.Tensor): Input label tensor.
        mapping (dict[int,int]): Mapping from old label ids to new label ids.

    Returns:
        torch.Tensor: A copy of `x` with values replaced according to mapping.
    """
    out = x.clone()
    for old, new in mapping.items():
        out[x == old] = new
    return out