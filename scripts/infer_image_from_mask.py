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
**Mask-conditioned** image-from-condition inference module.

This file is the **mask-specific wrapper** around the modality-agnostic core
``run_controlnet_conditioned_image_dm`` in ``scripts/utils_infer.py``. The
split is intentional: when a future ControlNet is trained on a different
conditioning modality, that wrapper can live in a sibling file and reuse
the same core.

Two ways to use this file:

1. **As a library** — import ``ldm_conditional_sample_one_image_from_mask``
   (or its backward-compat alias ``ldm_conditional_sample_one_image``) and
   call it directly. The mask-specific preprocessing
   (``binarize_labels``, ``remove_tumors`` for CFG) and post-processing
   (``crop_img_body_mask``) live in this file; the inner loop is delegated
   to ``utils_infer.run_controlnet_conditioned_image_dm``.

2. **As a CLI** — run ``python -m scripts.infer_image_from_mask --mask
   /path/to/mask.nii.gz -t <network-config>.json -e <env-config>.json``
   to generate a CT/MR image from a user-provided mask file. The CLI
   loads checkpoints, validates the input mask, auto-resamples to a valid
   (dim, spacing) target, runs inference, and saves the output.

The user-provided mask must contain integer labels in the MAISI 132-class
vocabulary (see ``configs/label_dict.json``). Unknown label values are
warned about but passed through unchanged.

See ``skills/image-from-mask.md`` for the algorithm walkthrough.
"""

import argparse
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import monai
import torch
from monai.data import MetaTensor
from monai.transforms import SaveImage
from monai.utils import set_determinism

from .augmentation import remove_tumors
from .utils import binarize_labels
from .utils_infer import (
    build_conditioning_tensors,
    load_image_models,
    run_controlnet_conditioned_image_dm,
)


def ldm_conditional_sample_one_image_from_mask(
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
    autoencoder_sliding_window_infer_size=(96, 96, 96),
    autoencoder_sliding_window_infer_overlap=0.6667,
    cfg_guidance_scale_tumor=0,
):
    """
    Generate a CT/MR image from a **3D label mask** via the ControlNet-
    conditioned image LDM.

    This is the **mask-specific** wrapper around the modality-agnostic core
    ``run_controlnet_conditioned_image_dm``. It does three mask-specific things:

      1. Pre-process: ``binarize_labels`` converts the 1-channel integer mask
         to the 8-channel binary ControlNet conditioning tensor.
      2. CFG (when ``cfg_guidance_scale_tumor > 0``): builds a tumor-free
         unconditional counterpart via ``remove_tumors`` + ``binarize_labels``.
      3. Post-process: ``crop_img_body_mask`` regularizes background voxels
         to ``a_min`` (CT: -1000; MR: 0) using the mask.

    A future image-conditioned variant would live in a sibling module
    (e.g. ``scripts/infer_image_from_image.py``) and do its own preprocessing
    while reusing the same ``run_controlnet_conditioned_image_dm`` core.

    Returns ``(synthetic_image, combine_label)`` — the mask is returned for
    downstream filtering (e.g. ``filter_mask_with_organs``).
    """
    # modality_tensor can be scalar (single mask) or shape (B,) (batch infer);
    # collapse to a single int so `if` doesn't choke on a multi-element bool tensor.
    if modality_tensor is not None and int(modality_tensor.flatten()[0]) <= 7:
        a_min = -1000  # CT background floor
    else:
        a_min = 0  # MR background floor

    combine_label = combine_label_or.to(device)
    if output_size[0] != combine_label.shape[2] or output_size[1] != combine_label.shape[3] or output_size[2] != combine_label.shape[4]:
        logging.info(
            "output_size is not a desired value. Need to interpolate the mask to "
            "match with output_size. The result image will be very low quality."
        )
        combine_label = torch.nn.functional.interpolate(combine_label, size=output_size, mode="nearest")

    # ── Mask-specific pre-processing ───────────────────────────────────────────
    # NOTE (modality-specific): the next line converts mask → ControlNet
    # conditioning. A future image-conditioned ControlNet would replace this
    # with image normalization in its own wrapper module.
    controlnet_cond_tensor = binarize_labels(combine_label.as_tensor().long()).half()

    controlnet_uncond_tensor = None
    if cfg_guidance_scale_tumor > 0:
        # Mask-specific unconditional branch: same mask with tumors removed.
        combine_label_no_tumor = torch.nn.functional.interpolate(
            remove_tumors(combine_label.squeeze(0)).unsqueeze(0).float(),
            size=output_size,
            mode="nearest",
        ).to(combine_label.dtype)
        controlnet_uncond_tensor = binarize_labels(combine_label_no_tumor.as_tensor().long()).half()
        del combine_label_no_tumor

    # ── Modality-agnostic core ─────────────────────────────────────────────────
    synthetic_images = run_controlnet_conditioned_image_dm(
        autoencoder=autoencoder,
        diffusion_unet=diffusion_unet,
        controlnet=controlnet,
        noise_scheduler=noise_scheduler,
        scale_factor=scale_factor,
        device=device,
        controlnet_cond_tensor=controlnet_cond_tensor,
        spacing_tensor=spacing_tensor,
        latent_shape=latent_shape,
        output_size=output_size,
        noise_factor=noise_factor,
        top_region_index_tensor=top_region_index_tensor,
        bottom_region_index_tensor=bottom_region_index_tensor,
        modality_tensor=modality_tensor,
        num_inference_steps=num_inference_steps,
        autoencoder_sliding_window_infer_size=autoencoder_sliding_window_infer_size,
        autoencoder_sliding_window_infer_overlap=autoencoder_sliding_window_infer_overlap,
        # Wrapper-level "tumor CFG" maps to the generic CFG arg on the
        # modality-agnostic core. The semantic ("strengthen tumor") comes
        # from how we built controlnet_uncond_tensor above (tumor-free mask).
        cfg_guidance_scale=cfg_guidance_scale_tumor,
        controlnet_uncond_tensor=controlnet_uncond_tensor,
    )

    # ── Mask-specific post-processing ──────────────────────────────────────────
    # Regularize background HU using the mask: voxels where mask==0 → a_min.
    synthetic_images = crop_img_body_mask(synthetic_images, combine_label, a_min=a_min)
    return synthetic_images, combine_label


# Backward-compat alias — existing callers (LDMSampler, infer_image_from_mask_batch,
# notebooks) import the old name. Keep it pointing at the mask wrapper.
ldm_conditional_sample_one_image = ldm_conditional_sample_one_image_from_mask


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


# =============================================================================
# CLI entry point
# =============================================================================
#
# Generates a CT/MR image from a user-provided mask file. Loads the necessary
# checkpoints, validates + (optionally) resamples the mask, runs inference, and
# saves the output.
#
# Usage:
#   python -m scripts.infer_image_from_mask --mask /path/to/mask.nii.gz \
#       -t ./configs/config_network_rflow.json \
#       -e ./configs/environment_rflow-ct.json \
#       --modality 1 --output-dir ./output_user_mask
# =============================================================================


# Valid (dim, spacing) constraints — mirror check_input_ct in scripts.sample_mask.
_VALID_DIM_XY = (256, 384, 512)
_VALID_DIM_Z = (128, 256, 384, 512, 640, 768)
_VALID_SPACING_XY_RANGE = (0.5, 3.0)
_VALID_SPACING_Z_RANGE = (0.5, 5.0)

# MAISI 132-class label vocabulary — values that should appear in a valid mask.
_MAISI_VALID_LABELS = set(range(0, 133)) | {200}  # 0..132 + body=200


_USER_MASK_FORMAT_WARNING = """\

╔══════════════════════════════════════════════════════════════════════════════╗
║  USER-PROVIDED MASK FORMAT REQUIREMENTS                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

Your mask MUST follow the MAISI label format for the generated image to be
meaningful. The CLI will warn (not error) on most violations and try to fix
them, but the output quality depends on the mask matching these rules:

  1. File format     : NIfTI (.nii or .nii.gz), 1-channel, integer dtype
  2. Label vocabulary: MAISI 132-class (see configs/label_dict.json).
                       Common values:  0=background, 1=liver, 3=spleen,
                       4=pancreas, 5=right kidney, 14=left kidney,
                       28-32=lung lobes, 33-57=vertebrae, 132=airway,
                       200=body (special anomalous value)
  3. Shape (dim)     : dim[0]==dim[1], dim[0] in {256, 384, 512},
                       dim[2] in {128, 256, 384, 512, 640, 768}
  4. Spacing (mm)    : spacing[0]==spacing[1] in [0.5, 3.0],
                       spacing[2] in [0.5, 5.0]
  5. Orientation     : will be reoriented to RAS internally; any input OK

If your mask comes from NV-Segment-CTMR/TotalSegmentator with different label
values, you'll need to remap it first. See scripts/sample_mask.py::remap_labels
and configs/label_dict_124_to_132.json.

If shape/spacing don't match the constraints above, the CLI will auto-resample
to the nearest valid target — this degrades label boundary quality. Pre-
resampling your mask yourself to a valid target gives the best results.
"""


def _print_mask_format_warning() -> None:
    """Print the prominent format-requirements banner at startup."""
    print(_USER_MASK_FORMAT_WARNING, file=sys.stderr, flush=True)


def _suggest_valid_target(
    current_shape: tuple[int, int, int],
    current_spacing: tuple[float, float, float],
) -> tuple[tuple[int, int, int], tuple[float, float, float]]:
    """
    Snap a user's (shape, spacing) to the closest valid target the model accepts.

    Picks the closest valid dim per axis (so the FOV stays close to the user's),
    then picks a spacing that keeps the original FOV as best it can within the
    allowed range.
    """

    def _closest(value, allowed):
        return min(allowed, key=lambda v: abs(v - value))

    def _clip(value, lo, hi):
        return max(lo, min(hi, value))

    # Snap dim XY: dim[0]==dim[1], both in _VALID_DIM_XY
    dim_xy = _closest((current_shape[0] + current_shape[1]) / 2, _VALID_DIM_XY)
    dim_z = _closest(current_shape[2], _VALID_DIM_Z)

    # Match FOV: keep original_fov = original_dim × original_spacing roughly constant.
    original_fov_xy = (current_shape[0] * current_spacing[0] + current_shape[1] * current_spacing[1]) / 2
    original_fov_z = current_shape[2] * current_spacing[2]

    spacing_xy = _clip(original_fov_xy / dim_xy, *_VALID_SPACING_XY_RANGE)
    spacing_z = _clip(original_fov_z / dim_z, *_VALID_SPACING_Z_RANGE)

    return (dim_xy, dim_xy, dim_z), (spacing_xy, spacing_xy, spacing_z)


def _is_valid_target(shape, spacing) -> bool:
    return (
        shape[0] == shape[1]
        and shape[0] in _VALID_DIM_XY
        and shape[2] in _VALID_DIM_Z
        and abs(spacing[0] - spacing[1]) < 1e-6
        and _VALID_SPACING_XY_RANGE[0] <= spacing[0] <= _VALID_SPACING_XY_RANGE[1]
        and _VALID_SPACING_Z_RANGE[0] <= spacing[2] <= _VALID_SPACING_Z_RANGE[1]
    )


def validate_user_mask(label_path: str | os.PathLike) -> dict:
    """
    Load + validate a user-provided mask NIfTI.

    Returns a dict with the resampled MetaTensor mask + the chosen (shape, spacing)
    target. Resamples if needed; warns about unknown label values; warns about
    shape/spacing mismatches.
    """
    label_path = Path(label_path).expanduser().resolve()
    if not label_path.exists():
        raise FileNotFoundError(f"Mask file not found: {label_path}")

    # Load as a MetaTensor with RAS orientation + integer dtype.
    transforms = monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=["label"], image_only=True),
            monai.transforms.EnsureChannelFirstd(keys=["label"]),
            monai.transforms.Orientationd(keys=["label"], axcodes="RAS"),
            monai.transforms.EnsureTyped(keys=["label"], dtype=torch.long),
        ]
    )
    data = transforms({"label": str(label_path)})
    label = data["label"]  # shape (1, H, W, D)

    current_shape = tuple(label.shape[1:])
    # L2 norm of each affine column gives true voxel spacing regardless of any
    # residual rotation in the affine — Orientationd reorders/flips axes but
    # does not strip rotation for oblique acquisitions, so affine[i, i] alone
    # underestimates spacing for oblique scans.
    current_spacing = tuple(float(torch.norm(label.affine[:3, i])) for i in range(3))
    print(f"[validate] loaded {label_path.name}: shape={current_shape}, spacing={current_spacing}", file=sys.stderr)

    # Check label vocabulary
    unique_labels = torch.unique(label).tolist()
    unique_label_ints = {int(v) for v in unique_labels}
    unknown = sorted(v for v in unique_label_ints if v not in _MAISI_VALID_LABELS)
    if unknown:
        print(
            f"[validate] ⚠️  mask contains {len(unknown)} label value(s) outside the MAISI 132-class vocabulary: "
            f"{unknown[:10]}{'...' if len(unknown) > 10 else ''}",
            file=sys.stderr,
        )
        print("[validate]    These will be passed through unchanged — generated image quality on unknown labels is unpredictable.", file=sys.stderr)
    else:
        print(f"[validate] ✓ all {len(unique_labels)} unique label values are in the MAISI 132-class vocabulary.", file=sys.stderr)

    # The released CT ControlNet expects label 200 (body envelope) for every body
    # voxel not labeled with a specific organ. Missing it is the single most common
    # mistake — see scripts.utils.add_body_envelope.
    if 200 not in unique_label_ints:
        print(
            "[validate] ⚠️  mask does NOT contain label 200 (body envelope). The released CT "
            "ControlNet expects label 200 on every body voxel not assigned a specific organ. "
            "Generated image quality will be poor without it — run scripts.utils.add_body_envelope "
            "on the mask first. See skills/infer_image-from-mask.md.",
            file=sys.stderr,
        )

    # Check shape + spacing constraints
    if _is_valid_target(current_shape, current_spacing):
        print("[validate] ✓ shape + spacing already match a valid target — no resample needed.", file=sys.stderr)
        target_shape = current_shape
        target_spacing = current_spacing
        resampled = label
    else:
        target_shape, target_spacing = _suggest_valid_target(current_shape, current_spacing)
        print(
            f"[validate] ⚠️  shape/spacing not in the model's valid set. Auto-resampling:\n"
            f"           shape    {current_shape} → {target_shape}\n"
            f"           spacing  {current_spacing} → {target_spacing}\n"
            f"[validate]    Auto-resampling degrades label boundaries; pre-resampling yourself gives better results.",
            file=sys.stderr,
        )
        resampled = monai.transforms.Spacing(pixdim=tuple(target_spacing), mode="nearest")(label)
        resampled = monai.transforms.ResizeWithPadOrCrop(spatial_size=tuple(target_shape))(resampled)

    # Add batch dim: (1, H, W, D) → (1, 1, H, W, D)
    if resampled.ndim == 4:
        resampled = resampled.unsqueeze(0)

    return {
        "label": resampled,
        "shape": tuple(target_shape),
        "spacing": tuple(target_spacing),
        "source_path": str(label_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.infer_image_from_mask",
        description=(
            "Generate a CT/MR image from a user-provided 3D label mask via the "
            "ControlNet-conditioned image LDM. See the prominent format-requirements "
            "banner that prints at startup."
        ),
    )
    parser.add_argument("--mask", required=True, help="Path to the user-provided mask NIfTI file.")
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
        help=(
            "Config json file that stores inference hyper-parameters (e.g. "
            "./configs/config_infer.json). Source of modality, num_inference_steps, "
            "autoencoder_sliding_window_infer_size/overlap, cfg_guidance_scale_tumor."
        ),
    )
    parser.add_argument("--random-seed", type=int, default=0)

    args = parser.parse_args()
    set_determinism(seed=args.random_seed)

    _print_mask_format_warning()

    # ── Resolve config paths ────────────────────────────────────────────────
    for p in (args.mask, args.environment_file, args.config_file, args.inference_file):
        if not Path(p).exists():
            print(f"[error] file not found: {p}", file=sys.stderr)
            return 2

    # ── Step 1: validate + load the user's mask ─────────────────────────────
    print("[step 1/4] validating user mask...", file=sys.stderr)
    mask_info = validate_user_mask(args.mask)

    # ── Step 2: load models ─────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[step 2/4] loading models on {device}...", file=sys.stderr)
    from .diff_model_setting import load_config

    # cfg now carries every config key from env + inference + network configs:
    # output_dir, modality, num_inference_steps,
    # autoencoder_sliding_window_infer_size/overlap, cfg_guidance_scale_tumor, etc.
    cfg = load_config(args.environment_file, args.inference_file, args.config_file)
    # Back-compat: accept the legacy key for one release.
    if hasattr(cfg, "cfg_guidance_scale") and not hasattr(cfg, "cfg_guidance_scale_tumor"):
        warnings.warn(
            "`cfg_guidance_scale` in image-from-mask configs is deprecated; "
            "rename it to `cfg_guidance_scale_tumor`.",
            DeprecationWarning,
            stacklevel=2,
        )
        cfg.cfg_guidance_scale_tumor = cfg.cfg_guidance_scale
    autoencoder, diffusion_unet, controlnet, scale_factor, noise_scheduler = load_image_models(cfg, device)

    include_body_region = diffusion_unet.include_top_region_index_input
    include_modality = diffusion_unet.num_class_embeds is not None
    if include_modality:
        print(f"[step 2/4] image DM uses modality conditioning; passing modality={cfg.modality}.", file=sys.stderr)
    if include_body_region:
        print("[step 2/4] image DM uses body-region conditioning; will derive top/bottom_region_index from mask.", file=sys.stderr)

    # ── Step 3: prepare conditioning tensors ────────────────────────────────
    label = mask_info["label"].to(device)
    shape = mask_info["shape"]
    spacing = mask_info["spacing"]
    latent_channels = cfg.latent_channels
    latent_shape = (latent_channels, shape[0] // 4, shape[1] // 4, shape[2] // 4)
    print(f"[step 3/4] preparing conditioning tensors (latent_shape={latent_shape})...", file=sys.stderr)

    spacing_tensor, top_region_index_tensor, bottom_region_index_tensor, modality_tensor = build_conditioning_tensors(
        label,
        spacing,
        cfg.modality,
        include_body_region,
        device,
    )

    # ── Step 4: inference ───────────────────────────────────────────────────
    from monai.networks.schedulers import DDPMScheduler

    num_inference_steps = int(cfg.num_inference_steps)
    if isinstance(noise_scheduler, DDPMScheduler) and num_inference_steps != 1000:
        print(
            f"[warn] DDPM scheduler typically requires num_inference_steps=1000; got {num_inference_steps}. Output quality not guaranteed.",
            file=sys.stderr,
        )

    print(f"[step 4/4] running inference (steps={num_inference_steps}, cfg_tumor={cfg.cfg_guidance_scale_tumor})...", file=sys.stderr)
    synthetic_image, returned_label = ldm_conditional_sample_one_image(
        autoencoder=autoencoder,
        diffusion_unet=diffusion_unet,
        controlnet=controlnet,
        noise_scheduler=noise_scheduler,
        scale_factor=scale_factor,
        device=device,
        combine_label_or=label,
        spacing_tensor=spacing_tensor,
        latent_shape=latent_shape,
        output_size=shape,
        noise_factor=1.0,
        top_region_index_tensor=top_region_index_tensor,
        bottom_region_index_tensor=bottom_region_index_tensor,
        modality_tensor=modality_tensor,
        num_inference_steps=num_inference_steps,
        autoencoder_sliding_window_infer_size=cfg.autoencoder_sliding_window_infer_size,
        autoencoder_sliding_window_infer_overlap=cfg.autoencoder_sliding_window_infer_overlap,
        cfg_guidance_scale_tumor=cfg.cfg_guidance_scale_tumor,
    )

    # ── Save output ─────────────────────────────────────────────────────────
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    src_stem = Path(args.mask).name.replace(".nii.gz", "").replace(".nii", "")

    label_metatensor = returned_label.cpu().detach()
    if not hasattr(label_metatensor, "meta") or "filename_or_obj" not in label_metatensor.meta:
        label_metatensor = MetaTensor(label_metatensor, meta=getattr(label, "meta", {}))
    label_metatensor.meta["filename_or_obj"] = f"{src_stem}.nii.gz"

    synthetic_metatensor = MetaTensor(synthetic_image.squeeze(0), meta=label_metatensor.meta)

    img_saver = SaveImage(
        output_dir=str(output_dir),
        output_postfix=f"{timestamp}_image",
        output_ext=".nii.gz",
        separate_folder=False,
    )
    img_saver(synthetic_metatensor)

    out_path = output_dir / f"{src_stem}_{timestamp}_image.nii.gz"
    print(f"\n[done] generated image saved to: {out_path}", file=sys.stderr)
    print(f"[done] source mask was: {args.mask}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
