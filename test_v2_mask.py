"""
Standalone test for the v2 mask diffusion model.
Tests that ldm_conditional_sample_one_mask runs end-to-end with:
  - DiffusionModelUNetMaisiAdaGN
  - RFlowScheduler
  - 19-d conditioning (anatomy + demographics)
  - CFG (guidance_scale=2.0)

Run inside the git repo:
    python test_v2_mask.py
"""

import logging
import os
import sys
import time

import nibabel as nib
import numpy as np
import torch

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="[%(asctime)s.%(msecs)03d][%(levelname)5s] %(message)s",
    datefmt="%H:%M:%S",
)

ROOT = "/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/users/canz/projects"
MASK_AE_PATH = f"{ROOT}/NV-Generate-BodyMask/model_weights/mask_generation_autoencoder.pt"
MASK_DM_PATH = f"{ROOT}/NV-Generate-BodyMask/model_weights/mask_generation_diffusion_unet_v2.pt"
REMAP_JSON = f"{ROOT}/NV-Generate-BodyMask_git/configs/label_dict_124_to_132.json"
OUTPUT_NII = f"{ROOT}/NV-Generate-BodyMask_git/output/test_v2_mask.nii.gz"

LATENT_SHAPE = [4, 64, 64, 64]
SPACING = [1.5, 1.5, 1.5]
NUM_STEPS = 100
CFG_SCALE = 2.0

os.makedirs(os.path.dirname(OUTPUT_NII), exist_ok=True)


def load_mask_models(device):
    from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
    from monai.networks.schedulers.rectified_flow import RFlowScheduler

    from scripts.adagn_unet import DiffusionModelUNetMaisiAdaGN

    ae = (
        AutoencoderKlMaisi(
            spatial_dims=3,
            in_channels=8,
            out_channels=125,
            latent_channels=4,
            num_channels=[32, 64, 128],
            num_res_blocks=[1, 2, 2],
            norm_num_groups=32,
            norm_eps=1e-6,
            attention_levels=[False, False, False],
            with_encoder_nonlocal_attn=False,
            with_decoder_nonlocal_attn=False,
            use_flash_attention=False,
            use_checkpointing=False,
            use_convtranspose=True,
            norm_float16=True,
            num_splits=1,
            dim_split=1,
        )
        .to(device)
        .eval()
    )
    ae.load_state_dict(torch.load(MASK_AE_PATH, weights_only=True))
    logging.info("Mask AE loaded.")

    unet = (
        DiffusionModelUNetMaisiAdaGN(
            spatial_dims=3,
            in_channels=4,
            out_channels=4,
            num_channels=[64, 128, 256, 512],
            attention_levels=[False, False, True, True],
            num_head_channels=[0, 0, 32, 32],
            num_res_blocks=2,
            use_flash_attention=True,
            with_conditioning=True,
            upcast_attention=True,
            include_fc=True,
            cross_attention_dim=19,
            include_spacing_input=True,
            include_top_region_index_input=False,
            include_bottom_region_index_input=False,
            adagn_condition_dim=19,
            adagn_context_embed_dim=256,
        )
        .to(device)
        .eval()
    )
    ck = torch.load(MASK_DM_PATH, weights_only=False, map_location=device)
    unet.load_state_dict(ck["unet_state_dict"])
    scale_factor = float(ck.get("scale_factor", 1.0055984258651733))
    logging.info(f"Mask DM (AdaGN) loaded. scale_factor={scale_factor:.6f}")

    scheduler = RFlowScheduler(
        num_train_timesteps=1000,
        use_discrete_timesteps=False,
        use_timestep_transform=True,
        sample_method="uniform",
        scale=1.0,
    )
    return ae, unet, scale_factor, scheduler


def main():
    device = torch.device("cuda")
    logging.info("=== Test 1: v2 mask-only generation ===")

    ae, unet, scale_factor, scheduler = load_mask_models(device)

    from scripts.sample_mask import ldm_conditional_sample_one_mask

    # 19-d conditioning: bone lesion size=0.5, all else -1
    #   anatomy slots 0-13 (liver,spleen,stomach,pancreas,colon,lkid,rkid,lung,gallbladder,
    #                        lung tumor,pancreatic tumor,hepatic tumor,colon cancer,bone lesion)
    #   demographics slots 14-18 (age,sex,weight,bmi,height)
    conditioning = [-1.0] * 19
    conditioning[13] = 0.5  # bone lesion

    t0 = time.time()
    mask = ldm_conditional_sample_one_mask(
        ae,
        unet,
        scheduler,
        scale_factor,
        conditioning,
        device,
        LATENT_SHAPE,
        REMAP_JSON,
        num_inference_steps=NUM_STEPS,
        cfg_guidance_scale=CFG_SCALE,
        spacing=SPACING,
    )
    elapsed = time.time() - t0
    logging.info(f"Mask generated in {elapsed:.1f}s. Shape: {tuple(mask.shape)}")

    data = mask.squeeze().cpu().numpy().astype(np.int16)
    unique_labels = sorted(np.unique(data).tolist())
    logging.info(f"Unique labels: {unique_labels[:20]}{'...' if len(unique_labels) > 20 else ''}")
    body_voxels = int((data > 0).sum())
    bone_lesion_voxels = int((data == 128).sum())
    logging.info(f"Body voxels: {body_voxels}, bone lesion (128) voxels: {bone_lesion_voxels}")

    affine = np.diag([SPACING[0], SPACING[1], SPACING[2], 1.0]).astype(np.float32)
    nib.save(nib.Nifti1Image(data, affine), OUTPUT_NII)
    logging.info(f"Saved to {OUTPUT_NII}")

    assert body_voxels > 0, "FAIL: mask has no body voxels"
    logging.info("=== Test 1 PASSED ===")


if __name__ == "__main__":
    torch.cuda.reset_peak_memory_stats()
    main()
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    logging.info(f"Peak GPU memory: {peak_gb:.2f} GB")
