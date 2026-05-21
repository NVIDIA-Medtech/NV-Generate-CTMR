"""
Parameterized inference smoke test — drives LDMSampler with config overrides
so we can test multiple model + config combinations from a single script.

Mirrors the body of inference_tutorial.ipynb but skips the visualization cells
and lets the caller override `generate_version` and `controllable_anatomy_size`
via CLI args.

Usage:
    # Path B (mask-database lookup) + rflow-ct — same as the notebook default
    python smoke_test_inference.py --version rflow-ct --controllable-size ""

    # Path A (mask DM generation) + rflow-ct — exercises ldm_conditional_sample_one_mask
    python smoke_test_inference.py --version rflow-ct --controllable-size '[["pancreas", 0.5]]'

    # ddpm-ct — auto-applies num_inference_steps=1000
    python smoke_test_inference.py --version ddpm-ct --controllable-size ""
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import monai  # noqa: F401  (registers transforms used in config)
import torch
from monai.utils import set_determinism

from scripts.diff_model_setting import setup_logging
from scripts.download_model_data import download_model_data
from scripts.sample import LDMSampler, check_input_ct
from scripts.utils import define_instance


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke test for NV-Generate-CTMR paired inference.")
    p.add_argument("--version", choices=["rflow-ct", "ddpm-ct"], required=True)
    p.add_argument(
        "--controllable-size",
        default="",
        help='JSON string for controllable_anatomy_size, e.g. \'[["pancreas", 0.5]]\'. Empty string = Path B (mask DB).',
    )
    p.add_argument("--random-seed", type=int, default=0)
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--root-dir", default="./temp_work_dir")
    return p.parse_args()


def main() -> int:
    cli = parse_args()
    logger = setup_logging("smoke")
    set_determinism(seed=cli.random_seed)

    logger.info(f"=== smoke test: version={cli.version} controllable-size={cli.controllable_size!r} ===")

    # Parse the controllable_anatomy_size override.
    if cli.controllable_size.strip():
        controllable_anatomy_size = json.loads(cli.controllable_size)
        # JSON gives [["pancreas", 0.5]] (list of lists) but the sampler expects
        # tuples — convert.
        controllable_anatomy_size = [tuple(x) for x in controllable_anatomy_size]
    else:
        controllable_anatomy_size = []

    # === Setup work dir ===
    os.environ["MONAI_DATA_DIRECTORY"] = cli.root_dir
    os.makedirs(cli.root_dir, exist_ok=True)
    root_dir = cli.root_dir

    # === Pick config files based on version ===
    if cli.version == "ddpm-ct":
        network = "ddpm"
        environment_file = "./configs/environment_ddpm-ct.json"
    else:
        network = "rflow"
        environment_file = "./configs/environment_rflow-ct.json"
    model_def_path = f"./configs/config_network_{network}.json"

    # === Download model weights ===
    download_model_data(cli.version, root_dir)

    # === Read environment paths ===
    args = argparse.Namespace()
    with open(environment_file) as f:
        env_dict = json.load(f)
    for k, v in env_dict.items():
        val = v if "datasets/" not in v else os.path.join(root_dir, v)
        setattr(args, k, val)

    # === Read network definition + inference config ===
    with open(model_def_path) as f:
        model_def = json.load(f)
    for k, v in model_def.items():
        setattr(args, k, v)

    with open("./configs/config_infer.json") as f:
        config_infer_dict = json.load(f)
    for k, v in config_infer_dict.items():
        setattr(args, k, v)

    # ── Apply OVERRIDES ─────────────────────────────────────
    args.controllable_anatomy_size = controllable_anatomy_size
    if cli.version == "ddpm-ct":
        # DDPM requires 1000 inference steps — the notebook applies this same override.
        args.num_inference_steps = 1000
        logger.warning(f"ddpm-ct: forcing num_inference_steps = {args.num_inference_steps}")
    # Keep num_output_samples at 1 (or as overridden by CLI)
    args.num_output_samples = cli.num_samples
    # ────────────────────────────────────────────────────────

    # Validate input parameters (CT-only path)
    check_input_ct(
        args.body_region,
        args.anatomy_list,
        args.label_dict_json,
        args.output_size,
        args.spacing,
        args.controllable_anatomy_size,
    )

    latent_shape = [
        args.latent_channels,
        args.output_size[0] // 4,
        args.output_size[1] // 4,
        args.output_size[2] // 4,
    ]

    # === Build noise schedulers + networks ===
    noise_scheduler = define_instance(args, "noise_scheduler")
    mask_generation_noise_scheduler = define_instance(args, "mask_generation_noise_scheduler")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    autoencoder = define_instance(args, "autoencoder_def").to(device)
    ckpt_ae = torch.load(args.trained_autoencoder_path)
    if "unet_state_dict" in ckpt_ae:
        ckpt_ae = ckpt_ae["unet_state_dict"]
    autoencoder.load_state_dict(ckpt_ae)

    diffusion_unet = define_instance(args, "diffusion_unet_def").to(device)
    ckpt_dm = torch.load(args.trained_diffusion_path, weights_only=False)
    diffusion_unet.load_state_dict(ckpt_dm["unet_state_dict"], strict=False)
    scale_factor = ckpt_dm["scale_factor"].to(device)

    controlnet = define_instance(args, "controlnet_def").to(device)
    ckpt_cn = torch.load(args.trained_controlnet_path, weights_only=False)
    monai.networks.utils.copy_model_state(controlnet, diffusion_unet.state_dict())
    controlnet.load_state_dict(ckpt_cn["controlnet_state_dict"], strict=False)

    mask_generation_autoencoder = define_instance(args, "mask_generation_autoencoder_def").to(device)
    ckpt_mae = torch.load(args.trained_mask_generation_autoencoder_path, weights_only=True)
    mask_generation_autoencoder.load_state_dict(ckpt_mae)

    mask_generation_diffusion_unet = define_instance(args, "mask_generation_diffusion_def").to(device)
    ckpt_mdm = torch.load(args.trained_mask_generation_diffusion_path, weights_only=True)
    mask_generation_diffusion_unet.load_state_dict(ckpt_mdm["unet_state_dict"])
    mask_generation_scale_factor = ckpt_mdm["scale_factor"]

    logger.info("All trained model weights loaded.")

    # === Build LDMSampler — this is the refactor's main consumer ===
    ldm_sampler = LDMSampler(
        args.body_region,
        args.anatomy_list,
        args.all_mask_files_json,
        args.all_anatomy_size_conditions_json,
        args.all_mask_files_base_dir,
        args.label_dict_json,
        args.label_dict_remap_json,
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
        args.mask_generation_latent_shape,
        args.output_size,
        args.output_dir,
        args.controllable_anatomy_size,
        image_output_ext=args.image_output_ext,
        label_output_ext=args.label_output_ext,
        spacing=args.spacing,
        modality=args.modality,
        num_inference_steps=args.num_inference_steps,
        mask_generation_num_inference_steps=args.mask_generation_num_inference_steps,
        random_seed=args.random_seed if hasattr(args, "random_seed") else cli.random_seed,
        autoencoder_sliding_window_infer_size=args.autoencoder_sliding_window_infer_size,
        autoencoder_sliding_window_infer_overlap=args.autoencoder_sliding_window_infer_overlap,
        cfg_guidance_scale=args.cfg_guidance_scale,
    )

    # === Run inference ===
    logger.info(f"Output dir: {args.output_dir}")
    if controllable_anatomy_size:
        logger.info(f"Path A (mask DM from scratch) triggered: {controllable_anatomy_size}")
    else:
        logger.info("Path B (mask-database lookup) triggered.")

    output_filenames = ldm_sampler.sample_multiple_images(args.num_output_samples)
    logger.info(f"MAISI generation finished. Outputs: {output_filenames}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
