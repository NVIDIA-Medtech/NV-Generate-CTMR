#!/usr/bin/env python3
# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

"""Noninteractive diffusion-UNet training workflow.

This script is a command-line extraction of ``train_diff_unet_tutorial.ipynb``.
It stages the model, environment, and network configs; creates latent
autoencoder embeddings; writes the per-embedding conditioning JSON files; runs
diffusion UNet training; and can optionally run inference with the trained
checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import nibabel as nib

try:
    from .diff_model_create_training_data import diff_model_create_training_data
    from .diff_model_infer import diff_model_infer
    from .diff_model_setting import setup_logging
    from .diff_model_train import diff_model_train
    from .download_model_data import download_model_data
except ImportError:  # pragma: no cover - direct script execution fallback.
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from scripts.diff_model_create_training_data import diff_model_create_training_data
    from scripts.diff_model_infer import diff_model_infer
    from scripts.diff_model_setting import setup_logging
    from scripts.diff_model_train import diff_model_train
    from scripts.download_model_data import download_model_data


SUPPORTED_VERSIONS = ("ddpm-ct", "rflow-ct", "rflow-mr", "rflow-mr-brain")
DEFAULT_MODALITY = {
    "ddpm-ct": "ct",
    "rflow-ct": "ct",
    "rflow-mr": "mri_t1",
    "rflow-mr-brain": "mri_t1",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=4, sort_keys=True) + "\n")
    return path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _network_config_name(version: str) -> str:
    if version == "ddpm-ct":
        return "config_network_ddpm.json"
    return "config_network_rflow.json"


def _config_paths(repo_root: Path, version: str) -> tuple[Path, Path, Path]:
    return (
        repo_root / "configs" / _network_config_name(version),
        repo_root / "configs" / f"environment_maisi_diff_model_{version}.json",
        repo_root / "configs" / f"config_maisi_diff_model_{version}.json",
    )


def _modality_mapping(repo_root: Path) -> dict[str, int]:
    path = repo_root / "configs" / "modality_mapping.json"
    if not path.is_file():
        return {}
    return {str(k): int(v) for k, v in _load_json(path).items()}


def _resolve_from_repo(repo_root: Path, value: str | None) -> str | None:
    if value in (None, ""):
        return value
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return str(path)
    return str((repo_root / path).resolve())


def _stage_datalist(
    data_base_dir: Path,
    input_path: Path,
    output_path: Path,
    default_modality: str,
) -> tuple[Path, dict[str, Any]]:
    raw = _load_json(input_path)
    if "training" not in raw or not isinstance(raw["training"], list) or not raw["training"]:
        raise ValueError("datalist must contain a non-empty training list")

    staged: dict[str, Any] = {"training": [], "testing": []}
    for split in ("training", "testing"):
        for item in raw.get(split, []):
            if not isinstance(item, dict) or "image" not in item:
                raise ValueError(f"{split} entries must be objects with an image field")
            image_path = data_base_dir / item["image"]
            if not image_path.is_file():
                raise FileNotFoundError(f"{split} image not found: {image_path}")
            next_item = dict(item)
            next_item.setdefault("modality", default_modality)
            staged[split].append(next_item)

    _write_json(output_path, staged)
    return output_path, staged


def _create_embedding_sidecars(
    embedding_base_dir: Path,
    modality: str,
    include_body_region: bool,
    top_region_index: list[int],
    bottom_region_index: list[int],
) -> list[Path]:
    sidecars: list[Path] = []
    for emb in sorted(embedding_base_dir.rglob("*_emb.nii.gz")):
        img = nib.load(str(emb))
        data = {
            "dim": [int(v) for v in img.shape[:3]],
            "spacing": [float(v) for v in img.header.get_zooms()[:3]],
            "modality": modality,
        }
        if include_body_region:
            data["top_region_index"] = top_region_index
            data["bottom_region_index"] = bottom_region_index
        sidecar = Path(str(emb) + ".json")
        _write_json(sidecar, data)
        sidecars.append(sidecar)
    return sidecars


def _stage_configs(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = _repo_root()
    model_def_src, env_src, model_src = _config_paths(repo_root, args.generate_version)
    for path in (model_def_src, env_src, model_src):
        if not path.is_file():
            raise FileNotFoundError(path)

    run_dir = args.work_dir.resolve()
    config_dir = run_dir / "configs"
    embedding_dir = run_dir / "embeddings"
    output_dir = args.output_dir.resolve()
    model_dir = output_dir / "models"
    inference_dir = output_dir / "inference"
    datalist_path, datalist = _stage_datalist(
        args.data_base_dir.resolve(),
        args.datalist.resolve(),
        run_dir / "dataset.json",
        args.modality,
    )

    model_def = copy.deepcopy(_load_json(model_def_src))
    env_config = copy.deepcopy(_load_json(env_src))
    model_config = copy.deepcopy(_load_json(model_src))

    env_config["data_base_dir"] = str(args.data_base_dir.resolve())
    env_config["embedding_base_dir"] = str(embedding_dir)
    env_config["json_data_list"] = str(datalist_path)
    env_config["model_dir"] = str(model_dir)
    env_config["output_dir"] = str(inference_dir)
    env_config["output_prefix"] = args.output_prefix
    env_config["modality_mapping_path"] = str((repo_root / "configs" / "modality_mapping.json").resolve())
    env_config["trained_autoencoder_path"] = (
        str(args.trained_autoencoder_path.resolve())
        if args.trained_autoencoder_path
        else _resolve_from_repo(repo_root, env_config.get("trained_autoencoder_path"))
    )
    if args.existing_ckpt_filepath:
        env_config["existing_ckpt_filepath"] = str(args.existing_ckpt_filepath.resolve())
    elif args.train_from_scratch:
        env_config["existing_ckpt_filepath"] = None
    else:
        env_config["existing_ckpt_filepath"] = _resolve_from_repo(repo_root, env_config.get("existing_ckpt_filepath"))
    if args.model_filename:
        env_config["model_filename"] = args.model_filename

    train_config = model_config.setdefault("diffusion_unet_train", {})
    train_config["n_epochs"] = args.epochs
    train_config["batch_size"] = args.batch_size
    train_config["lr"] = args.lr
    train_config["cache_rate"] = args.cache_rate

    modality_code = _modality_mapping(repo_root).get(args.modality)
    if modality_code is None:
        raise ValueError(f"modality {args.modality!r} not found in configs/modality_mapping.json")
    infer_config = model_config.setdefault("diffusion_unet_inference", {})
    infer_config["dim"] = args.infer_dim
    infer_config["spacing"] = args.infer_spacing
    infer_config["modality"] = modality_code
    infer_config["random_seed"] = args.random_seed
    infer_config["num_inference_steps"] = args.num_inference_steps
    infer_config["cfg_guidance_scale"] = args.cfg_guidance_scale
    infer_config["top_region_index"] = args.top_region_index
    infer_config["bottom_region_index"] = args.bottom_region_index

    if "autoencoder_def" in model_def and args.autoencoder_num_splits is not None:
        model_def["autoencoder_def"]["num_splits"] = args.autoencoder_num_splits

    staged = {
        "env_config": _write_json(config_dir / "environment_maisi_diff_model.json", env_config),
        "model_config": _write_json(config_dir / "config_maisi_diff_model.json", model_config),
        "model_def": _write_json(config_dir / "config_maisi.json", model_def),
        "embedding_dir": embedding_dir,
        "output_dir": output_dir,
        "model_dir": model_dir,
        "inference_dir": inference_dir,
        "datalist": datalist,
        "include_body_region": bool(model_def.get("include_body_region", False)),
        "modality_code": modality_code,
    }
    return staged


def _parse_int_triplet(value: str) -> list[int]:
    parts = [int(v.strip()) for v in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated integers")
    return parts


def _parse_float_triplet(value: str) -> list[float]:
    parts = [float(v.strip()) for v in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated numbers")
    return parts


def _parse_region(value: str) -> list[int]:
    parts = [int(v.strip()) for v in value.split(",")]
    if len(parts) != 4 or any(v not in (0, 1) for v in parts):
        raise argparse.ArgumentTypeError("expected four comma-separated 0/1 values")
    return parts


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate-version", choices=SUPPORTED_VERSIONS, default="rflow-mr-brain")
    parser.add_argument("--data-base-dir", type=Path, required=True)
    parser.add_argument("--datalist", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--modality", default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--cache-rate", type=float, default=0.0)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--autoencoder-num-splits", type=int, default=2)
    parser.add_argument("--existing-ckpt-filepath", type=Path)
    parser.add_argument("--trained-autoencoder-path", type=Path)
    parser.add_argument("--model-filename", default="")
    parser.add_argument("--train-from-scratch", action="store_true")
    parser.add_argument("--download-model-data", action="store_true")
    parser.add_argument("--skip-create-training-data", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--run-inference", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    parser.add_argument("--infer-dim", type=_parse_int_triplet, default=[256, 256, 256])
    parser.add_argument("--infer-spacing", type=_parse_float_triplet, default=[1.0, 1.0, 1.0])
    parser.add_argument("--top-region-index", type=_parse_region, default=[0, 1, 0, 0])
    parser.add_argument("--bottom-region-index", type=_parse_region, default=[0, 0, 1, 0])
    parser.add_argument("--random-seed", type=int, default=1234)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--cfg-guidance-scale", type=float, default=10.0)
    parser.add_argument("--output-prefix", default="diff_unet_finetuned")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.modality is None:
        args.modality = DEFAULT_MODALITY[args.generate_version]

    logger = setup_logging("diffusion_unet_train_workflow")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.download_model_data:
        logger.info("Downloading model data for %s", args.generate_version)
        download_model_data(args.generate_version, str(_repo_root()), model_only=True)

    staged = _stage_configs(args)
    logger.info("Staged configs under %s", args.work_dir)

    if not args.skip_create_training_data:
        diff_model_create_training_data(
            str(staged["env_config"]),
            str(staged["model_config"]),
            str(staged["model_def"]),
            args.num_gpus,
        )
    sidecars = _create_embedding_sidecars(
        staged["embedding_dir"],
        args.modality,
        staged["include_body_region"],
        args.top_region_index,
        args.bottom_region_index,
    )

    if not args.skip_train:
        diff_model_train(
            str(staged["env_config"]),
            str(staged["model_config"]),
            str(staged["model_def"]),
            args.num_gpus,
            args.amp,
        )

    inference_outputs: list[str] = []
    if args.run_inference:
        inference_outputs = diff_model_infer(
            str(staged["env_config"]),
            str(staged["model_config"]),
            str(staged["model_def"]),
            args.num_gpus,
        )

    model_filename = _load_json(staged["env_config"]).get("model_filename")
    summary = {
        "generate_version": args.generate_version,
        "modality": args.modality,
        "modality_code": staged["modality_code"],
        "training_cases": len(staged["datalist"].get("training", [])),
        "testing_cases": len(staged["datalist"].get("testing", [])),
        "embedding_sidecars": [str(p) for p in sidecars],
        "checkpoint": str(staged["model_dir"] / model_filename) if model_filename else None,
        "inference_outputs": inference_outputs,
        "staged_configs": {
            "env_config": str(staged["env_config"]),
            "model_config": str(staged["model_config"]),
            "model_def": str(staged["model_def"]),
        },
    }
    _write_json(args.output_dir / "workflow_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
