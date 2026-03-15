import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download
from monai.apps import download_url


def ensure_hf_download_tracked(
    repo_id: str,
    revision: str = "main",
    token: str | None = None,
) -> str:
    """
    Force a request to config.json so Hugging Face generic download tracking
    can register a download for repos that rely on the default query file.
    """
    return hf_hub_download(
        repo_id=repo_id,
        filename="config.json",
        revision=revision,
        token=token,
    )


def fetch_to_hf_path_cmd(
    items: list[dict[str, str]],
    root_dir: str = "./",
    revision: str = "main",
    overwrite: bool = False,
    token: str | None = None,
    track_download: bool = True,
) -> list[str]:
    """
    items: list of {"repo_id": "...", "filename": "path/in/repo.ext", "path": "local/target.ext"}
    Returns list of saved local paths.
    """
    saved = []
    tracked_repos: set[str] = set()

    for it in items:
        repo_id = it["repo_id"]
        repo_file = it["filename"]
        dst = Path(it["path"])
        dst.parent.mkdir(parents=True, exist_ok=True)

        # Hit config.json once per repo before downloading weights/data.
        if track_download and repo_id not in tracked_repos:
            try:
                ensure_hf_download_tracked(
                    repo_id=repo_id,
                    revision=revision,
                    token=token,
                )
            except Exception as e:
                print(f"Warning: failed to fetch config.json for tracking from {repo_id}: {e}")
            tracked_repos.add(repo_id)

        if dst.exists() and not overwrite:
            saved.append(str(dst))
            continue

        cached_path = hf_hub_download(
            repo_id=repo_id,
            filename=repo_file,
            revision=revision,
            token=token,
        )

        if dst.exists() and overwrite:
            dst.unlink()

        shutil.copy2(cached_path, dst)
        saved.append(str(dst))

    return saved


def download_model_data(generate_version, root_dir, model_only=False):
    # TODO: remove the `files` after the files are uploaded to the NGC
    if generate_version == "rflow-mr-brain":
        files = [
            {
                "path": "models/autoencoder_v1.pt",
                "repo_id": "nvidia/NV-Generate-CT",
                "filename": "models/autoencoder_v1.pt",
            },
            {
                "path": "models/diff_unet_3d_rflow-mr-brain_v0.pt",
                "repo_id": "nvidia/NV-Generate-MR-Brain",
                "filename": "models/diff_unet_3d_rflow-mr-brain_v0.pt",
            },
        ]
    elif generate_version == "ddpm-ct" or generate_version == "rflow-ct":
        files = [
            {
                "path": "models/autoencoder_v1.pt",
                "repo_id": "nvidia/NV-Generate-CT",
                "filename": "models/autoencoder_v1.pt",
            },
            {
                "path": "models/mask_generation_autoencoder.pt",
                "repo_id": "nvidia/NV-Generate-CT",
                "filename": "models/mask_generation_autoencoder.pt",
            },
            {
                "path": "models/mask_generation_diffusion_unet.pt",
                "repo_id": "nvidia/NV-Generate-CT",
                "filename": "models/mask_generation_diffusion_unet.pt",
            },
        ]
        if not model_only:
            files += [
                {
                    "path": "datasets/all_anatomy_size_conditions.json",
                    "repo_id": "nvidia/NV-Generate-CT",
                    "filename": "datasets/all_anatomy_size_conditions.json",
                },
                {
                    "path": "datasets/all_masks_flexible_size_and_spacing_4000.zip",
                    "repo_id": "nvidia/NV-Generate-CT",
                    "filename": "datasets/all_masks_flexible_size_and_spacing_4000.zip",
                },
            ]
    elif generate_version == "rflow-mr":
        files = [
            {
                "path": "models/autoencoder_v2.pt",
                "repo_id": "nvidia/NV-Generate-MR",
                "filename": "models/autoencoder_v2.pt",
            },
            {
                "path": "models/diff_unet_3d_rflow-mr.pt",
                "repo_id": "nvidia/NV-Generate-MR",
                "filename": "models/diff_unet_3d_rflow-mr.pt",
            },
        ]
    else:
        raise ValueError(f"generate_version has to be chosen from ['ddpm-ct', 'rflow-ct', 'rflow-mr'], yet got {generate_version}.")
    if generate_version == "ddpm-ct":
        files += [
            {
                "path": "models/diff_unet_3d_ddpm-ct.pt",
                "repo_id": "nvidia/NV-Generate-CT",
                "filename": "models/diff_unet_3d_ddpm-ct.pt",
            },
            {
                "path": "models/controlnet_3d_ddpm-ct.pt",
                "repo_id": "nvidia/NV-Generate-CT",
                "filename": "models/controlnet_3d_ddpm-ct.pt",
            },
        ]
        if not model_only:
            files += [
                {
                    "path": "datasets/candidate_masks_flexible_size_and_spacing_3000.json",
                    "repo_id": "nvidia/NV-Generate-CT",
                    "filename": "datasets/candidate_masks_flexible_size_and_spacing_3000.json",
                },
            ]
    elif generate_version == "rflow-ct":
        files += [
            {
                "path": "models/diff_unet_3d_rflow-ct.pt",
                "repo_id": "nvidia/NV-Generate-CT",
                "filename": "models/diff_unet_3d_rflow-ct.pt",
            },
            {
                "path": "models/controlnet_3d_rflow-ct.pt",
                "repo_id": "nvidia/NV-Generate-CT",
                "filename": "models/controlnet_3d_rflow-ct.pt",
            },
        ]
        if not model_only:
            files += [
                {
                    "path": "datasets/candidate_masks_flexible_size_and_spacing_4000.json",
                    "repo_id": "nvidia/NV-Generate-CT",
                    "filename": "datasets/candidate_masks_flexible_size_and_spacing_4000.json",
                },
            ]

    for file in files:
        file["path"] = file["path"] if "datasets/" not in file["path"] else os.path.join(root_dir, file["path"])
        if "repo_id" in file.keys():
            path = fetch_to_hf_path_cmd([file], root_dir=root_dir, revision="main")
            print("saved to:", path)
        else:
            download_url(url=file["url"], filepath=file["path"])
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model downloading")
    parser.add_argument(
        "--version",
        type=str,
        default="rflow-ct",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="./",
    )
    parser.add_argument("--model_only", dest="model_only", action="store_true", help="Download model only, not any dataset")

    args = parser.parse_args()
    download_model_data(args.version, args.root_dir, args.model_only)
