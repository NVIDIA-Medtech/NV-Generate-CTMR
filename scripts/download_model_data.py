import os
from monai.apps import download_url

def download_model_data(generate_version,root_dir):
    # TODO: remove the `files` after the files are uploaded to the NGC
    if generate_version == "ddpm-ct" or generate_version == "rflow-ct":
        files = [
            {
                "path": "models/autoencoder_v1.pt",
                "url": "https://developer.download.nvidia.com/assets/Clara/monai/tutorials"
                "/model_zoo/model_maisi_autoencoder_epoch273_alternative.pt",
            },
            {
                "path": "models/mask_generation_autoencoder.pt",
                "url": "https://developer.download.nvidia.com/assets/Clara/monai" "/tutorials/mask_generation_autoencoder.pt",
            },
            {
                "path": "models/mask_generation_diffusion_unet.pt",
                "url": "https://developer.download.nvidia.com/assets/Clara/monai"
                "/tutorials/model_zoo/model_maisi_mask_generation_diffusion_unet_v2.pt",
            },
            {
                "path": "configs/all_anatomy_size_conditions.json",
                "url": "https://developer.download.nvidia.com/assets/Clara/monai/tutorials/all_anatomy_size_condtions.json",
            },
            {
                "path": "datasets/all_masks_flexible_size_and_spacing_4000.zip",
                "url": "https://developer.download.nvidia.com/assets/Clara/monai"
                "/tutorials/all_masks_flexible_size_and_spacing_4000.zip",
            },
        ]
    if generate_version == "rflow-mr":
        files = [
            {
                "path": "models/autoencoder_v2.pt",
                "url": "https://huggingface.co/nvidia/NV-Generate-MR/blob/main/models/autoencoder_v2.pt"
            },
            {
                "path": "models/diff_unet_3d_rflow-mr.pt",
                "url": "https://huggingface.co/nvidia/NV-Generate-MR/blob/main/models/diff_unet_3d_rflow-mr.pt"
            }
        ]
    if generate_version == "ddpm-ct":
        files += [
            {
                "path": "models/diff_unet_3d_ddpm-ct.pt",
                "url": "https://developer.download.nvidia.com/assets/Clara/monai/tutorials/model_zoo"
                "/model_maisi_input_unet3d_data-all_steps1000size512ddpm_random_current_inputx_v1_alternative.pt",
            },
            {
                "path": "models/controlnet_3d_ddpm-ct.pt",
                "url": "https://developer.download.nvidia.com/assets/Clara/monai/tutorials/model_zoo"
                "/model_maisi_controlnet-20datasets-e20wl100fold0bc_noi_dia_fsize_current_alternative.pt",
            },
            {
                "path": "configs/candidate_masks_flexible_size_and_spacing_3000.json",
                "url": "https://developer.download.nvidia.com/assets/Clara/monai"
                "/tutorials/candidate_masks_flexible_size_and_spacing_3000.json",
            },
        ]
    elif generate_version == "rflow-ct":
        files += [
            {
                "path": "models/diff_unet_3d_rflow-ct.pt",
                "url": "https://developer.download.nvidia.com/assets/Clara/monai/tutorials/"
                "diff_unet_ckpt_rflow_epoch19350.pt",
            },
            {
                "path": "models/controlnet_3d_rflow-ct.pt",
                "url": "https://developer.download.nvidia.com/assets/Clara/monai/tutorials/controlnet_rflow_epoch60.pt",
            },
            {
                "path": "configs/candidate_masks_flexible_size_and_spacing_4000.json",
                "url": "https://developer.download.nvidia.com/assets/Clara/monai"
                "/tutorials/candidate_masks_flexible_size_and_spacing_4000.json",
            },
        ]
    elif generate_version == "rflow-mr":
        files += [
            {
                "path": "models/autoencoder_v2.pt",
                "url": "https://huggingface.co/nvidia/NV-Generate-MR/blob/main/models/autoencoder_v2.pt",
            },
            {
                "path": "models/diff_unet_3d_rflow-mr.pt",
                "url": "https://huggingface.co/nvidia/NV-Generate-MR/blob/main/models/diff_unet_3d_rflow-mr.pt",
            },
            {
                "path": "configs/candidate_masks_flexible_size_and_spacing_brats23.json",
                "url": "https://huggingface.co/nvidia/NV-Generate-MR"
                "/blob/main/example_data/candidate_masks_flexible_size_and_spacing_brats23.json",
            },
            {
                "path": "datasets/all_masks_flexible_size_and_spacing_brats23.zip",
                "url": "https://huggingface.co/nvidia/NV-GenerateMR"
                "/blob/main/example_data/all_masks_flexible_size_and_spacing_brats23.zip",
            },
        ]
    else:
        raise ValueError(f"generate_version has to be chosen from ['ddpm-ct', 'rflow-ct', 'rflow-mr'], yet got {generate_version}.")
    
    for file in files:
        file["path"] = file["path"] if "datasets/" not in file["path"] else os.path.join(root_dir, file["path"])
        download_url(url=file["url"], filepath=file["path"])
    return