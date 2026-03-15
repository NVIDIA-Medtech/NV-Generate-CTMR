# Troubleshooting

## Common Issues

### GPU Out of Memory (OOM)

**Symptom:** `CUDA out of memory` error during inference or training.

**Solutions:**

- **During inference:** Increase `autoencoder_tp_num_splits` or reduce `autoencoder_sliding_window_infer_size` in `./configs/config_infer.json`. See the [inference GPU memory table](inference.md#inference-time-cost-and-gpu-memory-usage) for recommended configurations by GPU size.
- **During training:** Reduce `patch_size` in the training config (e.g., from [128,128,128] to [64,64,64] for VAE training). See the [training GPU memory table](training.md#training-gpu-memory-usage) for memory requirements by image size.
- Use a smaller `output_size` to generate smaller volumes.

### Stitching Artifacts in Generated Images

**Symptom:** Visible seam lines in the generated images.

**Solutions:**

- Increase `autoencoder_sliding_window_infer_overlap` in `./configs/config_infer.json` (values closer to 1.0 reduce artifacts but increase time).
- Increase `autoencoder_sliding_window_infer_size` if GPU memory allows.

### MONAI Version Compatibility

**Symptom:** Import errors or missing modules when using MR generation.

**Solution:** MR generation (`rflow-mr`) requires `monai>=1.5.0`. CT generation works with `monai>=1.3.2`. Upgrade with:

```bash
pip install --upgrade monai>=1.5.0
```

### Model Weight Download Issues

**Symptom:** Errors when downloading model weights from HuggingFace.

**Solutions:**

- Ensure you have a working internet connection.
- Try setting the `MONAI_DATA_DIRECTORY` environment variable to a writable directory.
- Download manually using `python -m scripts.download_model_data --version <version> --root_dir "./" --model_only`.
- Check the [HuggingFace model pages](https://huggingface.co/nvidia/NV-Generate-CT) for any access requirements.

### Incorrect HU Values in Generated CT

**Symptom:** Generated CT images have unrealistic Hounsfield unit values.

**Solution:** The quality check function validates HU ranges for major organs. Run inference with quality checking enabled. If issues persist, try different random seeds or adjust the `spacing` and `output_size` parameters to match the recommended spacing tables in the [inference guide](inference.md#recommended-spacing-for-ct).

## Getting Help

- For questions about MONAI usage: [MONAI Discussions](https://github.com/Project-MONAI/MONAI/discussions)
- For MONAI bugs: [MONAI Issues](https://github.com/Project-MONAI/MONAI/issues)
- For NV-Generate-CTMR bugs: [Open an issue](https://github.com/nvidia-medtech/NV-Generate-CTMR/issues/new?template=bug_report.yml)
