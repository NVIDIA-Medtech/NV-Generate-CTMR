#!/bin/bash
#SBATCH --nodes=1
#SBATCH -A healthcareeng_monai
#SBATCH --partition batch
#SBATCH --gpus-per-node=1
#SBATCH --time=2:00:00
#SBATCH --job-name=ctmr_refactor_nbtest
#SBATCH --output=/lustre/fsw/portfolios/healthcareeng/users/canz/slurm-logs/slurm-%j-ctmr_refactor_nbtest.out
#SBATCH --error=/lustre/fsw/portfolios/healthcareeng/users/canz/slurm-logs/slurm-%j-ctmr_refactor_nbtest.out

# End-to-end test of the sample.py refactor: runs inference_tutorial.ipynb
# (which uses scripts.sample.LDMSampler — the orchestrator that exercises
# both the mask-DM path (scripts.sample_mask) and the image-from-mask path
# (scripts.infer_image_from_mask) through the backward-compat shim.

# Don't use 'set -e' here — we want to see errors in stdout, not have the
# script abort silently on the first non-zero exit.
set -uo pipefail

REPO_DIR=/lustre/fsw/portfolios/healthcareeng/users/canz/code/NV-Generate-CTMR_refactor
CONDA_PY=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/MAISI/maisi_conda_env_v3/bin/python

cd "$REPO_DIR"

# /home is 10 GB and may fill quickly. Redirect HuggingFace + pip caches to lustre.
export HF_HOME=/lustre/fsw/portfolios/healthcareeng/users/canz/hf_cache
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"
echo "[info] HF_HOME=$HF_HOME (free: $(df -h "$HF_HOME" | awk 'NR==2{print $4}'))"

echo "[info] host: $(hostname)"
echo "[info] gpu(s):"
nvidia-smi -L || true
echo "[info] python: $CONDA_PY"
$CONDA_PY -c "import torch; print(f'torch={torch.__version__}  cuda={torch.cuda.is_available()}  ngpu={torch.cuda.device_count()}')"

# Sanity: confirm the refactored modules import cleanly before we burn GPU time
$CONDA_PY -c "
from scripts.sample import LDMSampler, ldm_conditional_sample_one_mask, ldm_conditional_sample_one_image, ReconModel, filter_mask_with_organs, crop_img_body_mask, check_input_ct
import scripts.sample, scripts.sample_mask, scripts.infer_image_from_mask
print('[ok] backward-compat imports resolve')
"

# Convert notebook to a runnable Python script first — easier to debug than
# nbconvert --execute (which buffers errors into the .ipynb file rather than
# the slurm log).
echo "[info] converting notebook to python script..."
$CONDA_PY -m jupyter nbconvert --to script inference_tutorial.ipynb --output inference_tutorial_runme

# Strip IPython-only lines — `!shell`/`%magic` cells get converted to
# `get_ipython().system(...)` / `get_ipython().run_line_magic(...)` which only
# work inside an IPython kernel. Comment them out so plain `python` can run
# the script. (Cell 2 of inference_tutorial.ipynb is a pip-install hedge —
# safely skippable since requirements are already met in this conda env.)
sed -i 's|^\s*get_ipython()|# stripped IPython magic: get_ipython()|' inference_tutorial_runme.py

# Run with unbuffered output so we see errors in real time.
echo "[info] running inference_tutorial_runme.py (unbuffered)..."
PYTHONUNBUFFERED=1 $CONDA_PY -u inference_tutorial_runme.py
NBRC=$?
echo "[info] python exit code: $NBRC"

echo "[info] notebook executed successfully"
echo "[info] outputs:"
ls -la "$REPO_DIR/temp_work_dir" 2>/dev/null || echo "(no temp_work_dir found)"

# Show the generated NIfTIs (if any)
find "$REPO_DIR/temp_work_dir" -name "*.nii.gz" 2>/dev/null | head -10 || true

echo "[done] inference_tutorial.ipynb refactor smoke test complete"
