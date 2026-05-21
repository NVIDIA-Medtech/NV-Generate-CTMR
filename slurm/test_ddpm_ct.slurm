#!/bin/bash
#SBATCH --nodes=1
#SBATCH -A healthcareeng_monai
#SBATCH --partition batch
#SBATCH --gpus-per-node=1
#SBATCH --time=2:00:00
#SBATCH --job-name=ctmr_ddpm_ct
#SBATCH --output=/lustre/fsw/portfolios/healthcareeng/users/canz/slurm-logs/slurm-%j-ctmr_ddpm_ct.out
#SBATCH --error=/lustre/fsw/portfolios/healthcareeng/users/canz/slurm-logs/slurm-%j-ctmr_ddpm_ct.out

# ddpm-ct test: exercises the DDPM scheduler path through
# ldm_conditional_sample_one_image (scripts/infer_image_from_mask.py) with the
# mandatory num_inference_steps=1000. Uses Path B (mask DB lookup) since
# controllable_anatomy_size is empty. ~33x slower than rflow-ct due to step count.

set -uo pipefail

REPO_DIR=/lustre/fsw/portfolios/healthcareeng/users/canz/code/NV-Generate-CTMR_refactor
CONDA_PY=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/MAISI/maisi_conda_env_v3/bin/python

cd "$REPO_DIR"

export HF_HOME=/lustre/fsw/portfolios/healthcareeng/users/canz/hf_cache
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$HF_HOME"

# Add repo root to PYTHONPATH so `from scripts...` imports resolve when running
# the smoke test directly with `python <path>` (vs `python -m`).
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

echo "[info] host: $(hostname)"
nvidia-smi -L || true
$CONDA_PY -c "import torch; print(f'torch={torch.__version__} cuda={torch.cuda.is_available()} ngpu={torch.cuda.device_count()}')"

PYTHONUNBUFFERED=1 $CONDA_PY -u slurm/smoke_test_inference.py \
    --version ddpm-ct \
    --controllable-size "" \
    --random-seed 0 \
    --num-samples 1 \
    --root-dir ./temp_work_dir
RC=$?

echo "[info] python exit code: $RC"
echo "[info] outputs:"
ls -la "$REPO_DIR/output" 2>/dev/null || echo "(no output dir found)"
exit $RC
