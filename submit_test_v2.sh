#!/bin/bash
#SBATCH --nodes=1
#SBATCH -A healthcareeng_monai
#SBATCH --partition batch
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:45:00
#SBATCH --job-name=test_mask_v2
#SBATCH --output=/lustre/fsw/portfolios/healthcareeng/users/canz/slurm-logs/slurm-%j-test_mask_v2.out

set -uo pipefail

PY=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/MAISI/maisi_conda_env_v3/bin/python
REPO=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/users/canz/projects/NV-Generate-BodyMask_git
CT_HF=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/users/canz/projects/NV-Generate-CT_HF

export MONAI_DATA_DIRECTORY=$CT_HF
# Redirect HF cache to lustre so compute nodes don't fill /home/.cache
export HF_HOME=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/users/canz/.cache/huggingface
export PYTHONPATH=$REPO:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1

mkdir -p /lustre/fsw/portfolios/healthcareeng/users/canz/slurm-logs
mkdir -p $REPO/output/test_paired

# Symlink models/ and datasets/ into $REPO so that download_model_data's relative
# dst.exists() checks find existing files without downloading anything.
# The actual relative config paths (./configs/*) stay resolved from $REPO.
[ -L $REPO/models ] || ln -s $CT_HF/models $REPO/models
[ -L $REPO/datasets ] || ln -s $CT_HF/datasets $REPO/datasets

echo "=========================================="
echo "Test 1: mask-only generation (v2 DM + RFlow + CFG)"
echo "=========================================="
cd $REPO
srun "$PY" -u $REPO/test_v2_mask.py
echo "Test 1 exit code: $?"

echo ""
echo "=========================================="
echo "Test 2: paired mask+image generation (full inference)"
echo "=========================================="
cd $REPO
srun "$PY" -u -m scripts.inference \
    --environment-file $REPO/configs/environment_rflow-ct_test.json \
    --config-file $REPO/configs/config_network_rflow.json \
    --inference-file $REPO/configs/config_infer_test_v2.json \
    --version rflow-ct
echo "Test 2 exit code: $?"

echo "=========================================="
echo "All tests done. Check $REPO/output/ for results."
echo "=========================================="
