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

from __future__ import annotations

import os
import argparse
import json
import logging
import subprocess, tempfile

import torch
import torch.distributed as dist
from monai.utils import RankFilter


def setup_logging(logger_name: str = "") -> logging.Logger:
    """
    Setup the logging configuration.

    Args:
        logger_name (str): logger name.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(logger_name)
    if dist.is_initialized():
        logger.addFilter(RankFilter())
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logger


def load_config(env_config_path: str, model_config_path: str, model_def_path: str) -> argparse.Namespace:
    """
    Load configuration from JSON files.

    Args:
        env_config_path (str): Path to the environment configuration file.
        model_config_path (str): Path to the model configuration file.
        model_def_path (str): Path to the model definition file.

    Returns:
        argparse.Namespace: Loaded configuration.
    """
    args = argparse.Namespace()

    with open(env_config_path, "r") as f:
        env_config = json.load(f)
    for k, v in env_config.items():
        setattr(args, k, v)

    with open(model_config_path, "r") as f:
        model_config = json.load(f)
    for k, v in model_config.items():
        setattr(args, k, v)
        

    with open(model_def_path, "r") as f:
        model_def = json.load(f)
    for k, v in model_def.items():
        setattr(args, k, v)

    return args


def initialize_distributed(num_gpus: int) -> tuple:
    """
    Initialize distributed training.

    Returns:
        tuple: local_rank, world_size, and device.
    """
    if torch.cuda.is_available() and num_gpus > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        world_size = 1
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    return local_rank, world_size, device

def run_torchrun(module, module_args, num_gpus=1):
    num_nodes = 1

    # temp JSON path for outputs
    with tempfile.TemporaryDirectory() as tmpd:
        out_index = os.path.join(tmpd, "outputs.json")
        full_args = module_args + ["--out_index", out_index]

        cmd = [
            "torchrun",
            "--nproc_per_node", str(num_gpus),
            "--nnodes", str(num_nodes),
            "-m", module,
        ] + full_args

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=env
        )

        try:
            # stream stdout
            for line in iter(proc.stdout.readline, ""):
                if not line and proc.poll() is not None:
                    break
                if line:
                    print(line.rstrip())
        finally:
            stdout, stderr = proc.communicate()
            if stdout:
                print(stdout, end="")
            if stderr:
                print(stderr, end="")

        # collect result
        if os.path.exists(out_index):
            with open(out_index) as f:
                return json.load(f)  # list of per-rank paths
        return None