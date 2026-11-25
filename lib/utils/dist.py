"""
Distributed training utilities using torch.distributed.
Replaces HuggingFace Accelerate with native PyTorch distributed primitives.
"""

import os
import random
import numpy as np

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_distributed(timeout_seconds=3600):
    """
    Initialize the distributed process group using environment variables set by torchrun.
    Returns (rank, local_rank, world_size, device).
    """
    dist.init_process_group(backend="nccl", timeout=__import__("datetime").timedelta(seconds=timeout_seconds))
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    return rank, local_rank, world_size, device


def cleanup_distributed():
    """Destroy the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if the current process is the main process (rank 0)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank():
    """Get the rank of the current process."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    """Get the total number of processes."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def barrier():
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


def broadcast_params(model):
    """Broadcast model parameters and buffers from rank 0 to all processes."""
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    for buffer in model.buffers():
        dist.broadcast(buffer, src=0)


def wrap_model_ddp(model, device, find_unused_parameters=False):
    """Wrap a model with DistributedDataParallel."""
    return DDP(
        model,
        device_ids=[device.index] if device.type == "cuda" else None,
        find_unused_parameters=find_unused_parameters,
    )

def set_seed(seed, device_specific=True):
    """Set random seed for reproducibility, optionally offset by rank."""
    if device_specific:
        seed = seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GradScalerWrapper:
    """
    Wrapper around torch.amp.GradScaler for bf16/fp16 mixed precision training.
    For bf16, grad scaling is not needed, so this acts as a passthrough.
    """

    def __init__(self, mixed_precision="bf16"):
        self.mixed_precision = mixed_precision
        self.enabled = mixed_precision == "fp16"
        if self.enabled:
            self.scaler = torch.amp.GradScaler(
                "cuda",
                init_scale=2**14,
                growth_factor=1.0006933874625807,
                growth_interval=2,
                backoff_factor=0.2,
            )
        else:
            self.scaler = None
        self._step_was_skipped = False

    def get_autocast_dtype(self):
        """Return the dtype to use in autocast context."""
        if self.mixed_precision == "bf16":
            return torch.bfloat16
        elif self.mixed_precision == "fp16":
            return torch.float16
        return torch.float32

    def autocast(self):
        """Return the autocast context manager."""
        return torch.cuda.amp.autocast(dtype=self.get_autocast_dtype())

    def backward(self, loss):
        """Scale loss and call backward."""
        if self.enabled:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self, optimizer):
        """Optimizer step with optional grad scaling."""
        self._current_optimizer = optimizer
        if self.enabled:
            self.scaler.step(optimizer)
            self._step_was_skipped = any(
                v.item() for v in self.scaler._found_inf_per_device(optimizer).values()
            )
            self.scaler.update()
        else:
            optimizer.step()
            self._step_was_skipped = False

    def unscale(self, optimizer):
        """Unscale gradients for the given optimizer."""
        self._current_optimizer = optimizer
        if self.enabled:
            self.scaler.unscale_(optimizer)

    @property
    def step_was_skipped(self):
        """Whether the last optimizer step was skipped due to inf/nan grads."""
        return self._step_was_skipped
