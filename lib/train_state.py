"""
Simplified training utilities for denoising diffusion.
Only contains TrainState, EMA, optimizer, and LR scheduler.
"""

import os
import math
import collections

import torch
import torch.nn as nn
from absl import logging
from torch.optim.lr_scheduler import LambdaLR

from lib.model import model_cls
from lib.diffusion.denoiser import KarrasDenoiser
from lib.utils.dist import is_main_process


class MetricLogger:
    """Running average metric logger."""

    def __init__(self):
        self.metrics = collections.OrderedDict()

    def update(self, metric_of_the_step: dict):
        for key, value in metric_of_the_step.items():
            if isinstance(value, list):
                if len(value) == 0:
                    continue
                value = sum(value) / len(value)
            if key not in self.metrics:
                self.metrics[key] = {"value": 0, "cnt": 0}
            old_value = self.metrics[key]["value"]
            count = self.metrics[key]["cnt"]
            self.metrics[key]["value"] = old_value * count / (count + 1) + value / (count + 1)
            self.metrics[key]["cnt"] += 1

    def get(self, key=None):
        if key is None:
            return {k: self.metrics[k]["value"] for k in sorted(self.metrics.keys())}
        return self.metrics[key]["value"]

    def clean(self):
        self.metrics = collections.OrderedDict()


def ema_update(model_dest: nn.Module, model_src: nn.Module, rate: float):
    """Exponential moving average update: dest = rate * dest + (1 - rate) * src."""
    param_dict_src = dict(model_src.named_parameters())
    for param_name, param_dest in model_dest.named_parameters():
        param_src = param_dict_src[param_name]
        assert param_src is not param_dest
        param_dest.detach().mul_(rate).add_(param_src, alpha=1 - rate)


def create_model(model_name, **kwargs):
    """Create denoising ViT model."""
    return model_cls[model_name](**kwargs)


def create_diffusion(**kwargs):
    """Create Karras denoiser."""
    return KarrasDenoiser(**kwargs)


def _move_optimizer_state_to_device(optimizer):
    """Move optimizer state tensors to the same device as their parameters.

    PyTorch loads optimizer state dicts with map_location='cpu' by default.
    Fused optimizers (e.g. fused AdamW) require state tensors such as
    `state_steps` to live on the same device as the corresponding parameters,
    so we relocate them after loading a checkpoint.
    """
    for param, state in optimizer.state.items():
        if not isinstance(param, torch.Tensor):
            continue
        param_device = param.device
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(param_device)


class TrainState:
    """Training state container with save/load/resume support."""

    def __init__(self, step, optimizer, lr_scheduler, nnet=None, nnet_ema=None):
        self.step = step
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.nnet = nnet
        self.nnet_ema = nnet_ema

    def ema_update(self, rate=0.9999):
        if self.nnet_ema is not None:
            ema_update(self.nnet_ema, self.nnet, rate)

    def save(self, path):
        if not is_main_process():
            return

        os.makedirs(path, exist_ok=True)
        torch.save(self.step, os.path.join(path, "step.pth"))
        for key, val in self.__dict__.items():
            if key in ["optimizer", "lr_scheduler", "nnet", "nnet_ema"] and val is not None:
                torch.save(val.state_dict(), os.path.join(path, f"{key}.pth"))

    def load(self, path, load_step=False):
        logging.info(f"Loading checkpoint from {path}")
        step_ckpt_path = os.path.join(path, "step.pth")

        if load_step and os.path.exists(step_ckpt_path):
            self.step = torch.load(step_ckpt_path, map_location="cpu")
        else:
            logging.info("Step starts from 0")
            self.step = 0

        loaded_compoents = []
        for key, val in self.__dict__.items():
            if key in ["nnet", "nnet_ema"] and val is not None:
                ckpt_file = os.path.join(path, f"{key}.pth")
                if not os.path.exists(ckpt_file):
                    logging.info(f"File not found, skip loading {key}")
                    continue
                state_dict = torch.load(ckpt_file, map_location="cpu")
                missing, unexpected = val.load_state_dict(state_dict, strict=False)
                loaded_compoents.append(key)
                if missing:
                    logging.info(f"Missing keys when loading {key}: {missing}")
                if unexpected:
                    logging.info(f"Unexpected keys when loading {key}: {unexpected}")

            if key in ["optimizer", "lr_scheduler"] and val is not None and load_step:
                ckpt_file = os.path.join(path, f"{key}.pth")
                if not os.path.exists(ckpt_file):
                    logging.info(f"File not found, skip loading {key}")
                    continue
                try:
                    state_dict = torch.load(ckpt_file, map_location="cpu")
                    val.load_state_dict(state_dict)
                    if key == "optimizer":
                        _move_optimizer_state_to_device(val)
                    loaded_compoents.append(key)
                except Exception as error:
                    logging.info(f"Failed to load {key}, skip. Error: {error}")
        
        return loaded_compoents

    def resume(self, ckpt_root, step=None):
        if not os.path.exists(ckpt_root):
            logging.info("Training from scratch")
            return False
        if step is None:
            ckpts = [x for x in os.listdir(ckpt_root) if ".ckpt" in x and "latest" not in x]
            logging.info(f"Found checkpoints: {ckpts}")
            if len(ckpts) == 0:
                if os.path.exists(os.path.join(ckpt_root, "latest.ckpt")):
                    step = "latest"
                else:
                    return False
            else:
                # if current ckpts is at step larger than that of latest.ckpt
                steps = [int(x.split(".")[0]) for x in ckpts]
                step = max(steps) # by default, load the max-step checkpoint
                max_step = torch.load(
                    os.path.join(ckpt_root, f"{step}.ckpt/step.pth")
                )
                if os.path.exists(os.path.join(ckpt_root, "latest.ckpt")):
                    latest_step = torch.load(
                        os.path.join(ckpt_root, f"latest.ckpt/step.pth")
                    )
                    if max_step < latest_step:
                        step = "latest" # load the latest step checkpoint

        ckpt_path = os.path.join(ckpt_root, f"{step}.ckpt")
        logging.info(f"Resume from {ckpt_path}")
        self.load(ckpt_path, load_step=True)

        return True

def customized_lr_scheduler(optimizer, name="warmup", warmup_steps=-1, total_training_steps=100000, min_scale=-1):
    """Create learning rate scheduler with warmup."""
    if name == "warmup-cosine":
        def lr_lambda(step):
            if warmup_steps > 0:
                if step <= warmup_steps:
                    return min(step / warmup_steps, 1)
                elif step <= total_training_steps:
                    lr_scale = 0.5 * (1 + math.cos((step - warmup_steps) * math.pi / (total_training_steps - warmup_steps)))
                    if min_scale != -1:
                        lr_scale = max(lr_scale, min_scale)
                    return lr_scale
                else:
                    return min_scale if min_scale != -1 else 0
            return 1
    elif name == "warmup":
        def lr_lambda(step):
            if warmup_steps > 0 and step <= warmup_steps:
                return min(step / warmup_steps, 1)
            return 1
    else:
        raise NotImplementedError(f"Unknown scheduler: {name}")

    return LambdaLR(optimizer, lr_lambda)


def set_weight_decay(model, lr, weight_decay):
    """Separate params into weight_decay and no_weight_decay groups."""
    param_groups = {}
    no_decay_names = model.no_weight_decay()

    for param_name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        group_name = "no_wd" if param_name in no_decay_names else "wd"
        if group_name not in param_groups:
            param_groups[group_name] = {
                "lr": lr,
                "weight_decay": 0 if param_name in no_decay_names else weight_decay,
                "params": [],
            }
        param_groups[group_name]["params"].append(param)

    return list(param_groups.values())
