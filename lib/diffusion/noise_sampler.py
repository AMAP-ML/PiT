"""
Simplified noise schedule sampler.
Only contains LogNormalSampler for lognormal_continuous sampling.
"""

import logging
import torch


def create_schedule_sampler(name, **kwargs):
    """Create a schedule sampler by name."""
    logging.info(f"Using noise proposal distribution: {name}")
    if name in ("lognormal", "lognormal_continuous"):
        return LogNormalSampler(**kwargs)
    elif name == "logitnormal":
        return LogitNormalSampler(**kwargs)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


class LogNormalSampler:
    """Log-normal noise level sampler for continuous diffusion training."""

    def __init__(self, p_mean=-1.2, p_std=1.6, **kwargs):
        self.p_mean = p_mean
        self.p_std = p_std
        logging.info(f"LogNormal sampler params: mean={p_mean}, std={p_std}")

    def sample(self, batch_size, device, **kwargs):
        log_sigmas = self.p_mean + self.p_std * torch.randn(batch_size, device=device)
        sigmas = torch.exp(log_sigmas)
        return sigmas

class LogitNormalSampler:
    def __init__(self, p_mean=-1.2, p_std=1.6, **kwargs):
        self.p_mean = p_mean
        self.p_std = p_std
        logging.info(f"LogitNormal sampler params: mean={p_mean}, std={p_std}")

    def sample(self, batch_size, device, **kwargs):
        z = torch.randn(batch_size, device=device) * self.p_std + self.p_mean
        return torch.sigmoid(z)