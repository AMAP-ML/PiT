"""
Simplified Karras denoiser for denoising diffusion training.
Only supports denoise mode with x0 prediction and euler/heun sampling.
"""

import math
import torch
import torch.nn.functional as F

from .noise_sampler import create_schedule_sampler


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    if not isinstance(x, torch.Tensor):
        return x
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}")
    return x[(...,) + (None,) * dims_to_append]


class KarrasDenoiser:
    """
    Karras-style denoiser for diffusion model training and sampling.
    Supports x0 prediction with ICM weighting and lognormal_continuous sampling.
    """

    def __init__(
        self,
        sigma_data=0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        schedule_sampler="lognormal_continuous",
        num_timesteps=1280,
        sample_param=None,
        device="cuda",
        shift_sigma=True,
        **kwargs,
    ):
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.rho = rho
        self.shift_sigma = shift_sigma
        self.device = device

        self.time_sample_schedule = schedule_sampler
        sample_param = sample_param or {}
        self.schedule_sampler = create_schedule_sampler(schedule_sampler, **sample_param)

        self.num_timesteps = num_timesteps
        self.sigmas = self._build_sigma_schedule(self.num_timesteps)

    def _build_sigma_schedule(self, num_timesteps):
        """Build discrete sigma schedule for ICM weighting lookup."""
        sigmas = torch.tensor([
            (self.sigma_max ** (1 / self.rho) + idx / (num_timesteps - 1) *
             (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))) ** self.rho
            for idx in range(num_timesteps)
        ]).to(self.device)
        return sigmas

    def sample_schedule(self, batch_size, step=None):
        """Sample noise levels from the schedule."""
        return self.schedule_sampler.sample(batch_size, self.device, step=step)

    def find_nearest(self, t):
        """Find nearest discrete index for continuous sigma values."""
        idx = (t[:, None].expand(-1, self.num_timesteps) -
               self.sigmas[None, :].expand(t.shape[0], -1)).abs().argmin(dim=1)
        return idx

    def get_scalings(self, sigma, height=256, width=256):
        """Compute c_skip, c_out, c_in for boundary condition."""
        sigma_min_shifted = self.sigma_min * math.sqrt(height / 64 * width / 64) if self.shift_sigma else self.sigma_min
        c_skip = self.sigma_data ** 2 / ((sigma - sigma_min_shifted) ** 2 + self.sigma_data ** 2)
        c_out = ((sigma - sigma_min_shifted) * self.sigma_data /
                 (sigma ** 2 + self.sigma_data ** 2) ** 0.5)
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in

    def get_icm_weightings(self, idx):
        """Compute ICM loss weightings from discrete indices."""
        sigmas1 = (self.sigma_max ** (1 / self.rho) + idx / (self.num_timesteps - 1) *
                   (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))) ** self.rho
        sigmas2 = (self.sigma_max ** (1 / self.rho) + (idx + 1) / (self.num_timesteps - 1) *
                   (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))) ** self.rho
        weightings = 1 / (sigmas1 - sigmas2)
        weightings[sigmas1 == sigmas2] = 0.0
        return weightings

    def model_forward(self, model, x_t, sigmas, **model_kwargs):
        """Run model with proper input scaling and output parameterization."""
        c_skip, c_out, c_in = self.get_scalings(sigmas, height=x_t.shape[-2], width=x_t.shape[-1])
        rescaled_t = 1000 * 0.25 * torch.log(sigmas + 1e-44)
        model_output = model(append_dims(c_in, x_t.ndim) * x_t, rescaled_t, **model_kwargs)

        return model_output

    def denoise_training_step(
        self,
        model,
        x_start,
        sigmas=None,
        model_kwargs=None,
        **kwargs,
    ):
        """
        Single denoising training step.
        Returns dict with loss tensor.
        """
        if model_kwargs is None:
            model_kwargs = {}

        dims = x_start.ndim
        height, width = x_start.shape[-2:]  # 4D (B,C,H,W) and 5D (B,C,T,H,W) safe

        idx = self.find_nearest(sigmas)
        idx = torch.clamp(idx, max=self.num_timesteps - 1)

        if self.shift_sigma:
            sigmas = sigmas * math.sqrt(height * width / 64**2)

        # Add noise
        noise = torch.randn_like(x_start)
        x_t = x_start + noise * append_dims(sigmas, dims)

        # Model prediction
        denoised_output = self.model_forward(model, x_t, sigmas, **model_kwargs)

        # Compute loss
        target = x_start

        loss = F.mse_loss(
            denoised_output.float(),
            target,
            reduction="none"
        ).flatten(1).mean(dim=1)


        weightings = self.get_icm_weightings(append_dims(idx, loss.ndim))
        loss = weightings * loss

        return {"loss": loss}

    def sample(self, model, batch_size, shape, steps, sampler="euler",
               return_sample_traj=False, model_kwargs=None, **kwargs):
        """Generate samples using ODE solvers."""
        if model_kwargs is None:
            model_kwargs = {}

        x_T = torch.randn((batch_size, *shape), device=self.device)

        def denoiser(x, t):
            denoised = self.model_forward(model, x, t, **model_kwargs)
            return denoised.clamp(-1, 1)

        def get_sigmas_karras(n):
            ramp = torch.linspace(0, 1, n)
            min_inv_rho = self.sigma_min ** (1 / self.rho)
            max_inv_rho = self.sigma_max ** (1 / self.rho)
            sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
            if self.shift_sigma:
                sigmas = sigmas * math.sqrt(shape[-2]*shape[-1]/ 64**2)
            return sigmas

        x_T = x_T * self.sigma_max
        if self.shift_sigma:
            # shift according to image size 64
            x_T = x_T * math.sqrt(shape[-2]*shape[-1]/ 64**2)

        sigmas = get_sigmas_karras(steps).to(self.device)

        def x0_to_d(x, sigma, denoised):
            """Converts x0-prediction denoiser output to Karras ODE derivative."""
            return (x - denoised) / append_dims(sigma, x.ndim)

        samplers = {
            "euler": sample_euler,
            "heun": sample_heun,
        }
        sample_fn = samplers[sampler]
        return sample_fn(denoiser, x_T, sigmas,
            return_sample_traj=return_sample_traj,
            x0_to_d=x0_to_d
        )


def sample_euler(denoiser, x, sigmas, return_sample_traj=False, x0_to_d=None, **kwargs):
    """Euler ODE sampler."""
    s_in = x.new_ones([x.shape[0]])
    all_x = []

    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        denoised = denoiser(x, sigma * s_in)
        d = x0_to_d(x, sigma, denoised)
        dt = sigmas[i + 1] - sigma
        x = x + d * dt
        if return_sample_traj:
            all_x.append(x)

    if return_sample_traj:
        return torch.cat(all_x, dim=0)
    return x


def sample_heun(denoiser, x, sigmas, return_sample_traj=False, x0_to_d=None, **kwargs):
    """Heun (2nd order) ODE sampler."""
    s_in = x.new_ones([x.shape[0]])
    all_x = []

    for i in range(len(sigmas) - 1):
        sigma_hat = sigmas[i]
        denoised = denoiser(x, sigma_hat * s_in)
        d = x0_to_d(x, sigma_hat, denoised)
        dt = sigmas[i + 1] - sigma_hat

        if sigmas[i + 1] == 0:
            x = x + d * dt
        else:
            x_2 = x + d * dt
            denoised_2 = denoiser(x_2, sigmas[i + 1] * s_in)
            d_2 = x0_to_d(x_2, sigmas[i + 1], denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt

        if return_sample_traj:
            all_x.append(x)

    if return_sample_traj:
        return torch.cat(all_x, dim=0)
    return x
