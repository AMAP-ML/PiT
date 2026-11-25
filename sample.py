"""
Sampling script for denoising diffusion model.
Supports conditional, balanced-conditional, and CFG-guided sampling.
Uses native torch.distributed for multi-GPU inference.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import math
import functools
import tempfile

import torch
import torch.distributed as dist
from absl import flags, app, logging
from ml_collections import config_flags
from torchvision.utils import save_image
from tqdm import tqdm

from lib.utils.dist import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    barrier,
    set_seed,
)
from lib.train_state import (
    TrainState,
    create_model,
    create_diffusion,
)
from lib.dataset import create_dataloader
from lib.utils.helper import set_logger

@torch.no_grad()
def generate(args):
    """Main sampling function."""
    # Initialize distributed
    rank, local_rank, world_size, device = setup_distributed()
    set_seed(args.seed, device_specific=True)

    if is_main_process():
        set_logger(log_level="info")
    else:
        # logging.set_verbosity(logging.ERROR)
        set_logger(log_level="error")

    logging.info(f"Process {rank} using device: {device}, world size: {world_size}")

    # Create dataset (for label sampling and unpreprocessing)
    args.dataset.batch_size = args.sample.mini_batch_size * world_size
    dataset, data_loader = create_dataloader(**args.dataset, verbose=False, label_only=True)

    # Create model and load checkpoint
    model = create_model(args.model_name, **args.model_params).eval().to(device)
    ema_model = create_model(args.model_name, **args.model_params).eval().to(device)
    logging.info(f"Model parameters: {model.get_num_param()}")

    train_state = TrainState(
        step=0, optimizer=None, lr_scheduler=None,
        nnet=model, nnet_ema=ema_model,
    )

    # Load checkpoint
    assert os.path.exists(args.path), f"Checkpoint path does not exist: {args.path}"
    train_state.load(args.path, load_step=True)
    logging.info(f"Loaded checkpoint from {args.path}, step={train_state.step}")

    # Create diffusion
    diffusion = create_diffusion(**args.diffusion, device=device)

    # Prepare data generator for balanced-cond mode
    def get_data_generator():
        while True:
            for data in data_loader:
                yield data

    data_generator = get_data_generator()

    # Build forward function with optional CFG
    def ema_forward(x, t, **model_kwargs):
        return ema_model(x, t, **model_kwargs)

    def cfg_forward(x, t, **model_kwargs):
        """Classifier-Free Guidance with interval-based sigma thresholds."""
        cond = ema_model(x, t, **model_kwargs)

        if args.diffusion.shift_sigma:
            sigma_low = args.sample.cfg_sigma_low * x.shape[-2] / 64
            sigma_high = args.sample.cfg_sigma_high * x.shape[-2] / 64
        else:
            sigma_low = args.sample.cfg_sigma_low
            sigma_high = args.sample.cfg_sigma_high

        # Recover unscaled sigma from rescaled timestep
        t_value = t[0].cpu().item()
        unscaled_t = math.exp(t_value / 250) - 1e-44

        if unscaled_t > sigma_low and unscaled_t <= sigma_high:
            uncond_kwargs = dataset.get_uncond_token(x.shape[0], device=x.device)
            uncond = ema_model(x, t, **uncond_kwargs)
            return uncond + args.sample.cfg_scale * (cond - uncond)
        else:
            return cond

    # Select forward function
    if args.sample.cfg_scale > 0:
        forward_fn = cfg_forward
        logging.info(f"Using CFG with scale={args.sample.cfg_scale}, "
                     f"sigma_low={args.sample.cfg_sigma_low}, sigma_high={args.sample.cfg_sigma_high}")
    else:
        forward_fn = ema_forward
        logging.info("Using simple EMA forward (no guidance)")

    logging.info(f"sampler: {args.sample.sampler}, step: {args.sample.sampling_step}")
    # Sampling loop
    def run_sampling(output_path):
        steps = args.sample.sampling_step
        batch_size = args.sample.mini_batch_size
        sampler = args.sample.sampler

        global_batch_size = batch_size * world_size
        num_batches = args.sample.num_samples // global_batch_size
        remainder = args.sample.num_samples % global_batch_size
        batch_size_list = [batch_size] * num_batches
        if remainder > 0:
            batch_size_list.append(batch_size)

        sample_fn = functools.partial(
            diffusion.sample,
            model=forward_fn,
            shape=dataset.data_shape(),
            steps=steps,
            sampler=sampler,
            return_sample_traj=args.sample.save_sample_traj,
        )

        idx = 0
        for batch_idx, current_batch_size in enumerate(tqdm(batch_size_list, disable=not is_main_process())):
            if current_batch_size == 0:
                continue

            # Prepare conditioning
            labels = next(data_generator)
            if labels is not None:
                labels = labels.to(device) if not labels.is_cuda else labels
            model_kwargs = dict(y=labels[:current_batch_size])


            # Generate samples
            mini_sample = sample_fn(batch_size=current_batch_size, model_kwargs=model_kwargs)

            # Gather across processes
            if world_size > 1:
                gathered_samples = [torch.zeros_like(mini_sample) for _ in range(world_size)]
                dist.all_gather(gathered_samples, mini_sample)
                all_samples = torch.cat(gathered_samples, dim=0)
            else:
                all_samples = mini_sample

            # Unpreprocess and save
            all_samples = dataset.unpreprocess(all_samples)
            all_samples = all_samples.cpu()

            if args.sample.save_sample_traj:
                if is_main_process():
                    os.makedirs(output_path, exist_ok=True)
                    torch.save(all_samples, os.path.join(output_path, f"traj_{batch_idx}.pt"))
            else:
                # Trim to exact number of remaining samples
                remaining = args.sample.num_samples - idx
                all_samples = all_samples[:min(remaining, global_batch_size)]
                if is_main_process():
                    os.makedirs(output_path, exist_ok=True)
                    for sample in all_samples:
                        save_image(sample, os.path.join(output_path, f"{idx}.png"))
                        idx += 1

        logging.info(f"Generated {idx} samples in total.")
        # Compute FID
        if is_main_process() and hasattr(args.dataset, "fid_stat_path") and args.dataset.fid_stat_path:
            try:
                from evaluations.fid_score import calculate_fid_given_paths
                logging.info("Computing FID score...")
                fid = calculate_fid_given_paths((args.dataset.fid_stat_path, output_path))
                logging.info(f"FID score: {fid}")
            except ImportError:
                logging.warning("FID evaluation not available (evaluations module not found)")
            except Exception as error:
                logging.warning(f"FID computation failed: {error}")

    # Run sampling
    output_path = getattr(args.sample, "path", None)
    if output_path:
        run_sampling(output_path)
    else:
        with tempfile.TemporaryDirectory() as temp_path:
            run_sampling(temp_path)

    barrier()
    cleanup_distributed()


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Sampling configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])


def main(argv):
    config = FLAGS.config
    generate(config)


if __name__ == "__main__":
    app.run(main)
