import warnings
warnings.filterwarnings("ignore")

import os
import time
import shutil
import functools
import tempfile

import torch
import torch._dynamo
import torch.nn as nn
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
    broadcast_params,
    wrap_model_ddp,
    set_seed,
    GradScalerWrapper,
)
from lib.utils.helper import set_logger
from lib.train_state import (
    MetricLogger,
    TrainState,
    create_model,
    create_diffusion,
    customized_lr_scheduler,
    set_weight_decay,
)
from lib.dataset import create_dataloader


def train(args):
    torch.backends.cudnn.benchmark = True

    # Settings for torch.compile
    torch.set_float32_matmul_precision('highest') # Prevent torch.compile from downcasting FP32 to TF32
    torch._inductor.config.fallback_random = True  # Avoids Triton custom RNG discrepancies
    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False
    torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = False

    # Initialize distributed training
    rank, local_rank, world_size, device = setup_distributed()
    mixed_precision = getattr(args, "mixed_precision", "bf16")
    grad_scaler = GradScalerWrapper(mixed_precision=mixed_precision)
    set_seed(args.seed, device_specific=True)

    if is_main_process():
        os.makedirs(args.workdir, exist_ok=True)
        os.makedirs(args.sample_dir, exist_ok=True)
        set_logger(log_level="info", fname=os.path.join(args.workdir, "output.log"))
        logging.info(args)
    else:
        os.makedirs(args.workdir, exist_ok=True)
        set_logger(log_level="error", fname=os.path.join(args.workdir, "error.log"))

    logging.info(f"Training in: {mixed_precision} mode")

    # Create dataset and dataloader
    dataset, dataloader = create_dataloader(**args.dataset)

    # Create model
    model = create_model(args.model_name, **args.model_params).train().to(device)
    if world_size > 1:
        logging.info("Synchronize online model parameters within DP group...")
        broadcast_params(model)

    ema_model = create_model(args.model_name, **args.model_params).eval().to(device)
    logging.info(f"Model parameters: {model.get_num_param()}")

    # Create optimizer
    param_groups = set_weight_decay(model, lr=args.optimizer.lr, weight_decay=args.optimizer.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, **args.optimizer)
    lr_scheduler = customized_lr_scheduler(optimizer, **args.lr_scheduler)

    # Create train state
    train_state = TrainState(
        step=0, optimizer=optimizer, lr_scheduler=lr_scheduler,
        nnet=model, nnet_ema=ema_model,
    )
    logging.info("Synchronize EMA model parameters with online model..")
    train_state.ema_update(0.0)

    logging.info(f"Try to resume from {args.workdir}...")
    train_state.resume(args.workdir)

    # Wrap model with DDP and compile
    ddp_model = wrap_model_ddp(model, device) if world_size > 1 else model

    # Create diffusion
    diffusion = create_diffusion(**args.diffusion, device=device)

    # Training step
    def train_step(batch):
        x_start, model_kwargs = batch
        batch_size = x_start.shape[0]
        sigmas = diffusion.sample_schedule(batch_size)

        with grad_scaler.autocast():
            ret = diffusion.denoise_training_step(
                ddp_model, x_start,
                sigmas=sigmas,
                model_kwargs=model_kwargs,
            )
            loss = ret["loss"].mean()

        optimizer.zero_grad(set_to_none=True)
        grad_scaler.backward(loss)

        if args.train.gradclip > 0:
            grad_scaler.unscale(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.train.gradclip)

        grad_scaler.step(optimizer)
        torch.cuda.synchronize()

        if grad_scaler.step_was_skipped:
            logging.info("Found Inf grad, skip this iteration")
            return None

        lr_scheduler.step()
        train_state.ema_update(args.train.ema_rate)
        train_state.step += 1

        # logging
        param_norm = torch.norm(torch.stack([p.data.norm(2) for p in model.parameters()]), 2).item()
        grad_norm = torch.norm(torch.stack([p.grad.norm(2) for p in model.parameters() if p.grad is not None]), 2).item()

        sub_loss = {f"loss_q{i}": [] for i in range(4)}

        per_sample_loss = ret["loss"].detach()

        for i, sigma in enumerate(sigmas):
            r = min(sigma / diffusion.sigma_max, 1)
            quarter = int(r * 4)
            quarter = min(quarter, 3)
            sub_loss[f"loss_q{quarter}"].append(per_sample_loss[i].item())

        return {"loss": loss.item(), "grad_norm": grad_norm, "param_norm": param_norm, **sub_loss}

    # Sampling step
    @torch.no_grad()
    def sample_step(step):
        steps = args.sample.sampling_step
        batch_size = args.sample.mini_batch_size
        sampler = args.sample.sampler

        def ema_forward(x, t, **kwargs):
            return train_state.nnet_ema(x, t, **kwargs)

        model_kwargs = dataset.sample_label(batch_size, device=device)

        with grad_scaler.autocast():
            samples = diffusion.sample(
                ema_forward, batch_size, dataset.data_shape(),
                steps, sampler, model_kwargs=model_kwargs,
            )

        samples = dataset.unpreprocess(samples)
        save_path = os.path.join(args.sample_dir, f"{step}.jpg")
        tmp_path = ""
        try:
            os.makedirs(args.sample_dir, exist_ok=True)
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                tmp_path = tmp_file.name
            save_image(samples, tmp_path, nrow=10)
            shutil.move(tmp_path, save_path)
        except (PermissionError, OSError) as error:
            logging.info(f"Saving images failed: {error}")
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    # FID evaluation step
    @torch.no_grad()
    def evaluate_fid(step):

        fid_stat_path = getattr(args.dataset, "fid_stat_path", "")
        if not fid_stat_path or not os.path.exists(fid_stat_path):
            logging.warning(f"fid_stat_path not set or not found: '{fid_stat_path}', skipping FID.")
            return

        """Generate 10K samples across all GPUs and compute FID on rank 0."""
        eval_num_samples = 10000
        eval_sampling_steps = 50
        eval_batch_size = args.sample.mini_batch_size
        global_batch_size = eval_batch_size * world_size

        num_batches = eval_num_samples // global_batch_size
        remainder = eval_num_samples % global_batch_size
        # Add one extra batch if there's remainder, to ensure >= eval_num_samples
        if remainder > 0:
            num_batches += 1

        def ema_forward(x, t, **kwargs):
            return train_state.nnet_ema(x, t, **kwargs)

        sample_fn = functools.partial(
            diffusion.sample,
            model=ema_forward,
            shape=dataset.data_shape(),
            steps=eval_sampling_steps,
            sampler="euler",
        )

        temp_dir = tempfile.mkdtemp() if is_main_process() else None
        logging.info(f"[Rank {rank}] FID evaluation: generating {eval_num_samples} samples "
                     f"({num_batches} batches, batch_size={eval_batch_size})")

        saved_count = 0
        for batch_idx in tqdm(range(num_batches), disable=not is_main_process(), desc="FID sampling"):
            model_kwargs = dataset.sample_label(eval_batch_size, device=device)

            with grad_scaler.autocast():
                mini_samples = sample_fn(batch_size=eval_batch_size, model_kwargs=model_kwargs)

            mini_samples = dataset.unpreprocess(mini_samples)
            # Gather samples to rank0
            if world_size > 1:
                gathered = [torch.zeros_like(mini_samples) for _ in range(world_size)] if is_main_process() else None
                dist.gather(mini_samples, gathered, dst=0)

                # Only rank 0 saves images
                if is_main_process():
                    all_samples = torch.cat(gathered, dim=0)
                    all_samples = all_samples.cpu()
                    # Track global progress (consistent across all ranks)
                    batch_start = batch_idx * global_batch_size
                    remaining = eval_num_samples - batch_start
                    all_samples = all_samples[:remaining]

                    for sample in all_samples:
                        save_image(sample, os.path.join(temp_dir, f"{saved_count}.png"))
                        saved_count += 1

            else:
                all_samples = mini_samples

        barrier()
        # Compute FID on rank 0
        fid_value = None
        if is_main_process():
            try:
                from evaluations.fid_score import calculate_fid_given_paths
                logging.info(f"Computing FID score from {saved_count} samples...")
                fid_value = calculate_fid_given_paths(
                    (fid_stat_path, temp_dir),
                )
                logging.info(f"Step {step} | FID: {fid_value}")

                # Append to eval.log
                eval_log_path = os.path.join(args.workdir, "eval.log")
                with open(eval_log_path, "a+") as log_file:
                    log_file.write(f"step={step}, fid={fid_value}\n")
            except ImportError:
                logging.warning("FID evaluation not available (evaluations module not found)")
            except Exception as error:
                logging.warning(f"FID computation failed: {error}")

            # Cleanup temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)

        barrier()
        return fid_value

    # Main training loop
    metric_logger = MetricLogger()
    # logging.info("Training started...")

    while train_state.step < args.train.total_training_steps:
        start_time = time.time()
        batch = next(dataloader)
        data_time = time.time() - start_time

        start_time = time.time()
        metrics = train_step(batch)
        forward_time = time.time() - start_time

        if metrics is not None:
            metrics["data_time"] = data_time
            metrics["forward_time"] = forward_time
            metric_logger.update(metrics)

        # Save checkpoint
        if train_state.step != 0:
            if train_state.step % args.train.save_interval == 0:
                save_dir = os.path.join(args.workdir, f"{train_state.step}.ckpt")
                train_state.save(save_dir)
                torch.cuda.empty_cache()
            elif train_state.step % 10000 == 0:
                save_dir = os.path.join(args.workdir, "latest.ckpt")
                train_state.save(save_dir)
                torch.cuda.empty_cache()

        # Log metrics
        if train_state.step % args.train.log_interval == 0 and is_main_process():
            logging.info(dict(
                step=train_state.step,
                lr=lr_scheduler.get_last_lr(),
                **{k: f"{v:.6g}" for k, v in metric_logger.get().items()},
            ))
            metric_logger.clean()
        # Sample (visualization, rank 0 only)
        if train_state.step != 0 and train_state.step % args.train.sample_interval == 0 and is_main_process():
            sample_step(train_state.step)

        # FID evaluation (all ranks participate in sampling, rank 0 computes FID)
        eval_interval = getattr(args.train, "eval_interval", 100000)
        if train_state.step != 0 and train_state.step % eval_interval == 0:
            evaluate_fid(train_state.step)

    cleanup_distributed()

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=False)
flags.DEFINE_string("workdir", None, "Work unit directory.")
flags.mark_flags_as_required(["config"])


def main(argv):
    config = FLAGS.config
    config.workdir = FLAGS.workdir
    config.sample_dir = os.path.join(config.workdir, "samples")
    train(config)


if __name__ == "__main__":
    app.run(main)