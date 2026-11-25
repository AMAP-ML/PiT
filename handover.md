# PiT Coding-Agent Handover

This is the operational guide for the current open-source PiT tree. It mirrors the setup, data preparation, training, inference, and FID workflow in the EPG README, but describes the code that is actually present here.

Last verified: 2026-11-12.

## 1. Release scope and important invariants

The current release supports:

- Class-conditional, pixel-space denoising training on ImageNet-256 and ImageNet-512.
- EMA checkpoints and checkpoint resume.
- Euler and Heun sampling, including classifier-free guidance.
- FID evaluation during training or as a standalone command.

The following are intentionally not part of this release:

- Text-to-image and text-to-video training.
- Representation pretraining, latent diffusion, flow matching, REPA, and consistency training.
- The removed `train_t2i.sh`, `CC12M_t2i.py`, and T2I loader/model modules.

Keep the existing implementation in `lib/model/vit.py`. Its GateBlock/ResGateBlock architecture and `PiT-B/L/XL/G` registry are intentional. Do not replace it with another model implementation or assume that an external source checkout is required at runtime.

The diffusion implementation uses x0 prediction, ICM weighting, a fixed default of 1280 noise-schedule entries, and log-normal continuous noise sampling.

## 2. Repository map

```text
PiT_opensourced/
├── train.py                    # ImageNet training, sampling previews, FID, checkpoints
├── sample.py                   # Distributed ImageNet generation and optional FID
├── configs/
│   ├── IN256.py                # 256px, patch size 16
│   └── IN512.py                # 512px, patch size 32
├── scripts/
│   ├── train.sh                # Configurable 256px launcher
│   ├── train512.sh             # 512px launcher with environment-specific defaults
│   └── sample.sh               # Example 512px sampling launcher
├── lib/
│   ├── model/vit.py            # PiT-B/L/XL/G and model registry
│   ├── diffusion/denoiser.py   # Training loss and Euler/Heun samplers
│   ├── diffusion/noise_sampler.py
│   ├── dataset/imagenet_loader.py
│   ├── train_state.py          # EMA, optimizer, LR scheduler, checkpoint I/O
│   └── utils/                  # Distributed and logging helpers
└── evaluations/
    ├── measure_fid.py          # Recommended PyTorch FID CLI
    ├── prepare_npz.py          # Pack generated PNGs into an NPZ
    ├── evaluator.py            # ADM-compatible TensorFlow evaluator
    └── fid_score.py            # PyTorch FID implementation
```

Run commands from the repository root. Config paths and local imports assume this working directory.

## 3. Environment

### Requirements

The training and sampling smoke tests were verified with Python **3.10.14** and CUDA **12.4**, using these package versions:

| Package | Verified version |
|---|---|
| `torch` | 2.6.0+cu124 |
| `torchvision` | 0.21.0+cu124 |
| `flash-attn` | 2.8.3 |
| `timm` | 1.0.11 |
| `einops` | 0.8.0 |
| `ml_collections` | 1.1.0 |
| `absl-py` | 2.1.0 |
| `blobfile` | 3.2.0 |
| `numpy` | 1.26.4 |
| `scipy` | 1.14.1 |
| `Pillow` | 12.0.0 |
| `tqdm` | 4.66.5 |

Keep `torch`, `torchvision`, FlashAttention, and the CUDA runtime mutually compatible when recreating the environment.

The optional ADM-compatible evaluator also uses `requests` **2.32.3** and TensorFlow. TensorFlow was not installed in the verified environment and is not needed by `evaluations/measure_fid.py`; install a compatible TensorFlow version separately only if you need `evaluations/evaluator.py`.

Training uses NCCL and CUDA. Use `torchrun` even for one GPU; the trainer is not a CPU-only entry point.

## 4. Dataset preparation

- **Dataset:** download ImageNet from the [Hugging Face dataset repository](https://huggingface.co/datasets/jiachenlei/imagenet).
- **Preprocessing code:** use the [`prepare_dataset` tools in AMAP-ML/EPG](https://github.com/AMAP-ML/EPG/tree/master/prepare_dataset).
- **Download FID stats**:
  - in256_fid_train.npz, used in ImageNet-256 training: [Hugging Face Link](https://huggingface.co/jiachenlei/EPG/blob/main/in256_fid_train.npz)
  - in512_fid_train.npz, used in ImageNet-512 training: [Hugging Face Link](https://huggingface.co/jiachenlei/EPG/blob/main/in512_fid_train.npz)

Follow the preprocessing repository's instructions to center-crop and resize the raw images to 256x256 or 512x512 before training. The preprocessing code is external and is not vendored into this repository. The PiT loader does not resize images to match the model config.

Required layout:

```text
ImageNet256/
├── train/
│   ├── n01440764/
│   │   ├── n01440764_0000.png
│   │   ├── n01440764_0001.png
│   │   └── ...
│   ├── n01443537/
│   │   └── ...
│   └── ... 1000 class directories
└── in256_fid_train.npz               # FID computation during training
```

ImageNet-512 uses the same structure with 512x512 images and FID stats.

Important data-contract details:

- Only `train/*/*.png` is indexed.
- The class name is derived from the filename prefix before the first underscore. Keep names such as `n01440764_0000.png`; names such as `0000.png` can collapse different folders into the same derived class.
- Class indices are assigned by sorting the unique derived class names.
- `config.dataset.batch_size` is the global batch size across all ranks.
- Keep `config.dataset.num_workers >= 1`; the loader enables persistent workers.
- Sampling still needs the real training directory to derive class counts and sample labels, even though it does not load training images.
- Use FID statistics computed for the matching image resolution and evaluation protocol.

After preprocessing, verify that the output follows the directory layout and filename contract above before launching training.

## 5. Configs and registered models

| Config | Resolution | Patch size | Default model | Default steps | Global batch |
|---|---:|---:|---|---:|---:|
| `configs/IN256.py` | 256 | 16 | `PiT-XL` | 1,000,000 | 1024 |
| `configs/IN512.py` | 512 | 32 | `PiT-XL` | 1,000,000 | 1024 |

Registered models in `lib/model/vit.py`:

| Name | Encoder/decoder blocks | Width | Heads | Parameters |
|---|---:|---:|---:|---:|
| `PiT-B` | 6 + 6 | 768 | 12 | 122,631,936 |
| `PiT-L` | 10 + 10 | 1024 | 16 | 338,679,000 |
| `PiT-XL` | 15 + 15 | 1152 | 16 | 640,791,264 |
| `PiT-G` | 15 + 15 | 1536 | 16 | 1,562,361,216 |

Any config field can be overridden with `--config.<field>=<value>`. When changing the training duration, update both copies of the total step count:

```bash
--config.train.total_training_steps=200000 \
--config.lr_scheduler.total_training_steps=200000
```

## 6. Training

### ImageNet-256

`scripts/train.sh` defaults to 8 GPUs and `PiT-XL`, but supports environment overrides and forwards additional CLI arguments to the config system.

```bash
NUM_GPUS=8 \
MODEL_NAME=PiT-XL \
DATADIR=/path/to/ImageNet256/train \
FID_STAT_PATH=/path/to/ImageNet256/in256_fid_train.npz \
WORKDIR=/path/to/output/pitxl_in256 \
bash scripts/train.sh
```

Useful launcher variables:

| Variable | Default |
|---|---|
| `NUM_GPUS` | 8 |
| `MASTER_ADDR` | localhost |
| `MASTER_PORT` | 42123 |
| `MODEL_NAME` | PiT-XL |
| `DATADIR` | Set explicitly |
| `FID_STAT_PATH` | Set explicitly |
| `WORKDIR` | `./$EXPNAME` |

Always set `WORKDIR` explicitly. Leaving `EXPNAME` empty otherwise makes the work directory `./`, which writes logs and samples into the repository root.

Additional config overrides can be appended directly:

```bash
NUM_GPUS=8 MODEL_NAME=PiT-B WORKDIR=/path/to/output/pitb_in256 \
DATADIR=/path/to/ImageNet256/train \
FID_STAT_PATH=/path/to/ImageNet256/in256_fid_train.npz \
bash scripts/train.sh \
  --config.train.total_training_steps=200000 \
  --config.lr_scheduler.total_training_steps=200000 \
  --config.dataset.batch_size=256
```

### ImageNet-512

`scripts/train512.sh` accepts the same environment variables and trailing config overrides as `scripts/train.sh`, but uses `configs/IN512.py` and its ImageNet-512 defaults.

```bash
NUM_GPUS=8 \
MODEL_NAME=PiT-XL \
DATADIR=/path/to/ImageNet512/train \
FID_STAT_PATH=/path/to/ImageNet512/in512_fid_train.npz \
WORKDIR=/path/to/output/pitxl_in512 \
bash scripts/train512.sh
```

Set `NUM_GPUS`, `MASTER_ADDR`, and `MASTER_PORT` appropriately for your node. As with the 256px launcher, always set `WORKDIR` and the dataset/FID paths explicitly. Additional arguments are forwarded to `train.py`, so short experiments can override both training and scheduler step counts without editing the script.

### Checkpoints, logs, and resume

The work directory contains:

```text
WORKDIR/
├── output.log
├── samples/
├── <step>.ckpt/
│   ├── step.pth
│   ├── nnet.pth
│   ├── nnet_ema.pth
│   ├── optimizer.pth
│   └── lr_scheduler.pth
└── latest.ckpt/
```

Training automatically resumes from `WORKDIR`. It compares the highest numeric checkpoint with `latest.ckpt` when both are present. `config.path` is used by sampling; the current trainer does not use it to initialize from a separate checkpoint directory.

The training loop writes numeric checkpoints at `config.train.save_interval` and uses `train_state.save()` for both numeric and `latest.ckpt` snapshots at the applicable 10,000-step saves. `train.py` calls `cleanup_distributed()` after a successful run.

## 7. Verified two-step smoke test

Use a free GPU in `CUDA_VISIBLE_DEVICES`. This runs the real `scripts/train.sh` entry point with real ImageNet data, not an import-only test.

```bash
smoke_dir=$(mktemp -d)
echo "Smoke workdir: $smoke_dir"

CUDA_VISIBLE_DEVICES=<gpu_id> \
PYTHONDONTWRITEBYTECODE=1 \
NUM_GPUS=1 \
MODEL_NAME=PiT-B \
WORKDIR="$smoke_dir" \
timeout --signal=INT --kill-after=30s 540s \
bash scripts/train.sh \
  --config.train.total_training_steps=2 \
  --config.lr_scheduler.total_training_steps=2 \
  --config.dataset.batch_size=2 \
  --config.dataset.num_workers=2 \
  --config.train.log_interval=1 \
  --config.train.sample_interval=1000000 \
  --config.train.eval_interval=1000000 \
  --config.train.save_interval=1000000 \
  --config.sample.mini_batch_size=2
```

Inspect the two optimizer steps:

```bash
grep -E "'step': [12]" "$smoke_dir/output.log"
```

Expected behavior:

- The full ImageNet index reports 1,281,167 images with the verified dataset.
- PiT-B reports 122,631,936 parameters.
- Two finite-loss optimizer steps are logged.
- The process exits with status 0 and without an NCCL shutdown warning.
- No checkpoint, preview, or FID run is produced because their intervals were deliberately moved beyond step 2.

The first step includes model compilation and is much slower than the second. Dataset indexing can also take tens of seconds before model construction.

Verified run on 2026-11-12:

```text
step 1: loss=157.529, grad_norm=217.481
step 2: loss=152.919, grad_norm=200
```

These values are evidence from the verified environment, not assertions that every hardware/software stack must reproduce them exactly.

## 8. Sampling and inference

A checkpoint directory must contain at least `nnet_ema.pth`; use the same model name, image size, and patch size that produced the checkpoint.

Example ImageNet-256 sampling command:

```bash
torchrun \
  --nproc_per_node=8 \
  --master_addr=localhost \
  --master_port=12345 \
  sample.py \
  --config=configs/IN256.py \
  --config.path=/path/to/output/pitxl_in256/1000000.ckpt \
  --config.model_name=PiT-XL \
  --config.dataset.data_dir=/path/to/ImageNet256/train \
  --config.dataset.fid_stat_path=/path/to/ImageNet256/in256_fid_train.npz \
  --config.sample.sampler=heun \
  --config.sample.sampling_step=50 \
  --config.sample.num_samples=50000 \
  --config.sample.mini_batch_size=50 \
  --config.sample.cfg_scale=2.5 \
  --config.sample.path=/path/to/generated_images
```

For ImageNet-512, use `configs/IN512.py`, matching ImageNet-512 data/statistics, and an ImageNet-512 checkpoint.

`config.sample.path` behavior:

- Set it to a directory to retain generated PNGs.
- Leave it empty to use a temporary directory for FID; temporary images are removed afterward.

`scripts/sample.sh` is an ImageNet-512 example with Bash arrays for checkpoint paths and CFG scales. Edit its placeholder checkpoint, dataset, and FID paths first, and invoke it with `bash`, not `sh`.

## 9. FID evaluation

### During training

Set:

```bash
--config.dataset.fid_stat_path=/path/to/in256_fid_train.npz or in512_fid_train.npz \
--config.train.eval_interval=50000
```

The trainer generates 10,000 images with 50 Euler steps at each evaluation interval. If the FID path is empty or missing, it logs a warning and skips FID.

Current limitation: the built-in training FID path saves generated images only in the distributed `world_size > 1` branch. Do not rely on built-in FID for a single-GPU training run. Generate images with `sample.py` and use standalone FID instead.

### Recommended standalone PyTorch FID

From the repository root:

```bash
python evaluations/measure_fid.py \
  /path/to/fid_train.npz \
  /path/to/generated_images \
  --device cuda \
  --batch_size 50
```

The arguments can be image directories or precomputed FID statistics containing `mu` and `sigma`.

### ADM-compatible NPZ workflow

```bash
cd evaluations
python prepare_npz.py /path/to/generated_images /path/to/generated.npz
python evaluator.py /path/to/reference.npz /path/to/generated.npz
```

`prepare_npz.py` reads only immediate `*.png` files and holds all images in memory before writing `arr_0`. `evaluator.py` requires TensorFlow and uses the bundled `classify_image_graph_def.pb`.

When evaluating using ADM toolkits, remember to download the ADM official reference stats:
- For Imagenet-256: https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
- For ImageNet-512: https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/512/VIRTUAL_imagenet512.npz

## 10. Validation checklist for future changes

After editing launchers:

```bash
for script in scripts/*.sh; do
  bash -n "$script" || exit 1
done
```

After editing Python:

```bash
python - <<'PY'
from pathlib import Path
for path in Path('.').rglob('*.py'):
    compile(path.read_text(), str(path), 'exec')
print('Python syntax: OK')
PY
```

After changing model, diffusion, optimizer, or checkpoint code:

1. Run the two-step `scripts/train.sh` smoke test above.
2. Confirm two optimizer steps, finite losses/gradients, and a clean exit.
3. For sampling changes, generate a small batch with both Euler and Heun and verify output shapes and finite tensors.
4. For checkpoint changes, perform a save/load round-trip and verify model, EMA, optimizer, scheduler, and step restoration.

Do not treat imports or syntax checks as a substitute for a real training step.

## 11. Operational notes and known limitations

- Full multi-GPU training and the default million-step recipes have not been run in this open-source tree. Validation covered a real single-GPU two-step launcher run, small GPU forward/backward tests for both decoder variants, Euler/Heun sampling, and a BF16 optimizer/checkpoint round-trip.
- Model/checkpoint mismatches can produce missing or unexpected keys. Match model name, resolution, and patch size explicitly.
- Built-in FID is intended for multi-GPU training in the current implementation; use standalone FID for single-GPU workflows.
