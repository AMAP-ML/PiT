#!/bin/bash
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=OFF
export TORCHDYNAMO_VERBOSE=1

NUM_GPUS=${NUM_GPUS:-8}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-42123}
torchrun_params="
    --nnodes 1 \
    --node-rank 0 \
    --nproc_per_node=$NUM_GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT"

datadir="${DATADIR:-/path/to/ImageNet256/train}"
fid_stat_path="${FID_STAT_PATH:-/path/to/ImageNet256/in256_fid_train.npz}"

expname=${EXPNAME:-""}
workdir=${WORKDIR:-"./$expname"}
model_name=${MODEL_NAME:-"PiT-XL"}
torchrun $torchrun_params \
    train.py \
    --workdir="$workdir" \
    --config=configs/IN256.py \
    --config.dataset.data_dir="$datadir" \
    --config.model_name="$model_name" \
    --config.dataset.fid_stat_path="$fid_stat_path" \
    "$@"