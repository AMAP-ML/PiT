#!/bin/bash

export TORCH_DISTRIBUTED_DEBUG=OFF
export NCCL_DEBUG=OFF

NUM_GPUS=${NUM_GPUS:-8}
MASTER_ADDR="localhost"
MASTER_PORT=12345
torchrun_params="
    --nproc_per_node=$NUM_GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT
"

datadir="Path to ImageNet /train directory"
fid_stat_path="Path to fid statistics .npz file"
# To report FID results, you should evaluate with ADM statistics 
# following the instructions in README.md


ckpt_path=( "/path/to/saved/.ckpt/directory")
scale_lst=(2.5)
for ckpt in ${ckpt_path[@]};do
    for scale in ${scale_lst[@]};do
        torchrun $torchrun_params \
            sample.py --config=configs/IN256.py \
            --config.path=$ckpt \
            --config.sample.sampling_step=50 \
            --config.sample.sampler="heun" \
            --config.sample.num_samples=50000 \
            --config.sample.mini_batch_size=50 \
            --config.dataset.fid_stat_path=$fid_stat_path \
            --config.dataset.data_dir=$datadir \
            --config.model_name="PiT-XL" \
            --config.sample.cfg_scale=$scale \
            --config.sample.path="" # if left empty, save to temp dir
    done
done