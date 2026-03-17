#!/bin/bash

SEEDS=(1234 2024 3407 6666 8888)

for SEED in "${SEEDS[@]}"
do
  torchrun --nproc_per_node=3 train.py \
    --dataset TN3K \
    --root_path ./data/TN3K \
    --cfg ./configs/swin_base_patch4_window12_384_22k.yaml \
    --output_dir ./results \
    --exp_name proposed \
    --img_size 384 \
    --max_epochs 100 \
    --batch_size 4 \
    --seed ${SEED} \
    --eval_interval 1 \
    --accumulation_steps 1 \
    --num_workers 8
done