#!/bin/bash

python "train_framefusionMoE.py" \
    --exp_name="Demo" \
    --config="configs/msrvtt_framefusion_moe.yaml" \
    --dataset_name="MSRVTT" \
    --path_data="data/MSRVTT_10_dataset.pkl" \
    --videos_dir="datasets/msrvtt_data/MSRVTT_Frames" \
    --arch="frame_fusion_moe" \
    --seed=42 \
    --task_num=10 \