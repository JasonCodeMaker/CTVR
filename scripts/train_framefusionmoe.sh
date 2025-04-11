#!/bin/bash

# Train FrameFusionMoE on MSRVTT with 10 tasks
python "train_framefusionMoE.py" \
    --exp_name="msrvtt_10" \
    --config="configs/msrvtt_framefusion_moe.yaml" \
    --dataset_name="MSRVTT" \
    --path_data="data/MSRVTT_10_dataset.pkl" \
    --videos_dir="datasets/msrvtt_data/MSRVTT_Frames" \
    --arch="frame_fusion_moe" \
    --seed=42 \
    --task_num=10

# Train FrameFusionMoE on MSRVTT with 20 tasks
python "train_framefusionMoE.py" \
    --exp_name="msrvtt_20" \
    --config="configs/msrvtt_framefusion_moe.yaml" \
    --dataset_name="MSRVTT" \
    --path_data="data/MSRVTT_20_dataset.pkl" \
    --videos_dir="datasets/msrvtt_data/MSRVTT_Frames" \
    --arch="frame_fusion_moe" \
    --seed=42 \
    --task_num=20

# Train FrameFusionMoE on ACTNET with 10 tasks
python "train_framefusionMoE.py" \
    --exp_name="actnet_10" \
    --config="configs/actnet_framefusion_moe.yaml" \
    --dataset_name="ACTNET" \
    --path_data="data/ACTNET_10_dataset.pkl" \
    --videos_dir="datasets/ACTNET/Activity_Clip_Frames" \
    --arch="frame_fusion_moe" \
    --seed=42 \
    --task_num=10

# Train FrameFusionMoE on ACTNET with 20 tasks
python "train_framefusionMoE.py" \
    --exp_name="actnet_20" \
    --config="configs/actnet_framefusion_moe.yaml" \
    --dataset_name="ACTNET" \
    --path_data="data/ACTNET_20_dataset.pkl" \
    --videos_dir="datasets/ACTNET/Activity_Clip_Frames" \
    --arch="frame_fusion_moe" \
    --seed=42 \
    --task_num=20
