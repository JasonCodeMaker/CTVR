#!/bin/bash

# MSRVTT_10 Single Task
python "main.py" \
    --eval \
    --exp_name="Eval_MSRVTT10_Single" \
    --config="config/frame_fusion_moe_config.yaml" \
    --dataset_name="MSRVTT" \
    --path_data="data/MSRVTT_10_dataset.pkl" \
    --videos_dir="datasets/MSRVTT/MSRVTT_Frames" \
    --arch="frame_fusion_moe" \
    --seed=42 \
    --eval_task_id=10 \
    --eval_mode="single" \
    --eval_path="checkpoints/MSRVTT10_framefusion2MoE_42_3e-6"

# MSRVTT_20 Single Task
python "main.py" \
    --eval \
    --exp_name="Eval_MSRVTT20_Single" \
    --config="config/frame_fusion_moe_config.yaml" \
    --dataset_name="MSRVTT" \
    --path_data="data/MSRVTT_20_dataset.pkl" \
    --videos_dir="datasets/MSRVTT/MSRVTT_Frames" \
    --arch="frame_fusion_moe" \
    --seed=42 \
    --eval_task_id=20 \
    --eval_mode="single" \
    --eval_path="checkpoints/MSRVTT20_framefusion2MoE_42_4e-6"

# ACTNET_10 Single Task
python "main.py" \
    --eval \
    --exp_name="Eval_ACTNET10_Single" \
    --config="config/default_config.yaml" \
    --dataset_name="ACTNET" \
    --path_data="data/ACTNET_10_dataset.pkl" \
    --videos_dir="datasets/ACTNET/Activity_Clip_Frames" \
    --arch="frame_fusion_moe" \
    --seed=42 \
    --eval_task_id=10 \
    --eval_mode="single" \
    --eval_path="checkpoints/ACTNET10_framefusionMoE_220_6e-6"

# ACTNET_20 Single Task
python "main.py" \
    --eval \
    --exp_name="Eval_ACTNET20_Single" \
    --config="config/default_config.yaml" \
    --dataset_name="ACTNET" \
    --path_data="data/ACTNET_20_dataset.pkl" \
    --videos_dir="datasets/ACTNET/Activity_Clip_Frames" \
    --arch="frame_fusion_moe" \
    --seed=42 \
    --eval_task_id=20 \
    --eval_mode="single" \
    --eval_path="checkpoints/ACTNET20_framefusionMoE_42_6e-6"
