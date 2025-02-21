# List of seeds and clip learning rates
SEEDS=(42 220 3407)
CLIP_LR=(3e-6)

NUM_TASKS=10
# Train the model for each seed and clip_lr
for SEED in "${SEEDS[@]}"; do
    for LR in "${CLIP_LR[@]}"; do
        # Define the dataset path with the current index
        DATA_PATH="data/MSRVTT_${NUM_TASKS}_dataset.pkl"
        # Define the experiment name
        EXP_NAME="MSRVTT${NUM_TASKS}_framefusionMoE_${SEED}_${LR}"
        # Train the model
        python "train_framefusionMoE.py" \
            --exp_name="$EXP_NAME" \
            --project_name=1-MSRVTT_${NUM_TASKS}_FrameFusionMoE \
            --dataset_name=MSRVTT \
            --path_data="$DATA_PATH" \
            --videos_dir=datasets/msrvtt_data/MSRVTT_Frames \
            --benchmark=cap \
            --arch=frame_fusion_moe \
            --loss=triplet \
            --loss_scale=0.6 \
            --lora_dropout=0.2 \
            --lora_num=10 \
            --topk=2 \
            --grad_acc_steps=1 \
            --num_frames=12 \
            --batch_size=8 \
            --clip_v_lr="$LR" \
            --clip_t_lr="$LR" \
            --noclip_lr=1e-5 \
            --transformer_dropout=0.3 \
            --adapter_applied_layer=10 \
            --warmup_proportion=0.1 \
            --weight_decay=0.4 \
            --max_num_epochs=10 \
            --evals_per_epoch=2 \
            --seed="$SEED" \
            --task_num="$NUM_TASKS" \
            --load_best \
            --init_validation \
            --huggingface
    done
done