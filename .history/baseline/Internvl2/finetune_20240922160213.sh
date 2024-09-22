#!/bin/bash

# 安装 ms-swift
pip install ms-swift

# Swift 微调命令
swift sft --model_type internvl2-8b \
    --model_id_or_path ./models/internvl2-8b \
    --template_type internvl2 \
    --sft_type lora \
    --tuner_backend peft \
    --dtype AUTO \
    --output_dir output \
    --dataset ./data/new_71_99/puzzle_99_train.json \
    --val_dataset ./data/new_71_99/puzzle_99_val.json \
    --num_train_epochs 5 \
    --max_length 4096 \
    --check_dataset_strategy warning \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout_p 0.05 \
    --gradient_checkpointing true \
    --batch_size 1 \
    --weight_decay 0.1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 16 \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.03 \
    --eval_steps 100 \
    --save_steps 500 \
    --logging_steps 10 \
    --use_flash_attn true

# Swift 合并模型命令
swift export \
    --model_id_or_path ./models/internvl2-8b \
    --ckpt_dir ./swift/output/internvl2-8b/v27-20240919-203320/checkpoint-500 \
    --merge_lora true
