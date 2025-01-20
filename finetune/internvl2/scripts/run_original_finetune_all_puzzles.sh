#!/bin/bash

skip_puzzle_ids=(18 61 62 69 71 73 77 94 99)
model_type="internvl2-8b"

for puzzle_id in {1..101}
do
    # Check if the current puzzle is in the skip list
    if [[ " ${skip_puzzle_ids[@]} " =~ " $puzzle_id " ]]; then
        echo "skip puzzle: $puzzle_id"
        continue
    fi
    
    echo "original finetune for puzzle: ${puzzle_id}"

    date=$(date '+%Y%m%d-%H%M%S')
    CUDA_VISIBLE_DEVICES=0,1,2,3 swift sft \
        --model_id_or_path ${INTERNVL_MODEL_PATH} \
        --model_type ${model_type} \
        --custom_dataset_info './dataset/finetune_dataset/smart101.json' \
        --dataset smart101_finetune_inference_${puzzle_id}_train_0 \
        --val_dataset smart101_finetune_inference_${puzzle_id}_val_0  \
        --lora_rank '128' \
        --lora_alpha '256' \
        --num_train_epochs '5' \
        --learning_rate '1e-4' \
        --gradient_accumulation_steps '16' \
        --eval_steps '200' \
        --save_steps '200' \
        --eval_batch_size '1' \
        --max_length 4096 \
        --add_output_dir_suffix False \
        --output_dir ./output/${model_type}/${puzzle_id}/finetune-models/original-finetune/v0-${date}-question-split-0 \
        --logging_dir ./output/${model_type}/${puzzle_id}/finetune-models/original-finetune/v0-${date}-question-split-0/runs \
        --ignore_args_error True 

    sleep 30
done
