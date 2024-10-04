#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "Please provide the puzzle_id, start_stage, max_stage, step: e.g: ./run_stage_finetune_full.sh 18 1 9 1"
    exit 1
fi

puzzle_id=$1
start_stage=$2
max_stage=$3
step=$4
date=$(date '+%Y%m%d-%H%M%S')
model_type="llava1_6-mistral-7b-instruct"

if [[ $start_stage -lt 0 ]] || [[ $start_stage -gt $max_stage ]]; then
    echo "start stage value should great than 0 less than max_stage"
    exit
fi

for ((stage = start_stage; stage <= max_stage; stage += step))
do
    if [[ $stage -eq 0 ]]; then
        echo "finetune stage:${stage}"
        CUDA_VISIBLE_DEVICES=0,1,2,3 swift sft \
            --model_type ${model_type} \
            --custom_dataset_info './dataset/finetune_dataset/smart101.json' \
            --dataset smart101_implicit_cot_${puzzle_id}_train_${stage} \
            --dataset_test_ratio '0.059' \
            --sft_type full \
            --num_train_epochs '5' \
            --gradient_accumulation_steps '16' \
            --eval_steps '200' \
            --save_steps '200' \
            --eval_batch_size '1' \
            --add_output_dir_suffix False \
            --output_dir ./output/${model_type}/${puzzle_id}/finetune-models/decomposition-finetune-shift-${step}/v0-${date}-question-split-0 \
            --logging_dir ./output/${model_type}/${puzzle_id}/finetune-models/decomposition-finetune-shift-${step}/v0-${date}-question-split-0/runs \
            --ignore_args_error True
    else
        previous_stage=$((stage - step))
        model_checkpoint=`find . -type d -wholename "./output/${model_type}/${puzzle_id}/finetune-models/decomposition-finetune-shift-${step}/v0-*-question-split-${previous_stage}/checkpoint-*" -prune | sort | tail -n 1`

        echo "finetune stage:${stage} using checkpoint: ${model_checkpoint}"

        # see command meaning: https://github.com/modelscope/ms-swift/blob/8c41771e9ffd6c90de20c980e6116c74f9d8b8fe/docs/source_en/Instruction/Command-line-parameters.md
        CUDA_VISIBLE_DEVICES=0,1,2,3 swift sft \
            --model_type ${model_type} \
            --resume_from_checkpoint ${model_checkpoint} \
            --resume_only_model True \
            --custom_dataset_info './dataset/finetune_dataset/smart101.json' \
            --dataset smart101_implicit_cot_${puzzle_id}_train_${stage} \
            --dataset_test_ratio '0.059' \
            --sft_type full \
            --num_train_epochs '5' \
            --gradient_accumulation_steps '16' \
            --eval_steps '200' \
            --save_steps '200' \
            --eval_batch_size '1' \
            --add_output_dir_suffix False \
            --output_dir ./output/${model_type}/${puzzle_id}/finetune-models/decomposition-finetune-shift-${step}/v0-${date}-question-split-${stage} \
            --logging_dir ./output/${model_type}/${puzzle_id}/finetune-models/decomposition-finetune-shift-${step}/v0-${date}-question-split-${stage}/runs \
            --ignore_args_error True
    fi
    sleep 30
done