#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Please provide the puzzle id or all, and cuda visibe devices"
    exit 1
fi

puzzle_id=$1
model_type="llava1_6-mistral-7b-instruct"

CUDA_VISIBLE_DEVICES=$2 python ./finetune/llava/infer_finetune.py \
 --model-type ${model_type} \
 --output-root ./output/${model_type}/${puzzle_id} \
 --ckpt-dir-root ./output/${model_type}/${puzzle_id}/finetune-models \
 --val-dataset-root  ./dataset/finetune_dataset/${puzzle_id}/question_split \
 --infer-types pretrain-inference,pretrain-cot,finetune-inference,finetune-cot,implicit-cot \
 --step 1 \
 --puzzle-id ${puzzle_id}