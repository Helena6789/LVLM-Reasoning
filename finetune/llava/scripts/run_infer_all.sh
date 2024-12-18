#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Please provide the puzzle id and implicit-cot shift steps."
    exit 1
fi

puzzle_id=$1
step=$2
model_type="llava1_6-mistral-7b-instruct"

CUDA_VISIBLE_DEVICES=0 python ./finetune/infer_finetune.py \
 --model-type ${model_type} \
 --output-root ./output/${model_type}/${puzzle_id} \
 --ckpt-dir-root ./output/${model_type}/${puzzle_id}/finetune-models \
 --val-dataset-root  ./dataset/finetune_dataset/${puzzle_id}/question_split \
 --infer-types pretrain-inference,pretrain-cot,finetune-inference,finetune-cot,implicit-cot \
 --step ${step} \
 --puzzle-id ${puzzle_id}