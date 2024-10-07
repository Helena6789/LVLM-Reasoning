#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Please provide the puzzle id and implicit-cot shift steps"
    exit 1
fi

puzzle_id=$1
step=$2
model_type="internvl2-8b"

python ./finetune/plot_infer_results.py \
 --model-type ${model_type} \
 --infer-results-path ./output/${model_type}/${puzzle_id}/custom_infer_results \
 --infer-type pretrain-inference,pretrain-cot,finetune-inference,finetune-cot,implicit-cot \
 --step ${step} \
 --puzzle-id ${puzzle_id}