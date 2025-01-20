#!/bin/bash

skip_puzzle_ids=(18 61 62 69 71 73 77 94 99)

clip_images_path=""

for puzzle_id in {1..101}
do
    # Check if the current puzzle is in the skip list
    if [[ " ${skip_puzzle_ids[@]} " =~ " $puzzle_id " ]]; then
        echo "skip puzzle: $puzzle_id"
        continue
    fi
    
    echo "using clip_images_path: ${clip_images_path} for puzzle: ${puzzle_id}"
    python baseline/llava/finetune_data_generator.py \
        --output-root ./dataset/finetune_dataset/ \
        --input-data ./dataset/decomposition_dataset/generate_data_no_sub_question_puzzle_${puzzle_id}.csv \
        --smart101-data-root dataset/SMART101-release-v1/SMART101-Data \
        --clip-image-root "${clip_images_path}" \
        --puzzle-id ${puzzle_id} \
        --output-data-types pretrain-inference,pretrain-cot,finetune-inference,finetune-cot \
        --seed 42
done
