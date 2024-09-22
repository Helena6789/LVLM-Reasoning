#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Please provide the puzzle id or all and skip step"
    exit 1
fi

puzzle_id=$1
step=$2

python ../finetune_data_generator.py \
--output-root /home/hq/LVLM/LVLM-Reasoning/dataset/finetune_dataset/ \
--input-data /home/hq/LVLM/LVLM-Reasoning/dataset/decomposition_dataset/generate_data_by_question_decomposition_clip_images_puzzle_${puzzle_id}.csv \
--smart101-data-root dataset/SMART101-release-v1/SMART101-Data \
--clip-image-root '' \
--puzzle-id ${puzzle_id} \
--skip-stage-step ${step} \
--seed 42
