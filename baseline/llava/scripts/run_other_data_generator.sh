#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Please provide dataset name. e.g. MathLLMs/MathVision testmini"
    exit 1
fi

dataset_name=$1
dataset_split=$2

/scratch/xey6dx/.conda/envs/internv-fn-val/bin/python ./baseline/llava/other_dataset_generator.py \
--output-root ./dataset/eval_dataset/ \
--input-data ${dataset_name} \
--dataset-split ${dataset_split}
