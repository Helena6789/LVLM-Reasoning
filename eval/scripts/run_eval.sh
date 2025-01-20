#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Please provide the test dataset name. e.g. ./run_eval.sh MathVision"
    exit 1
fi

eval_dataset=$1
model_type="internvl2"

CUDA_VISIBLE_DEVICES=0,1,2,3 swift eval \
    --model ${INTERNVL_MODEL_PATH} \
    --model_type ${model_type} \
    --eval_dataset ${eval_dataset}