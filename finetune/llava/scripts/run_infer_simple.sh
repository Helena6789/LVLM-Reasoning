
#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Please provide the ckpt_dir path and the test dataset name. e.g. ./run_infer.sh  ./output/checkpoint-500 smart101_finetune_inference_18_test_0"
    exit 1
fi

ckpt_dir=$1
val_dataset=$2
model_type="llava1_6-mistral-7b-instruct"

CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir ${ckpt_dir} \
    --model_type ${model_type} \
    --custom_dataset_info './dataset/finetune_dataset/smart101.json' \
    --val_dataset ${val_dataset} \