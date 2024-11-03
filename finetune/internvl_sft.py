# Experimental environment: A10, 3090, V100, ...
# 20GB GPU memory
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,6,7'
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4,6,7'

import torch
import os, sys
from swift.llm import (
    DatasetName, InferArguments, ModelType, SftArguments,
    infer_main, sft_main, app_ui_main, merge_lora
)
# print("ModelType", ModelType)

# model_type = ModelType.internvl2_8b
model_type = ModelType.internvl2_8b_att




sft_args = SftArguments(
    model_type=model_type,
    dataset=["dataset/finetune_dataset_subimage/18/question_split/pretrain-inference/puzzle_18_test_0.json"],
    output_dir='finetune/test_output',
    sft_type='lora', #'lora',
    batch_size=4,
    # per_device_train_batch_size=2,
    # gradient_accumulation_steps=8,
    gradient_checkpointing=False,
    save_steps=100,
    eval_steps=10000,
    acc_steps=20,
    report_to=['wandb'],
    seed=22,
    # lora_rank=4,
    # lora_alpha=8,
    optim="adamw_torch_fused",
    save_total_limit=100,
    num_train_epochs=1)

result = sft_main(sft_args)
best_model_checkpoint = result['best_model_checkpoint']

print(f'best_model_checkpoint: {best_model_checkpoint}')
torch.cuda.empty_cache()

# infer_args = InferArguments(
#     ckpt_dir=best_model_checkpoint,
#     load_args_from_ckpt_dir=True)
#     # load_dataset_config=True)
# merge_lora(infer_args, device_map='cpu')
# result = infer_main(infer_args)
# torch.cuda.empty_cache()

# app_ui_main(infer_args)


