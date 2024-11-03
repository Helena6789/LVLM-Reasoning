# Experimental environment: A10, 3090, V100, ...
# 20GB GPU memory
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,6,7'
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4,6,7'

import torch
import os, sys
# sys.path.append("/scratch/czr/LVLM-Reasoning/finetune/swift2")
from swift.llm import (
    DatasetName, InferArguments, ModelType, SftArguments,
    infer_main, sft_main, app_ui_main, merge_lora
)
# print("ModelType", ModelType)

# model_type = ModelType.internvl2_8b
model_type = ModelType.internvl2_8b_att
# model_type = ModelType.internvl2_8b_vgp_v2
# model_type = ModelType.internvl2_8b_vgp_v2_dpo_121



sft_args = SftArguments(
    model_type=model_type,
    dataset=["/scratch/czr/LVLM-Reasoning/finetune/test_data_1.json"],
    output_dir='/scratch/czr/LVLM-Reasoning/finetune/test_output',
    sft_type='lora', #'lora',
    batch_size=1,
    # per_device_train_batch_size=2,
    # gradient_accumulation_steps=8,
    gradient_checkpointing=False,
    save_steps=50,
    eval_steps=10000,
    acc_steps=20,
    # report_to=['wandb'],
    seed=22,
    # lora_rank=4,
    # lora_alpha=8,
    optim="adamw_torch_fused",
    save_total_limit=100,
    # resume_from_checkpoint="/scratch/czr/Video_moderation/sft_checkpoints/internvl2_swift_8b_10_20/internvl2-8b-vg/v0-20241020-112140/checkpoint-1500",
    # resume_from_checkpoint="/scratch/czr/Video_moderation/sft_checkpoints/internvl2_swift_8b_09_18/internvl2-8b-vgp-v2-dpo-121/v1-20240918-164542/checkpoint-201",
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


