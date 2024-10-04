# LVLM-Reasoning

## Finetune

#### Install
We using [ms-swift](https://github.com/modelscope/ms-swift) to help us finetune llava. For more details see [llava-best-practice](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Multi-Modal/llava-best-practice.md).

```
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
pip install 'ms-swift[llm]' -U
```

### Dataset 

The puzzle finetune JSON data are available in this repo, but the images file need to download separately due to too large. 

#### Download the clip_images

Download the [clip_images.zip](https://drive.google.com/file/d/1jgVEYvZCdye04s7mvJWNF1wzJk2BMeUE/view?usp=drive_link) to `dataset/clip_images`

```
cd dataset/clip_images
unzip clip_images.zip
```

#### Download smart101 original images

Download the [smart101_original_images.zip](https://drive.google.com/file/d/16I1frv5A5ijClzBH1K_jQv45rHl0gf6L/view?usp=drive_link) to `dataset/SMART101-release-v1/SMART101-Data`

```
cd dataset/SMART101-release-v1/SMART101-Data
unzip smart101_original_images.zip
```

#### Define Custom Dataset

> We already auto generated the custom dataset file, you can skip this step.

In order to finetune llava using our own smart101 dataset, we can follow the [offical guide](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/LLM/Customization.md#custom-dataset) to define the ataset. 

We added a `dataset/finetune_dataset/smart101.json` file to define our own dataset like following:

```json
{
  "smart101_finetune_cot_18_test_0": {
    "dataset_path": "18/question_split/finetune-cot/puzzle_18_test_0.json",
    "tags": [
      "smart101"
    ]
  },
  "smart101_implicit_cot_18_train_0": {
    "dataset_path": "18/question_split/implicit-cot/puzzle_18_train_0.json",
    "tags": [
      "smart101"
    ]
  }
}
```

### Finetune llava

To finetune llava run the following command (make sure your run the script in the LVLM-Reasoning root path.):

#### Original simple funetune

Using original question and answer to finetune

```
sh ./finetune/llava/scripts/run_original_finetune.sh <puzzle_id>
```

#### Stage finetune using `lora` sft-type

Using decomposed sub-questions and sub-images stage by stage finetune

```
sh ./finetune/llava/scripts/run_stage_finetune_lora.sh <puzzle_id> <start_stage> <max_stage> <shift_step>
```

#### Stage finetune using `full` sft-type

Using decomposed sub-questions and sub-images stage by stage finetune

```
sh ./finetune/llava/scripts/run_stage_finetune_full.sh <puzzle_id> <start_stage> <max_stage> <shift_step>
```

#### Stage finetune using `full` sft-type with `deepspeed`

Using decomposed sub-questions and sub-images stage by stage finetune

```
sh ./finetune/llava/scripts/run_stage_finetune_deepspeed.sh <puzzle_id> <start_stage> <max_stage> <shift_step>
```
See details in [modelscope LLM finetune guide](https://swift.readthedocs.io/en/latest/Instruction/LLM-fine-tuning.html)

Please change the parameters accordingly based on your input and output path. We can also change the `MODELSCOPE_CACHE` to reset the enviroment variable to `MODELSCOPE_CACHE=<you_cache_path>`.

### Inference

To inference the fine-tuned model for a particular checkpoint run the following command:

```
sh ./finetune/llava/scripts/run_infer_simple.sh <your_ckpt_path> <your_test_data>
```
It will pass the sub questions and its answers to the model, to get the final question answer. The result will be save on the <your_ckpt_path> and named as `infer_result`.

To run all the five types inference `pretrain-inference,pretrain-cot,finetune-inference,finetune-cot,implicit-cot` with the following command:

```
sh ./finetune/llava/scripts/run_infer_all.sh <puzzle_id>
```

For other options, please see [the guide](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/LLM/LLM-fine-tuning.md#inference) how to inference using fine-tuned model. 

### Plots

To generate the accurary plots for different types:


```
sh ./finetune/llava/scripts/run_plot_infer_results.sh <puzzle_id>
```
Please change the parameters accordingly if only intrested in part of the inference type.