## Supporting Subimage Ranking Loss 

### How to run

1. Install the swift in the local directory `finetune/swift`, to do this, run the following command in the `finetune/swift` directory (make sure this step is done as we have rewritten multiple files in swift to support passing subimage as input):
```
cd finetune/swift
pip install -e .
```
2. replace the `modeling_internlm2.py` in the default InternVL2 model directory with `finetune/modeling_internlm2.py` (made important changes on the model attention and loss calculation to support subimage ranking loss).

You will also need to manually register this model so that swift can find it. We have already added this model `internvl2_8b_att` to `finetune/swift/swift/llm/utils/model.py`. You should manually update these lines (line 4537) and put the path of the new model directory in the second argument (I would strongly suggest you copy the original internvl2-8b model to a new directory and make changes in the new directory).
```
@register_model(
    ModelType.internvl2_8b_att,
    '/scratch/czr/hf_models/InternVL2-8B-lvlm',
    LoRATM.internvl,
    TemplateType.internvl2,
    requires=['transformers>=4.36', 'timm'],
    ignore_file_pattern=[r'.+\.zip$'],
    support_flash_attn=True,
    support_lmdeploy=True,
    support_vllm=True,
    placeholder_tokens=['<IMG_CONTEXT>'],
    tags=['multi-modal', 'vision', 'video'],
    hf_model_id='OpenGVLab/InternVL2-8B-att')
```

3. run this code to update the dataset file: (please make sure to check the code to know what it does). Basically this code runs BLIP in a hierachical way to obtain a list of scores for each subimage patch of each image given the subquestion and subsolution in each entry of the dataset file. (Also make sure your dataset file does NOT contain those json files already with subimages as input because this intuitively CONTRADICTS with what this code does)
```
python dataset/transform_subimage.py
```
4. update the training script to use the new dataset file (also make sure you are using the local swift installed in Step 1 and using the internvl2 model directory in Step 2). See an example script in `finetune/internvl_sft.py`. But you should definitely be able to use the original script under `finetune/internvl2/scripts`.


After these steps you should be good to go:))))

### What You Should Do

You should tune the weight hyperparameter which balance the original autoregressive loss and the ranking loss in `modeling_internvl_chat.py` (line 1210) to achieve a good performance.
```
### Aggregated loss
lambda_r = 0.1
loss = loss + lambda_r * ranking_loss
```


The following changes are made to support the ranking loss with subimage:

1. Made several changes to `finetune/swift/swift/trainers/trainers.py`, `finetune/swift/swift/llm/utils/preprocess.py`, `finetune/swift/swift/llm/utils/dataset.py`, `finetune/swift/swift/llm/utils/template.py`.
2. Add a new loss function `compute_ranking_loss` in `modeling_internvl_chat.py`
