## Evaluation

To evaluate the model with other datasets, such as MathVision, you can utilize SWIFT. SWIFT supports evaluation capabilities to provide standardized assessment metrics. 
For more details, refer to the [Evaluation Guide](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/Evaluation.md).

## Environment Preparation

### 1. Install `ms-swift` Evaluation Tools

Clone the repository and install the evaluation tools:

```bash

git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
pip install -e '.[eval]'

```

Alternatively, if you do not need to override the local code, you can install directly via pip:

```bash

pip install ms-swift[eval] -U

```

### 2. Set Up OpenAI Keys

`ms-swift` uses [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) for multimodal evaluation. Evaluating MathVision requires a working OpenAI API. 
You can configure it using one of the following methods:

#### Using Azure OpenAI

If you are using Azure OpenAI endpoints, ensure the endpoint follows the pattern:

```
{endpoint}openai/deployments/{deployment_name}/chat/completions?api-version={api_version}

```

Set the following environment variables:

```bash
export AZURE_OPENAI_ENDPOINT=YOUR_AZURE_OPENAI_ENDPOINT
export AZURE_OPENAI_DEPLOYMENT_NAME=YOUR_AZURE_OPENAI_DEPLOYMENT_NAME
export OPENAI_API_VERSION=YOUR_OPENAI_API_VERSION
export AZURE_OPENAI_API_KEY=YOUR_AZURE_OPENAI_API_KEY

```

#### Using Default OpenAI Settings

To use OpenAI's default URL and keys, set the following:


```bash
export OPENAI_API_BASE=YOUR_OPENAI_API_BASE
export OPENAI_API_KEY=YOUR_OPENAI_API_KEY
```
You can verify the settings and debug the OpenAI API with the following Python script:

```python

from vlmeval.api import OpenAIWrapper
model = OpenAIWrapper('gpt-4o', verbose=True)
msgs = [dict(type='text', value='Hello!')]
code, answer, resp = model.generate_inner(msgs)
print(code, answer, resp)
```

If the API call fails, the script will display the specific error.

### Running the Evaluation

To run the evaluation for the `InternVL2-8B` model with MathVision:

```bash

sh eval/internvl2/scripts/run_eval.sh MathVision

```

Once the evaluation process completes, the results will be saved in the `result/InternVL2-8B/eval_result` directory. 
Note that this process may take several days depending on your environment and computational resources.

## Custom Evaluation

You can also perform a custom evaluation based on inference rather than using the existing tools. Follow these steps:

1. Convert the dataset to the conversation model format supported by the ms-swift tool. See details in the [Custom Dataset Guide](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Customization/Custom-dataset.md). For example, run the following command to generate the MathVision test dataset.

```bash
sh baseline/llava/scripts/run_other_data_generator.sh  MathLLMs/MathVision test
```

2. Run inference to generate model answers from the dataset, similarily as we done for `smart-101`.

```bash

sh eval/internvl2/scripts/run_infer.sh dataset/eval_dataset/MathVision/test/MathVision.json

```

3. Evaluate the accuracy based on prior work.

The model answers will be saved in `result/InternVL2-8B/infer_result` for the `InternVL2-8B` model. 
To evaluate accuracy, you can either:

- Force the model to provide answers in a specific pattern.
- Use similarity tools to compare the model answers with the ground truth.

