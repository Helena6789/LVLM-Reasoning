import json
import csv
from tqdm import tqdm
import os
import subprocess
from PIL import Image
from transformers import AutoTokenizer
import gc
import torch
from vllm import LLM,SamplingParams


#internvl2
model_path="./models/internvl2-8b"
llm = LLM(model=model_path,trust_remote_code=True, max_num_seqs=5,max_model_len=16384)
tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)


def get_msg(question):
    messages = [{'role': 'user', 'content': f"<image>\n{question}"}]
    return messages

def get_multi_msg(messages,generated_text,next_question):
    assistant_msg={"role":"assistant","content":generated_text}
    messages.append(assistant_msg)
    next_query={"role":"user","content":next_question}
    messages.append(next_query)
    return messages
    
def get_final_msg(messages,generated_text):
    # messages = [{'role': 'user', 'content': f"<image>\n{question}"}]
    assistant_msg={"role":"assistant","content":generated_text}
    messages.append(assistant_msg)
    final_query={"role":"user","content":"Please only answer from choices 'A','B','C','D','E' as final answer"}
    messages.append(final_query)
    return messages


def internvl_gen(messages,img):

    prompt = tokenizer.apply_chat_template(messages,
                                            tokenize=False,
                                            add_generation_prompt=True)
    image=Image.open(img)
    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
    sampling_params = SamplingParams(temperature=0.2,
                                        max_tokens=512,
                                        stop_token_ids=stop_token_ids)
    
    inputs = {
        "prompt": prompt,
        "multi_modal_data": {
                "image": image
        },
    }
    outputs = llm.generate(inputs, sampling_params=sampling_params)

    for o in outputs:
        generated_text = o.outputs[0].text
        # print(generated_text)
    return generated_text

base_path = "/home/yyf/data/llava_data/new_71_99/"

file_list={
    "18":"18/question_split/finetune-inference/puzzle_18_test_0.json",
}





for file in file_list:
      print(f"开始处理数据集{file}:")
      json_file_path = base_path+file_list[file]
      with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
      csv_file_path ="/home/yyf/data/llava_data/finetune_dataset/internvl2_sft_preidct_"+file+".csv"
      # csv_file_path ="/home/yyf/data/llava_data/new_71_99/internvl2_99_sft_preidct_"+file+".csv"
      with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:     
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['puzzle_id', 'image_id', 'prompt', 'true_answer', 'predit_answer'])
            # csvwriter.writerow(['puzzle_id','sub_question_id','image_id', 'prompt', 'true_answer', 'final_answer']) 
            for item in tqdm(data, desc="Processing", unit="item"):      
                  puzzle_id = item["id"]
                  image_id = item["images"][0]
                  true_answer = item["messages"][1]["content"]
                  prompt = item["messages"][0]["content"].replace("<image>","")
                  messages = get_msg(prompt)
                  predit_answer = internvl_gen(messages, image_id)
                  
                  # final_msg = get_final_msg(messages=messages, generated_text=predit_answer)
                  # final_answer = internvl_gen(final_msg, image_id)
                  csvwriter.writerow([puzzle_id,  image_id, prompt, true_answer, predit_answer])
                  gc.collect()
                  torch.cuda.empty_cache()
                 
                  


      print(f"数据集{file}处理完毕，结果已写入CSV文件。")



