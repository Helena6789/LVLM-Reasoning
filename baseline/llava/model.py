#!/usr/bin/env python3

import base64
import constants
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, LlavaProcessor, LlavaForConditionalGeneration
import torch
from openai import OpenAI
import os
import logging
import time

logger = logging.getLogger(__name__)

def load_preptrained_model(modle_id, device="cuda"):
    return LlavaClient(modle_id, device)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
      return base64.b64encode(image_file.read()).decode('utf-8')

class LlavaClient:
    def __init__(self, model_id, device):
        self.device = device
        self.model_id = model_id

        if 'llava-v1.6-mistral-7b' in model_id:
          self.model = LlavaNextForConditionalGeneration.from_pretrained(model_id,
                                                                        torch_dtype=torch.float16,
                                                                        low_cpu_mem_usage=True)
          self.processor = LlavaNextProcessor.from_pretrained(model_id)
        elif 'llava-1.5-7b' in model_id:
          self.model = LlavaForConditionalGeneration.from_pretrained(model_id,
                                                                     torch_dtype=torch.float16,
                                                                     low_cpu_mem_usage=True)
          self.processor = LlavaProcessor.from_pretrained(model_id)
        else:
          raise "unknown model:{}".format(model_id)
        self.model.to(self.device)

    def generate(self, prompts, images, **kwargs):
        inputs = self.processor(prompts, images, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, **kwargs)
        generated_texts = self.processor.batch_decode(outputs, skip_special_tokens=True)
        answers = []
        for text in generated_texts:
          if 'llava-v1.6-mistral-7b' in self.model_id:
            answers.append(text.split("[/INST] ")[-1])
          elif 'llava-1.5-7b' in self.model_id:
            answers.append(text.split("ASSISTANT:")[-1])
        return answers

class SubQuestionGenerator:
    def __init__(self, model="gpt-4o"):
      if model == "gpt-4o":
        self.client = OpenAI(api_key=os.getenv('GPT_API_KEY'),
                             base_url=os.getenv('GPT_BASE_URL'))
      else:
        raise Exception("Unknown model: {}".format(model))

    def generate_output(self, image_dir, origin_question, generate_type = 'sub_question', previous_qa = "", max_depth=10):

      image_data = encode_image(image_dir)
      if generate_type == 'sub_question':
        prompt = constants.GENERATE_SUB_QUESTION_PROMTP.format(origin_question, constants.SAMPLE_QUESTION, constants.SAMPLE_OUTPUT)
      elif generate_type == 'caption':
        prompt = constants.GENERATE_CAPTION_PROMTP.format(origin_question, constants.SAMPLE_QUESTION, constants.SAMPLE_CAPTION_OUTPUT)
      elif generate_type == 'tot_sub_question':
        prompt = constants.TOT_GENERATE_SUB_QUESTION_PROMTP.format(origin_question, constants.TOT_SIMPLE_QUESTION, constants.TOT_SIMPLE_OUTPUT)
      elif generate_type == 'tot_follow_up_question':
        prompt = constants.TOT_GENERATE_FOLLOW_UP_QUESTION_PROMTP.format(previous_qa, origin_question, constants.TOT_SIMPLE_QUESTION,
                                                                         constants.TOT_FOLLOW_UP_OUTPUT, max_depth, constants.TOT_FINAL_OUTPUT)
      else:
        raise Exception("Unknown generate type:{}".format(generate_type))

      logger.debug(prompt)

      failure_count = 0
      while True:
        try:
          response = self.client.chat.completions.create(
              model="gpt-4o",
              messages=[
                  {
                      "role": "user",
                      "content": [
                          {
                              "type": "text",
                              "text": f"{prompt}"
                          },
                          {
                              "type": "image_url",
                              "image_url": {
                                  "url": f"data:image/jpeg;base64,{image_data}"
                              },
                          }
                      ],
                  }
              ],
              max_tokens=2048,
              temperature=1
          )

          gpt4_output = response.choices[0].message.content
          return gpt4_output
        except Exception as e:
          if 'out of range' in str(e):
            logger.info(gpt4_output)
          elif 'exceeded token rate limit' in str(e):
            logger.info("Reach rate limit, sleep 2s...")
            time.sleep(2)
          elif 'The response was filtered due to' in str(e):
            logger.error("Can't get response based on prompt:{}".format(prompt))
            return "No Data"
          elif 'Not allowed' in str(e):
            logger.error("API Not allowed to call, sleep...")
            time.sleep(3600)
          elif '404' in str(e):
            failure_count += 1
            logger.error("404 error, sleep {} seconds and gradually increase...".format(2^failure_count))
            time.sleep(2^failure_count)
          else:
            logger.error(e)
          continue