#!/usr/bin/env python3

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch

class LlavaClient:
    def __init__(self, model_id, device):
        self.device = device
        # assert model_id == "llava-hf/llava-v1.6-mistral-7b-hf"
        self.model = LlavaNextForConditionalGeneration.from_pretrained(model_id, 
                                                                       torch_dtype=torch.float16, 
                                                                       low_cpu_mem_usage=True)
        self.processor = LlavaNextProcessor.from_pretrained(model_id)
        self.model.to(self.device)

    def generate(self, prompts, images, **kwargs):
        inputs = self.processor(prompts, images, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, **kwargs)
        generated_texts = self.processor.batch_decode(outputs, skip_special_tokens=True)
        answers = []
        for text in generated_texts:
            answers.append(text.split("[/INST] ")[-1])
        return answers

def load_preptrained_model(modle_id, device="cuda"):
    return LlavaClient(modle_id, device)