#!/usr/bin/env python3

import constants

from utils import set_seed
import argparse
import pandas as pd
from datasets import load_dataset
import os
import json
import numpy as np
import time
import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)

QUESTION_WITH_OPTIONS = """
Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.
Question: {}\nChoices:\n(A) {}\n(B) {}\n(C) {}\n(D) {}\n(E) {}
"""

QUESTION_WITHOUT_OPTIONS = """
Hint: Please answer the question and provide the final answer at the end.\nQuestion: {}"
"""

def generate_question(question, options):
    if options and len(options) == 5:
        return QUESTION_WITH_OPTIONS.format(question, options[0], options[1], options[2], options[3], options[4])
    return QUESTION_WITHOUT_OPTIONS.format(question)

def generate_eval_dataset_json_file(data_output_root, data_type_name, dataset_split, output_json_filename):
    output_filename = os.path.join(data_output_root, "eval_dataset.json")
    json_data = {}
    if os.path.exists(output_filename):
        with open(output_filename, 'r') as file:
            json_data = json.load(file)
    
    json_data.update({
        '{}_{}'.format(data_type_name, dataset_split): {
            'dataset_path': output_json_filename.replace(data_output_root, ''),
            'tags': [data_type_name]
        }
    })
    
    with open(output_filename, 'w') as file:
        json.dump(json_data, file, indent=2)

def main(args):
    # 0. set seed
    set_seed(args.seed)

    # split data
    dataset = load_dataset(args.input_data)
    output_json_list = []
    
    data_type_name = args.input_data.split('/')[-1]
    dataset_output_path = os.path.join(args.output_root, data_type_name, args.dataset_split)
    if not os.path.exists(dataset_output_path):
        os.makedirs(dataset_output_path)
    
    for row in dataset[args.dataset_split]:
        # save images
        image_data = row['decoded_image'] 
        images_path = os.path.join(dataset_output_path, os.path.dirname(row['image']))
        if not os.path.exists(images_path):
            os.makedirs(images_path)
        image_path_with_name = os.path.join(images_path, os.path.basename(row['image']))
        image_data.save(image_path_with_name)
        
        conversation = {
            "messages": [
                {"role": "user", "content": f"<image> {generate_question(row['question'], row['options'])}"},
                {"role": "assistant", "content": row["answer"]}
            ],
            "level": row['level'],
            "subject": row['subject'],
            "images": [image_path_with_name]  # Use the existing Base64-encoded image
        }
        output_json_list.append(conversation)
    
    if len(output_json_list) == 0:
        return

    
    output_filename = os.path.join(dataset_output_path, "{}.json".format(data_type_name))    
    logger.info("write data file {} with {} records.".format(output_filename, len(output_json_list)))
    with open(output_filename, 'w') as f:
        json.dump(output_json_list, f, indent=2)  
    
    # generate eval dataset
    generate_eval_dataset_json_file(args.output_root, data_type_name, args.dataset_split, output_filename)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A tool for generate finetune dataset.")
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="The output folder to save all the results.",
    )
    parser.add_argument(
        "--input-data",
        type=str,
        required=True,
        help="The input data name from huggingface.",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        required=True,
        help="The input data split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed to use"
    )
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')

    args = parser.parse_args()

    output_root = args.output_root
    if not os.path.exists(output_root):
      os.makedirs(output_root)

    # setup logging
    logging_level = logging.INFO
    if args.debug:
      logging_level = logging.DEBUG

    logging.basicConfig(
      level=logging_level,
      format="%(asctime)s [%(levelname)s] %(message)s",
      handlers=[
          RotatingFileHandler(os.path.join(args.output_root,"other_dataset_generator-{}.log".format(time.strftime("%Y-%m-%d-%H-%M-%S"))),
                                      maxBytes=1024 * 1024 * 10),
          logging.StreamHandler()
      ]
    )
    logger.info(args)

    main(args)