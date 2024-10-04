#!/usr/bin/env python3

import os
import argparse
import json
import datetime as dt
import pandas as pd
import time
import matplotlib.pyplot as plt
import textwrap
import torch
import re
import logging
import glob
from logging.handlers import RotatingFileHandler

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type, get_dataset
)
from swift.tuners import Swift

logger = logging.getLogger(__name__)


def generate_infer_results(model, val_dataset, template, infer_type, output_json_file, puzzle_id):
    with open(output_json_file, 'w') as output_file:
        count = 0
        for data in val_dataset:
            request = data['query']
            true_sub_qa = data.get('history')
            label = data.get('response')
            images = data.get('images')
            sub_puzzle_id = data.get('id')

            query_history = None
            history_label = []
            query = request
            logging.debug(query)
            query_response, _ = inference(model, template, query, images=images)
            
            reasoning_steps = ''
            query_response = query_response.strip()
            logging.debug("query_response:{}".format(query_response))
            if infer_type in ['pretrain-cot', 'finetune-cot', 'implicit-cot'] and len(query_response)!=1:
                reasoning_steps = query_response.replace('```json', '').replace('```','').replace('\n','')

                pattern = r'\\?\"?final\s?_?answer\\?\"?:\s*\\?\"?([A-Za-z])\\?\"?'
                match = re.search(pattern, query_response)
                if match:
                    query_response = match.group(1)
                else:
                    query_response = 'NOT_FOUND'
            output_records = {
                'response' : query_response.strip(),
                'label' : label,
                'history' : query_history,
                'history_label': history_label,
                'reasoning_steps': reasoning_steps,
                'id': puzzle_id,
                'sub_puzzle_id': sub_puzzle_id
            }
            count += 1
            logging.info(f'Completed: {count}/{len(val_dataset)} records.')
            output_file.write( json.dumps(output_records) + '\n')
            if count % 2 == 0:
               output_file.flush() 


def custom_inference(args, infer_type, ckpt_dir, file_suffix):
    model_type = args.model_type
    template_type = get_default_template_type(model_type)

    model, tokenizer = get_model_tokenizer(model_type, None, model_kwargs={'device_map': 'auto'})
    model.generation_config.max_new_tokens = 2048

    if ckpt_dir and os.path.exists(ckpt_dir):
        logging.info("loading ckpt_dir:{}".format(ckpt_dir))
        model = Swift.from_pretrained(model, ckpt_dir, inference_mode=True)
    
    template = get_template(template_type, tokenizer)

    val_dataset_path = os.path.join(args.val_dataset_root, infer_type,
                                    'puzzle_{}_test_0.json'.format(args.puzzle_id))
    logging.info("using val_dataset_path:{}".format(val_dataset_path))

    _, val_dataset_from_model = get_dataset(val_dataset_path, 1.0)

    output_folder = os.path.join(args.output_root, 'custom_infer_results', infer_type)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    output_json_file = os.path.join(output_folder, '{}_{}.jsonl'.format(dt.datetime.now().strftime('%Y%m%d_%H%M%S'), file_suffix))
    generate_infer_results(model, val_dataset_from_model, template, infer_type, output_json_file, args.puzzle_id)
    
    df = pd.read_json(path_or_buf=output_json_file, lines=True)
    correct = df[df['response']==df['label']]
    ratio = len(correct)/ len(df)
    logging.info('{} accuracy: {}/{}={:.2f}'.format(output_json_file, len(correct), len(df), ratio))
    return {infer_type : [len(correct),  len(df), ratio]}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A tool for generate infer accuracy.")
    parser.add_argument(
        "--ckpt-dir-root",
        required=True,
        type=str,
        help="The fine-tuned model checkpoint root folder.",
    )
    parser.add_argument(
        "--val-dataset-root",
        required=True,
        type=str,
        help="The root path of validation dataset path.",
    )
    parser.add_argument(
        "--model-type",
        required=True,
        type=str,
        help="The model type to inference.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=str,
        help="The infer result output root.",
    )
    parser.add_argument(
        "--infer-types",
        required=True,
        type=str,
        help="The inference type: 'pretrain-inference', 'pretrain-cot', 'finetune-inference', 'finetune-cot', 'implicit-cot'.",
    )
    parser.add_argument(
        "--puzzle-id",
        required=True,
        type=int,
        help="The puzzle to process from 1 to max value (101)."
    )
    parser.add_argument(
        "--step",
        required=True,
        default=1,
        type=int,
        help="The implicit cot finetune jump step."
    )
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    val_dataset_root = args.val_dataset_root
    if not os.path.exists(val_dataset_root):
      raise ValueError("input val_dataset_root {} not exist.".format(val_dataset_root))

    output_root = args.output_root
    if not os.path.exists(output_root):
      os.makedirs(output_root, exist_ok=True)

    infer_types = args.infer_types.split(',')
    for infer_type in infer_types:
        if infer_type not in ['pretrain-inference', 'pretrain-cot', 'finetune-inference', 'finetune-cot', 'implicit-cot']:
            raise ValueError("unknown infer_type {}. Valid values are pretrain-inference, pretrain-cot, finetune-inference, finetune-cot, implicit-cot.".format(infer_type))

    # setup logging
    logging_level = logging.INFO
    if args.debug:
      logging_level = logging.DEBUG

    logs_output_root = os.path.join(args.output_root, "logs")
    if not os.path.exists(logs_output_root):
      os.makedirs(logs_output_root, exist_ok=True)

    logging.basicConfig(
      level=logging_level,
      format="%(asctime)s [%(levelname)s] %(message)s",
      handlers=[
          RotatingFileHandler(os.path.join(logs_output_root, "infer_finetune_{}.log".format(time.strftime("%Y-%m-%d-%H-%M-%S"))),
                                      maxBytes=1024 * 1024 * 10),
          logging.StreamHandler()
      ]
    )

    logger.info(args)

    accuracy_list = []
    max_stage = 9
    for infer_type in infer_types:
        logging.info('running infer {}'.format(infer_type))
        ckpt_dir = None
        file_suffix = infer_type.replace('-', '_')
        if infer_type in ['finetune-inference', 'finetune-cot']:
            ckpt_dir_pattern = os.path.join(args.ckpt_dir_root, 'original-finetune', 'v0-*-question-split-0', 'checkpoint-*')
            ckpt_dirs = glob.glob(ckpt_dir_pattern)
            logging.info('All founded ckpt_dirs:{}'.format(ckpt_dirs))
            ckpt_dir = sorted(ckpt_dirs)[-1]
            accuracy_list.append(custom_inference(args, infer_type, ckpt_dir, file_suffix))
        elif infer_type in ['implicit-cot']:
            for stage in range(0, max_stage + 1):
                ckpt_dir_pattern = os.path.join(args.ckpt_dir_root, 'decomposition-finetune-shift-{}'.format(args.step), 'v0-*-question-split-{}'.format(stage), 'checkpoint-*')
                ckpt_dirs = glob.glob(ckpt_dir_pattern)
                logging.info('Stage {}: all founded ckpt_dirs:{}'.format(stage, ckpt_dirs))
                ckpt_dir = sorted(ckpt_dirs)[-1]
                logging.info('Stage {}: using ckpt_dir:{}'.format(stage, ckpt_dir))
                accuracy_list.append(custom_inference(args, infer_type, ckpt_dir, 'stage_{}_{}'.format(stage, file_suffix)))
                stage += 1
        else:
            accuracy_list.append(custom_inference(args, infer_type, None, file_suffix))
    
    # plots
    logging.info(accuracy_list)
            
                
