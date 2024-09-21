#!/usr/bin/env python3

import constants

from utils import set_seed
import argparse
import pandas as pd
import os
import matplotlib.pyplot as plt
import glob
import json
import numpy as np
import time
import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)

def generate_llama_output(df, minmum_level, prompt_type, data_type, full_image_path, sub_image_path, include_sub_qa):
    output_json_list = []

    for image_path, group in df.groupby('image_path'):
        messages = []
        subimage_paths = []
        for _, row in group.iterrows():
            if int(row['level']) < minmum_level:
                continue
            if pd.notna(row['sub_question']) and pd.notna(row['answer']) and include_sub_qa:
                messages.append({'content': '<image>\n{}'.format(row['sub_question']), 'role': 'user'})
                messages.append({'content': row['answer'], 'role': 'assistant'})
            elif pd.isna(row['sub_question']):
                # For test data we need to provide two different prompt: one for original inference, one for cot inference.
                if data_type == 'test':
                    if prompt_type == 'cot':
                        messages.append({'content': constants.SIMPLE_COT_PROMPT.format(row['origin_question'], constants.SIMPLE_COT_OUTPUT), 'role': 'user'})
                    else:
                        prompt = '<image>\nPlease answer the following questions: {}. You should provide the answer in a single upper case letter, e.g: A'
                        messages.append({'content': prompt.format(row['origin_question']), 'role': 'user'})
                else:
                    # if train data need to include the sub questions and answer, add a different prompt to help llm understand the context.
                    if include_sub_qa:
                        prompt = '<image>\nBased on the previous answered questions, please answer the following questions: {}. You should provide the answer in a single upper case letter, e.g: A'
                    else:
                        prompt = '<image>\n{}'

                    messages.append({'content': prompt.format(row['origin_question']), 'role': 'user'})
                messages.append({'content': row['true_answer'], 'role': 'assistant'})
            
            if pd.notna(row['subquestion_image_path']):
                subimage_paths.append(os.path.join(sub_image_path, row['subquestion_image_path']))
        
        if messages:
            output_json_list.append({
                'messages': messages,
                'images': subimage_paths if include_sub_qa else [os.path.join(full_image_path, image_path)],
                'id': row['sub_puzzle_id']
            })
    return output_json_list

def split_data(data, train_pct=0.8, val_pct=0.05, test_pct=0.15):
    data = np.array(data)
    np.random.shuffle(data)
    
    train_size = int(train_pct * len(data))
    val_size = int(val_pct * len(data))
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    return train_data, val_data, test_data

def output_json_file(df, output_data_path, split_type, prompt_type, data_type,
                     skip_stage_step, full_image_path, sub_image_path, include_sub_qa):
    if len(df) == 0:
        return

    if split_type == 'branch':
        questions_ids = df['sub_question_id'].unique().tolist()
    elif split_type == 'question':
        # only pick one branch from the sub question, otherwise the test data might have same sub question as train data.
        questions_ids = df.groupby(['puzzle_id', 'sub_puzzle_id']).head(1)['sub_question_id'].unique().tolist()

    df = df[df['sub_question_id'].isin(questions_ids)]

    # convert nan level as previous row value + 1
    df.loc[:, 'level'] = df['level'].ffill() + df['level'].isna().astype(float)

    max_split_level = 10 if data_type != 'test' and include_sub_qa else 1
    for minmum_level in range(0, max_split_level, skip_stage_step):
        output_filename = os.path.join(output_data_path, "puzzle_{}_{}_{}.json".format(df['puzzle_id'].unique().tolist()[0], data_type, minmum_level))
        output_json_list = generate_llama_output(df, minmum_level, prompt_type, data_type, full_image_path, sub_image_path, include_sub_qa)

        if len(output_json_list) == 0:
            continue
        
        logger.info("write {} file {} with {} records.".format(data_type, output_filename, len(output_json_list)))
        with open(output_filename, 'w') as f:
            json.dump(output_json_list, f, indent=2)

def generate_llava_fune_tune_output(args, train_df, val_df, test_df, output_data_type):
    
    prompt_type= 'default'
    include_sub_qa_train = False
    only_correct_qa = False
    include_train_file = False

    if output_data_type == 'pretrain-inference':
        prompt_type= 'default'
    elif output_data_type == 'pretrain-cot':
        prompt_type = 'cot'
    # only keep one copy original question finetune.
    elif output_data_type == 'finetune-inference':
        prompt_type= 'default'
        include_train_file = True
    elif output_data_type == 'finetune-cot':
        prompt_type = 'cot'
    elif output_data_type == 'implicit-cot':
        prompt_type= 'default'
        include_train_file = True
        include_sub_qa_train = True
    
    # for implicit cot: combine train and validation set and filter only decomposition correct answer from gpt-4o
    if output_data_type == 'implicit-cot':
        correct_train_ids = train_df[(train_df['true_answer'] == train_df['answer']) & train_df['sub_question'].isna()]['sub_question_id'].to_list()
        train_df = train_df[train_df['sub_question_id'].isin(correct_train_ids)]

        correct_val_ids = val_df[(val_df['true_answer'] == val_df['answer']) & val_df['sub_question'].isna()]['sub_question_id'].to_list()
        val_df = val_df[val_df['sub_question_id'].isin(correct_val_ids)]

        train_df = pd.concat([train_df, val_df])

    # generate regular train-validation-test data
    output_data_path = os.path.join(args.output_root, str(args.puzzle_id), '{}_split'.format(args.split_type), output_data_type)
    if not os.path.exists(output_data_path):
      os.makedirs(output_data_path)

    if include_train_file:
        output_json_file(train_df, output_data_path, args.split_type, prompt_type, 'train',
                         args.skip_stage_step, args.smart101_data_root, args.clip_image_root, include_sub_qa_train)
    
    # for implicit-cot, train and validation merged together.
    if include_train_file and output_data_type != 'implicit-cot':
        output_json_file(val_df, output_data_path, args.split_type, prompt_type, 'val',
                        args.skip_stage_step, args.smart101_data_root, args.clip_image_root, include_sub_qa_train)
    
    output_json_file(test_df, output_data_path, args.split_type, prompt_type, 'test',
                     args.skip_stage_step, args.smart101_data_root, args.clip_image_root, False)

    test_sub_puzzle_ids =  sorted(test_df['sub_puzzle_id'].unique().tolist())
    logger.debug('{} test total: {}, question_ids checksum:{}'.format(output_data_type, len(test_sub_puzzle_ids),  hash(tuple(test_sub_puzzle_ids))))
    val_sub_puzzle_ids = sorted(val_df['sub_puzzle_id'].unique().tolist())
    logger.debug('{} val_validation total:{} question_ids checksum:{}'.format(output_data_type, len(val_sub_puzzle_ids), hash(tuple(val_sub_puzzle_ids))))
    train_sub_puzzle_ids = sorted(train_df['sub_puzzle_id'].unique().tolist())
    logger.debug('{} train_validation total:{} question_ids:{} checksum'.format(output_data_type, len(train_sub_puzzle_ids), hash(tuple(train_sub_puzzle_ids))))
    
    # validate train and test data should no overlap.
    if len(set(train_sub_puzzle_ids).intersection(set(test_sub_puzzle_ids))) != 0:
        raise "test data contains train sub puzzle ids"
    # validate train and val data should no overlap.
    if output_data_type != 'implicit-cot' and len(set(train_sub_puzzle_ids).intersection(set(val_sub_puzzle_ids))) != 0:
        raise "val data contains train sub puzzle ids"

def main(args):
    # 0. set seed
    set_seed(args.seed)

    # split data
    df = pd.read_csv(args.input_data)
    all_sub_puzzle_ids = sorted(df['sub_puzzle_id'].unique().tolist())
    train_ids, val_ids, test_ids = split_data(all_sub_puzzle_ids)  
    train_df = df[df['sub_puzzle_id'].isin(train_ids)]
    val_df = df[df['sub_puzzle_id'].isin(val_ids)]
    test_df =  df[df['sub_puzzle_id'].isin(test_ids)]

    # generate LLava fine-tunning output
    for output_data_type in ['pretrain-inference', 'pretrain-cot', 'finetune-inference', 'finetune-cot', 'implicit-cot']:
        generate_llava_fune_tune_output(args, train_df, val_df, test_df, output_data_type)

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
        help="The input combination data including subquesiton decompostion and image splits.",
    )
    parser.add_argument(
        "--smart101-data-root",
        type=str,
        required=True,
        help="location of the smart101 dataset.",
    )
    parser.add_argument(
        "--clip-image-root",
        type=str,
        required=True,
        help="The path saved all the clip images.",
    )
    parser.add_argument(
        "--puzzle-id",
        required=True,
        type=int,
        help="The puzzle id data to generate."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed to use"
    )
    parser.add_argument(
        "--split-type",
        type=str,
        default="question",
        help="Type to split train and test data: branch or question. For question type, only filter one branch of a question to finetune dataset",
    )
    parser.add_argument(
        "--skip-stage-step",
        type=int,
        default=1,
        help="The number of stage to skip when generate stage level data."
    )
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')

    args = parser.parse_args()
    
    input_data_dir = args.input_data
    if not os.path.exists(input_data_dir):
      raise ValueError("input data path {} not exist.".format(input_data_dir))

    output_root = args.output_root
    if not os.path.exists(output_root):
      os.makedirs(output_root)

    puzzle_id = args.puzzle_id
    if puzzle_id < 1 or puzzle_id > 101:
      raise ValueError("puzzle id can only range from 1 to 101.")

    if args.split_type not in ['branch', 'question']:
        raise 'unknown split type {} for finetune dataset: only accept value branch or question'.format(args.split_type)

    # setup logging
    logging_level = logging.INFO
    if args.debug:
      logging_level = logging.DEBUG

    logging.basicConfig(
      level=logging_level,
      format="%(asctime)s [%(levelname)s] %(message)s",
      handlers=[
          RotatingFileHandler(os.path.join(args.output_root,"finetune_data_generator-{}.log".format(time.strftime("%Y-%m-%d-%H-%M-%S"))),
                                      maxBytes=1024 * 1024 * 10),
          logging.StreamHandler()
      ]
    )
    logger.info(args)

    main(args)