#!/usr/bin/env python3

from utils import set_seed, loader_dataset
from model import SubQuestionGenerator, load_preptrained_model

import argparse
import constants 
from dotenv import load_dotenv
import os
import numpy as np
from PIL import Image
import csv
import pandas as pd
import time
import json
import logging
from logging.handlers import RotatingFileHandler
import glob

logger = logging.getLogger(__name__)

def get_follw_up_question(data):
    if 'follow_up_questions' in data:
      return data['follow_up_questions']
    return [data['follow_up_question']]
    
def get_previous_questions_answer(questions, answers):
    qa_list = []
    for i in range(len(questions)):
        qa_list.append("Step " + str(i) + ", Question:" + questions[i] + "  Answer:" + str(answers[i]))
    
    return "\n".join(qa_list)    

def generate_tot_sub_questions(generate_model, puzzle_count, output_path, max_depth, split_type):
    model_processor = SubQuestionGenerator(generate_model)
    output_header = ['puzzle_id', 'sub_puzzle_id', 'image_path', 'origin_question', 'true_answer', 'sub_question', 'answer', 'level', 'sub_question_id']

    if split_type == "instance":
      puzzle_ids = constants.INSTANCE_SPLIT_PUZZLES
    elif split_type == "puzzle":
      puzzle_ids = constants.PUZZLE_SPLIT_PUZZLES
    elif split_type == "standard":
      puzzle_ids = np.random.choice(range(1, constants.MAX_PUZZLE_ID + 1), puzzle_count, replace=False)
    else:
      raise "unknown split type:{}".format(split_type)

    logger.debug("puzzle_ids:{}".format(puzzle_ids))

    for puzzle_id in puzzle_ids:
      output_csv_file = os.path.join(output_path, "generated_tot_sub_questions_data_{}.csv".format(puzzle_id))
      with open(output_csv_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(output_header)

        logger.info("process puzzle: {}".format(puzzle_id))
        puzzles = loader_dataset(input_data_dir, puzzle_id, puzzle_subset_size, False)
        for i in range(len(puzzles)):
          puzzle = puzzles[i]
          initial_level = 0
          model_output = model_processor.generate_output(puzzle['image_path'], puzzle['Prompt'], 'tot_sub_question')
          logger.info("=======model_output===========")
          logger.info(model_output)
          try:
            data = json.loads(model_output)
          except Exception as ex:
            logger.error("process puzzle: {}, puzzle_id:{} failed.".format(puzzle_id, puzzle['id']))
            continue
          branch_id = 1
          for sub_question in data['sub_questions']:
            sub_question_id = "{}-{}-{}".format(puzzle_id, puzzle['id'], branch_id)
            branch_id += 1
            csvwriter.writerow([puzzle_id, puzzle['id'], puzzle['image_path'], puzzle['Prompt'], puzzle['Answer'],
                              sub_question['question'], sub_question['answer'], initial_level, sub_question_id])
            
            # generate child sub question based on the inital three parent sub squestions. 
            prev_sub_questions = [sub_question['question']]
            prev_sub_answers = [sub_question['answer']]
            
            start_time = time.perf_counter()
            has_final_answer = False
            for next_step in range(1, max_depth):
                logger.info("process puzzle: {}, branch: {}, step: {}".format(puzzle_id, branch_id - 1, next_step))
                
                previous_qa = get_previous_questions_answer(prev_sub_questions, prev_sub_answers)
                next_step_output = model_processor.generate_output(puzzle['image_path'], puzzle['Prompt'], 'tot_follow_up_question', previous_qa, max_depth)
                logger.info("=======next_step_output===========")
                logger.info(next_step_output)
                
                try:
                  data = json.loads(next_step_output)
                  # break if no follow_up_questions.
                  if len(get_follw_up_question(data)) == 0:
                    # write an additional row final question and its answer
                    csvwriter.writerow([puzzle_id, puzzle['id'], puzzle['image_path'], puzzle['Prompt'], puzzle['Answer'],
                                        np.nan, data['final_answer'], np.nan, sub_question_id])
                    has_final_answer = True
                    break
                  
                  for next_step_question in get_follw_up_question(data):
                      csvwriter.writerow([puzzle_id, puzzle['id'], puzzle['image_path'], puzzle['Prompt'], puzzle['Answer'],
                                      next_step_question['question'], next_step_question['answer'], next_step, sub_question_id])
                      # add the generate sub question.
                      prev_sub_questions.append(next_step_question['question'])
                      prev_sub_answers.append(next_step_question['answer'])
                  time.sleep(0.1)
                except Exception as ex:
                  logger.error("process sub_question_id: {} failed.".format(sub_question_id))
                  break
            # if gpt-4o finally not generate answer, add one records to indicate.
            if not has_final_answer:
              csvwriter.writerow([puzzle_id, puzzle['id'], puzzle['image_path'], puzzle['Prompt'], puzzle['Answer'], np.nan, 'NO_FINAL_ANSWER', np.nan, sub_question_id])
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            logger.info("sub_question_id loop: {} cost time: {:.4f} seconds".format(sub_question_id, elapsed_time))
            # flush per puzzle per chain
            csvfile.flush()


def main(args):
    # 0. set seed
    set_seed(args.seed)
  
    # generate tot data
    generate_tot_sub_questions(args.generate_model, args.puzzle_count, args.output_root, args.max_depth, args.split_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A tool for generate tot subquestion for smart101 dataset.")
    parser.add_argument(
        "--data-root",
        type=str,
        default="./dataset/SMART101-release-v1/SMART101-Data/",
        help="location of the smart101 dataset.",
    )
    parser.add_argument(
        "--puzzle-count",
        default=10,
        type=int,
        help="The number of random picked puzzle from 1 to max value (101)."
    )
    parser.add_argument(
        "--subset-size",
        default=1,
        type=int,
        help="The size of subset puzzle to process for each puzzle folder, max value(2000)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="seed to use"
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="output",
        help="The output folder to save all the results.",
    )
    parser.add_argument(
        "--generate-model",
        type=str,
        default="gpt-4o",
        help="The model used to generate sub questions and captions.",
    )
    parser.add_argument(
        "--max-depth",
        default=10,
        type=int,
        help="The max depth of the tot."
    )
    parser.add_argument(
        "--split-type",
        type=str,
        default="standard",
        help="type of data split: stanard/puzzle/instance"
    )
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')

    args = parser.parse_args()

    input_data_dir = args.data_root
    if not os.path.exists(input_data_dir):
      raise ValueError("input data path {} not exist.".format(input_data_dir))

    output_root = args.output_root
    if not os.path.exists(output_root):
      os.makedirs(output_root)

    puzzle_count = args.puzzle_count
    if puzzle_count < 1 or puzzle_count > 101:
      raise ValueError("puzzle count can only range from 1 to 101.")

    puzzle_subset_size = args.subset_size
    if puzzle_subset_size < 0 or puzzle_subset_size > 2000:
      raise ValueError("puzzle subset size can only range from 1 to 2000.")

    # setup logging
    logging_level = logging.INFO
    if args.debug:
      logging_level = logging.DEBUG

    logging.basicConfig(
      level=logging_level,
      format="%(asctime)s [%(levelname)s] %(message)s",
      handlers=[
          RotatingFileHandler(os.path.join(args.output_root,"tot_data_generator-{}.log".format(time.strftime("%Y-%m-%d-%H-%M-%S"))),
                                      maxBytes=1024 * 1024 * 10),
          logging.StreamHandler()
      ]
    )
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logger.info(args)
    # load env from .env file
    load_dotenv()

    main(args)
