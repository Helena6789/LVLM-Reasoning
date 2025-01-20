#!/usr/bin/env python3

from utils import set_seed, loader_dataset

import argparse
import constants 
import os
import numpy as np
import csv
import pandas as pd
import time
import json
import logging
from logging.handlers import RotatingFileHandler
import glob

logger = logging.getLogger(__name__)


def generate_data(puzzle_count, output_path, data_root):
    output_header = ['puzzle_id', 'sub_puzzle_id', 'image_path', 'origin_question', 'true_answer', 'sub_question', 'answer', 'level', 'sub_question_id', 'subquestion_image_path']

    for puzzle_id in range(1, constants.MAX_PUZZLE_ID + 1):
      if puzzle_id in constants.INSTANCE_SPLIT_PUZZLES:
        continue
      output_csv_file = os.path.join(output_path, "generate_data_no_sub_question_puzzle_{}.csv".format(puzzle_id))
      with open(output_csv_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(output_header)

        logger.info("process puzzle: {}".format(puzzle_id))
        puzzles = loader_dataset(input_data_dir, puzzle_id, puzzle_subset_size, False)
        for i in range(len(puzzles)):
          puzzle = puzzles[i]
          sub_question_id = "{}-{}-{}".format(puzzle_id, puzzle['id'], 1)
          image_path = puzzle['image_path'].replace(data_root,"")
          csvwriter.writerow([puzzle_id, puzzle['id'], image_path, puzzle['Prompt'], puzzle['Answer'], np.nan, 'NO_FINAL_ANSWER', np.nan, sub_question_id, np.nan])
      logger.info("output processed file: {}".format(output_csv_file))

def main(args):
    # 0. set seed
    set_seed(args.seed)
  
    # generate tot data
    generate_data(args.puzzle_count, args.output_root, args.data_root)

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
          RotatingFileHandler(os.path.join(args.output_root,"tot_data_generator_no_sub_qa-{}.log".format(time.strftime("%Y-%m-%d-%H-%M-%S"))),
                                      maxBytes=1024 * 1024 * 10),
          logging.StreamHandler()
      ]
    )
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logger.info(args)
    main(args)
