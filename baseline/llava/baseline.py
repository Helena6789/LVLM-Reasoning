#!/usr/bin/env python3

import argparse
import os
from PIL import Image
import numpy as np
from model import load_preptrained_model
import csv
import constants
import time
from accuracy import plot_accuracy
from utils import loader_dataset, set_seed, read_csv

# Transfer llava-1-5 prompt to llava-1-6
def get_prompt(prompt, model_id):
    if 'llava-1.5-7b' in model_id:
        return prompt.replace('[INST]', 'USER').replace('[/INST]', 'ASSISTANT:')
    return prompt

def main(args):
    # 0. set seed
    set_seed(args.seed)

    # 2. load pretrained model
    print("=================predict===============")
    generation_configs = {'do_sample': False, 'max_new_tokens': 128}

    if not args.test:
      client = load_preptrained_model(args.model_id)

    # 3. get response from model and save response
    output_header = ['puzzle_id', 'image_id', 'prompt', 'true_answer', 'predit_answer', 'time_cost(s)']
    output_csv_file = os.path.join(args.output_root, 'output.csv')
    with open(output_csv_file, 'w', newline='') as csvfile:
      csvwriter = csv.writer(csvfile)
      csvwriter.writerow(output_header)

      for puzzle_id in range(1, args.puzzle_max + 1):
        # load data
        print("=================load data===============")
        puzzles = loader_dataset(input_data_dir, puzzle_id, puzzle_subset_size, True)
        for i in range(len(puzzles)):
          puzzle = puzzles[i]
          # get model predict result
          start_time = time.perf_counter()
          #call the model to get the predict output.
          if args.test:
             predict_answers = np.random.choice(['A', 'B', 'C', 'D', 'E'], 1)
          else:
             prompt = get_prompt(puzzle['Prompt'], args.model_id)
             predict_answers = client.generate([prompt], [Image.open(puzzle['image_path'])], **generation_configs)

          end_time = time.perf_counter()
          elapsed_time = end_time - start_time
          print("No. {} predict {} cost time: {:.4f} seconds".format(puzzle['id'], puzzle['image'], elapsed_time))

          csvwriter.writerow([puzzle_id, puzzle['image'], prompt, puzzle['Answer'],
                              predict_answers[0].strip(), float("{:.4f}".format(elapsed_time))])
          if i % 50 == 0:
            csvfile.flush()

    # 5. accuracy
    plot_accuracy(output_csv_file, args.smart_info_v2_csv, args.puzzle_max, args.output_root, args.subset_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A tool for Llava baseline.")
    parser.add_argument(
        "--data-root",
        type=str,
        default="./dataset/SMART101-release-v1/SMART101-Data/",
        help="location of the smart101 dataset.",
    )
    parser.add_argument(
        "--puzzle-max",
        default=1,
        type=int,
        help="The subfolder of the puzzle to process from 1 to max value (101)."
    )
    parser.add_argument(
        "--subset-size",
        default=10,
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
        "--model-id",
        type=str,
        default="./models--llava-hf--llava-v1.6-mistral-7b-hf/snapshots/d7f9c48a21f52c49df120dcbf87e05767734c71a",
        help="location of llava model.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="output",
        help="The output folder to save all the results.",
    )
    parser.add_argument(
       "--test",
       action="store_true",
       help="test random generate output."
    )
    parser.add_argument(
        "--smart-info-v2-csv",
        type=str,
        default="./dataset/SMART_info_v2.csv",
        help="The smart101 information csv file (include puzzle difficulty and puzzle type).",
    )

    args = parser.parse_args()

    print(args)

    input_data_dir = args.data_root
    if not os.path.exists(input_data_dir):
      raise ValueError("input data path {} not exist.".format(input_data_dir))

    output_root = args.output_root
    if not os.path.exists(output_root):
      os.makedirs(output_root)

    puzzle_subfolder_max = args.puzzle_max
    if puzzle_subfolder_max < 1 or puzzle_subfolder_max > 101:
      raise ValueError("puzzle max can only range from 1 to 101.")

    puzzle_subset_size = args.subset_size
    if puzzle_subset_size < 0 or puzzle_subset_size > 2000:
      raise ValueError("puzzle subset size can only range from 1 to 2000.")

    smart_info_v2_csv = args.smart_info_v2_csv
    if not os.path.exists(smart_info_v2_csv):
      raise ValueError("input smart csv file path {} not exist.".format(smart_info_v2_csv))

    main(args)