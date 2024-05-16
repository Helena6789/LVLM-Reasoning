#!/usr/bin/env python3

import argparse
import pandas as pd
import os
from PIL import Image
import numpy as np
import torch
from model import load_preptrained_model
import csv 
import time
from accuracy import get_group_puzzle_accuracy

PROMPT_TEMPLATE = "[INST] <image>\n{} [/INST]"
ANSWER_TEMPATE = "Please provide the answer in a single upper case letter, e.g: A"

def set_seed(seed):
    manualSeed = np.random.randint(10000) if seed == -1 else seed
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.deterministic = True
    print("seed = %d" % (manualSeed))

# input: [1, 2, 3, 4, 5]
# output: A: 1, B: 2, C: 3, D: 4, E: 5
def construct_option_text(options):
  char_offset = 0
  option_texts = []
  for option in options:
    option_texts.append(chr(ord('A')+char_offset) + ": "+ str(option))
    char_offset += 1
  return ",".join(option_texts)

def construct_question(question, options):
  return PROMPT_TEMPLATE.format(question + " " + options + ".\n" + ANSWER_TEMPATE)

def read_csv(csvfilename, puzzle_id):
    import csv

    qa_info = []
    with open(csvfilename, newline="") as csvfile:
        datareader = csv.DictReader(csvfile)
        for row in datareader:
            row["puzzle_id"] = str(puzzle_id)
            if len(row["A"]) == 0:
                row["A"] = "A"
                row["B"] = "B"
                row["C"] = "C"
                row["D"] = "D"
                row["E"] = "E"
            qa_info.append(row)
    return qa_info

def loader_dataset(input_data_dir, puzzle_id, puzzle_subset_size):
  # Initialize list to hold all prompt_list data
  processed_puzzle = []

  subfolder = os.path.join(input_data_dir, str(puzzle_id))
  if not os.path.exists(subfolder):
      return

  csv_file = os.path.join(subfolder, 'puzzle_{}.csv'.format(puzzle_id))
  # read the csv file
  puzzles = read_csv(csv_file, puzzle_id)
  # process each record in the puzzle_x.csv
  subset_puzzles = np.random.choice(puzzles, puzzle_subset_size)

  for puzzle in subset_puzzles:      
      #e.g. flower, disk, book, drink, ball
      options = [puzzle['A'], puzzle['B'], puzzle['C'], puzzle['D'], puzzle['E']]
      puzzle['Question'] = construct_question(puzzle['Question'], construct_option_text(options))

      # construct image
      image_subfolder = os.path.join(subfolder, 'img')
      image_path = os.path.join(image_subfolder, puzzle['image'])
      puzzle['image_path'] =  image_path

      processed_puzzle.append(puzzle)

  return processed_puzzle

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
        puzzles = loader_dataset(input_data_dir, puzzle_id, puzzle_subset_size)
        for i in range(len(puzzles)):
          puzzle = puzzles[i]
          # get model predict result
          start_time = time.perf_counter() 
          #call the model to get the predict output. 
          if args.test:
             predict_answers = np.random.choice(['A', 'B', 'C', 'D', 'E'], 1)
          else:
             predict_answers = client.generate([puzzle['Question']], [Image.open(puzzle['image_path'])], **generation_configs)
          
          end_time = time.perf_counter()
          elapsed_time = end_time - start_time
          print("No. {} predict {} cost time: {:.4f} seconds".format(puzzle['id'], puzzle['image'], elapsed_time))

          csvwriter.writerow([puzzle_id, puzzle['image'], puzzle['Question'], puzzle['Answer'], 
                              predict_answers[0].strip(), float("{:.4f}".format(elapsed_time))])
          if i % 50 == 0:
            csvfile.flush()
    
    # 5. accuracy 
    get_group_puzzle_accuracy(output_csv_file, args.puzzle_max, args.output_root)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A tool for Llava baseline.")
    parser.add_argument(
        "--data-root",
        type=str,
        default="/home/qh/LVLM-Reasoning/dataset/SMART101-release-v1/SMART101-Data/",
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
        default="/project/models--llava-hf--llava-v1.6-mistral-7b-hf/snapshots/d7f9c48a21f52c49df120dcbf87e05767734c71a",
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
    
    main(args)
