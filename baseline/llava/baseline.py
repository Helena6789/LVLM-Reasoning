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

def construct_option_text(options):
  char_offset = 0
  option_texts = []
  for option in options:
    option_texts.append(chr(ord('A')+char_offset) + ": "+ str(option))
    char_offset += 1
  return ",".join(option_texts)

def construct_question(question, options):
  return PROMPT_TEMPLATE.format(question + " " + options + ".\n" + ANSWER_TEMPATE)

def loader_dataset(input_data_dir, puzzle_id, puzzle_subset_size):
  # Initialize list to hold all prompt_list data
  prompt_list = []
  image_list = []
  image_id_list = []
  answers_list = []

  subfolder = os.path.join(input_data_dir, str(puzzle_id))
  if not os.path.exists(subfolder):
      return

  csv_file = os.path.join(subfolder, 'puzzle_{}.csv'.format(puzzle_id))
  # read the csv file
  puzzles = pd.read_csv(csv_file)
  ids = puzzles['id'].tolist()
  questions = puzzles['Question'].tolist()
  answers = puzzles['Answer'].tolist()
  image_paths = puzzles['image'].tolist()
  options_a = puzzles['A'].tolist()
  options_b = puzzles['B'].tolist()
  options_c = puzzles['C'].tolist()
  options_d = puzzles['D'].tolist()
  options_e = puzzles['E'].tolist()
  notes = puzzles['Note'].tolist()

  # process each record in the puzzle_x.csv
  subset_puzzle_ids = np.random.choice(ids, puzzle_subset_size)
  # print("subfolder_id:{}, subset_puzzle_ids: {}".format(subfolder_id, subset_puzzle_ids))
  for id in subset_puzzle_ids:
      i = int(id) - 1
      
      #e.g. flower, disk, book, drink, ball
      options = [options_a[i], options_b[i], options_c[i],
                  options_d[i], options_e[i]]
      prompt_list.append(construct_question(questions[i], construct_option_text(options)))

      # construct image
      image_subfolder = os.path.join(subfolder, 'img')
      image_path = os.path.join(image_subfolder, image_paths[i])
      image_list.append(image_path)

      image_id_list.append(image_paths[i])
      answers_list.append(answers[i])

  return prompt_list, image_list, image_id_list, answers_list

def main(args):
    # 0. set seed
    set_seed(args.seed)

    # 2. load pretrained model 
    print("=================predict===============")
    generation_configs = {'do_sample': False, 'max_new_tokens': 128}
    client = load_preptrained_model(args.model_id)
    
    # 3. get response from model and save response 
    output_header = ['puzzle_id', 'image_id', 'prompt', 'true_answer', 'predit_answer', 'time_cost(s)']
    with open(args.output_file, 'w', newline='') as csvfile:
      csvwriter = csv.writer(csvfile)
      csvwriter.writerow(output_header)

      for puzzle_id in range(1, args.puzzle_max + 1):
        # load data
        print("=================load data===============")
        prompt_list, image_list, image_id_list, answers_list = loader_dataset(input_data_dir, 
                                                                              puzzle_id,
                                                                              puzzle_subset_size)
        # print(prompt_list)
        for i in range(len(prompt_list)):
          # get model predict result
          start_time = time.perf_counter() 
          predict_answers = client.generate([prompt_list[i]], [Image.open(image_list[i])], **generation_configs)
          end_time = time.perf_counter()
          elapsed_time = end_time - start_time
          print("No. {} predict {} cost time: {:.4f} seconds".format(i, image_id_list[i], elapsed_time))

          csvwriter.writerow([puzzle_id, image_id_list[i], prompt_list[i], answers_list[i], 
                              predict_answers[0].strip(), float("{:.4f}".format(elapsed_time))])
          if i % 50 == 0:
            csvfile.flush()
    
    # 5. accuracy 
    # TODO: calculate acc

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
        "--output-file",
        type=str,
        default="output.csv",
        help="output the predict answer.",
    )

    args = parser.parse_args()

    print(args)

    input_data_dir = args.data_root
    if not os.path.exists(input_data_dir):
      raise ValueError("input data path {} not exist.".format(input_data_dir))

    puzzle_subfolder_max = args.puzzle_max 
    if puzzle_subfolder_max < 1 or puzzle_subfolder_max > 101:
      raise ValueError("puzzle max can only range from 1 to 101.")

    puzzle_subset_size = args.subset_size 
    if puzzle_subset_size < 0 or puzzle_subset_size > 2000:
      raise ValueError("puzzle subset size can only range from 1 to 2000.")
    
    main(args)
