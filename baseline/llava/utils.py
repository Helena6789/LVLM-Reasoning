#!/usr/bin/env python3

import constants

import csv
import numpy as np
import os
import torch

def set_seed(seed):
    manualSeed = np.random.randint(10000) if seed == -1 else seed
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.deterministic = True
    print("seed = %d" % (manualSeed))

def read_csv(csvfilename, puzzle_id):
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

# input: [1, 2, 3, 4, 5]
# output: A: 1, B: 2, C: 3, D: 4, E: 5
def construct_option_text(options):
  char_offset = 0
  option_texts = []
  for option in options:
    option_texts.append(chr(ord('A')+char_offset) + ": "+ str(option))
    char_offset += 1
  return ",".join(option_texts)

def construct_question(question, options, include_llava_prompt):
  if include_llava_prompt:
    return constants.LLAVA_PROMPT_TEMPLATE.format(question + " " + options + ".\n" + constants.LLAVA_ANSWER_TEMPATE)
  return question + " " + options

def loader_dataset(input_data_dir, puzzle_id, puzzle_subset_size, include_llava_prompt):
    # Initialize list to hold all prompt_list data
    processed_puzzle = []

    subfolder = os.path.join(input_data_dir, str(puzzle_id))
    if not os.path.exists(subfolder):
        return

    csv_file = os.path.join(subfolder, 'puzzle_{}.csv'.format(puzzle_id))
    # read the csv file
    puzzles = read_csv(csv_file, puzzle_id)
    # process each record in the puzzle_x.csv
    if puzzle_subset_size == len(puzzles):
        subset_puzzles = puzzles
    else:
        subset_puzzles = np.random.choice(puzzles, puzzle_subset_size, replace=False)

    for puzzle in subset_puzzles:
        #e.g. flower, disk, book, drink, ball
        options = [puzzle['A'], puzzle['B'], puzzle['C'], puzzle['D'], puzzle['E']]
        puzzle['Prompt'] = construct_question(puzzle['Question'], construct_option_text(options), include_llava_prompt)

        # construct image
        image_subfolder = os.path.join(subfolder, 'img')
        image_path = os.path.join(image_subfolder, puzzle['image'])
        puzzle['image_path'] =  image_path

        processed_puzzle.append(puzzle)

    return processed_puzzle