#!/usr/bin/env python3

import argparse
from baseline import set_seed, loader_dataset
import constants 
from dotenv import load_dotenv
import os
import numpy as np
from PIL import Image
import csv
import pandas as pd
import time
from model import SubQuestionGenerator, load_preptrained_model
import json


def construct_prompt_with_sub_questions(sub_questions, sub_answers, new_question, limit_option = False):
  qa_list = []
  for i in range(len(sub_questions)):
    qa_list.append("Question:" + sub_questions[i] + "  Answer:" + str(sub_answers[i]))
  if len(qa_list) == 0:
    qa_str = "No previous answered quetions."
  else:
    qa_str = "\n".join(qa_list)

  prompt = constants.BASELINE_SUB_QUESTION_PROMTP.format(qa_str, new_question)
  if not limit_option:
    return constants.LLAVA_PROMPT_TEMPLATE.format(prompt)
  return constants.LLAVA_PROMPT_TEMPLATE.format("{}\n{}".format(prompt, constants.LLAVA_ANSWER_TEMPATE))


def construct_prompt_with_caption(caption, question):
  prompt = constants.BASELINE_CAPTION_PROMTP.format(caption, question)
  return constants.LLAVA_PROMPT_TEMPLATE.format("{}\n{}".format(prompt, constants.LLAVA_ANSWER_TEMPATE))


def generate_sub_questions(generate_model, puzzle_count, output_csv_file):
    model_processor = SubQuestionGenerator(generate_model)
    output_header = ['puzzle_id', 'sub_puzzle_id', 'image_path', 'origin_question', 'sub_question', 'sub_question_answer']

    with open(output_csv_file, 'w', newline='') as csvfile:
      csvwriter = csv.writer(csvfile)
      csvwriter.writerow(output_header)

      puzzle_ids = np.random.choice(range(1, constants.MAX_PUZZLE_ID + 1), puzzle_count)
      puzzle_ids = np.append(puzzle_ids, constants.ADDITIONAL_PUZZLES)
      for puzzle_id in puzzle_ids:
        puzzles = loader_dataset(input_data_dir, puzzle_id, puzzle_subset_size, False)
        for i in range(len(puzzles)):
          puzzle = puzzles[i]
          model_output = model_processor.generate_output(puzzle['image_path'], puzzle['Prompt'], 'sub_question')
          print(model_output)
          data = json.loads(model_output)
          for sub_question in data['sub_questions']:
            csvwriter.writerow([puzzle_id, puzzle['id'], puzzle['image_path'], puzzle['Prompt'],
                              sub_question['question'], sub_question['answer']])

          # write an additional row final question and its answer
          csvwriter.writerow([puzzle_id, puzzle['id'], puzzle['image_path'], puzzle['Prompt'],
                              np.nan, puzzle['Answer']])
          if i % 50 == 0:
            csvfile.flush()

def generate_image_captions(generate_model, puzzle_count, output_csv_file):
    model_processor = SubQuestionGenerator(generate_model)
    output_header = ['puzzle_id', 'sub_puzzle_id', 'image_path', 'origin_question', 'caption', 'answer']

    with open(output_csv_file, 'w', newline='') as csvfile:
      csvwriter = csv.writer(csvfile)
      csvwriter.writerow(output_header)

      puzzle_ids = np.random.choice(range(1, constants.MAX_PUZZLE_ID + 1), puzzle_count)
      puzzle_ids = np.append(puzzle_ids, constants.ADDITIONAL_PUZZLES)
      for puzzle_id in puzzle_ids:
        puzzles = loader_dataset(input_data_dir, puzzle_id, puzzle_subset_size, False)
        for i in range(len(puzzles)):
          puzzle = puzzles[i]
          model_output = model_processor.generate_output(puzzle['image_path'], puzzle['Prompt'], 'caption')
          print(model_output)
          json_data = json.loads(model_output)
          csvwriter.writerow([puzzle_id, puzzle['id'], puzzle['image_path'], puzzle['Prompt'],
                              json_data['caption'], puzzle['Answer']])
          if i % 50 == 0:
            csvfile.flush()

def baseline_sub_questions(baseline_model_id, output_root, input_csv_data):
    print("=================predict===============")
    generation_configs = {'do_sample': False, 'max_new_tokens': 2048}

    client = load_preptrained_model(baseline_model_id)

    # 3. get response from model and save response
    output_header = ['puzzle_id', 'sub_puzzle_id', 'image_path', 'prompt',
                     'sub_question', 'sub_question_answer', 'predit_answer', 'time_cost(s)']
    output_csv_file = os.path.join(output_root, 'output_sub_questions.csv')
    with open(output_csv_file, 'w', newline='') as csvfile:
      csvwriter = csv.writer(csvfile)
      csvwriter.writerow(output_header)

      df = pd.read_csv(input_csv_data)
      i = 0
      for puzzle_id, puzzles in df.groupby("puzzle_id"):
          prev_sub_questions = []
          prev_sub_answers = []
          for _, puzzle in puzzles.iterrows():
            if type(puzzle['sub_question'])== float and np.isnan(puzzle['sub_question']):
              prompt = construct_prompt_with_sub_questions(prev_sub_questions, prev_sub_answers, puzzle['origin_question'], True)
            else:
              prompt = construct_prompt_with_sub_questions(prev_sub_questions, prev_sub_answers, puzzle['sub_question'])

            # get model predict result
            start_time = time.perf_counter()
            #call the model to get the predict output.
            predict_answers = client.generate([prompt], [Image.open(puzzle['image_path'])], **generation_configs)
            end_time = time.perf_counter()

            elapsed_time = end_time - start_time
            print("No. {} predict {} cost time: {:.4f} seconds".format(puzzle['puzzle_id'], puzzle['sub_puzzle_id'], elapsed_time))

            csvwriter.writerow([puzzle_id, puzzle['sub_puzzle_id'], puzzle['image_path'], prompt,
                              puzzle['sub_question'], puzzle['sub_question_answer'],
                              predict_answers[0].strip(), float("{:.4f}".format(elapsed_time))])
            if i % 50 == 0:
              csvfile.flush()

            prev_sub_questions.append(puzzle['sub_question'])
            prev_sub_answers.append(predict_answers[0].strip())
            i += 1


def baseline_captions(baseline_model_id, output_root, input_csv_data, use_blank_image):
    print("=================predict based on captions===============")
    generation_configs = {'do_sample': False, 'max_new_tokens': 2048}

    client = load_preptrained_model(baseline_model_id)

    # 3. get response from model and save response
    output_header = ['puzzle_id', 'sub_puzzle_id', 'image_path', 'prompt', 
                     'caption', 'answer', 'predit_answer', 'time_cost(s)']
    if use_blank_image:
      output_csv_file = os.path.join(output_root, 'output_captions_blank_image.csv')
    else:
      output_csv_file = os.path.join(output_root, 'output_captions.csv')

    with open(output_csv_file, 'w', newline='') as csvfile:
      csvwriter = csv.writer(csvfile)
      csvwriter.writerow(output_header)

      df = pd.read_csv(input_csv_data)
      i = 0
      for puzzle_id, puzzles in df.groupby("puzzle_id"):
          for _, puzzle in puzzles.iterrows():
            
            prompt = construct_prompt_with_caption(puzzle['caption'], puzzle['origin_question'])
            # get model predict result
            start_time = time.perf_counter()
            #call the model to get the predict output.
            if use_blank_image:
              predict_answers = client.generate([prompt], [Image.open("./img/blank.png")], **generation_configs)
            else:
              predict_answers = client.generate([prompt], [Image.open(puzzle['image_path'])], **generation_configs)
            end_time = time.perf_counter()

            elapsed_time = end_time - start_time
            print("No. {} predict {} cost time: {:.4f} seconds".format(puzzle['puzzle_id'], puzzle['sub_puzzle_id'], elapsed_time))

            csvwriter.writerow([puzzle_id, puzzle['sub_puzzle_id'], puzzle['image_path'], prompt,
                              puzzle['caption'], puzzle['answer'], predict_answers[0].strip(), float("{:.4f}".format(elapsed_time))])
            if i % 50 == 0:
              csvfile.flush()

            i += 1

def main(args):
    # 0. set seed
    set_seed(args.seed)

    output_sub_questions_file = os.path.join(args.output_root, 'generated_sub_questions_data.csv')
    output_captions_file = os.path.join(args.output_root, 'generated_captions_data.csv')
    # 1. whether generate sub_question data
    if args.generate_sub_question:
      generate_sub_questions(args.generate_model, args.puzzle_count, output_sub_questions_file)

    # 2. generate the baseline output based on the sub questions.
    baseline_sub_questions(args.model_id, args.output_root, output_sub_questions_file)

    if args.generate_caption:
      generate_image_captions(args.generate_model, args.puzzle_count, output_captions_file)

    # 3. generate the baseline output based on the image captions.
    baseline_captions(args.model_id, args.output_root, output_captions_file, args.use_blank_image)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A tool for generate subquestion for smart101 dataset.")
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
       "--generate-sub-question",
       action="store_true",
       help="Whehter generate sub question data."
    )
    parser.add_argument(
       "--generate-caption",
       action="store_true",
       help="Whehter generate image caption data."
    )
    parser.add_argument(
       "--use-blank-image",
       action="store_true",
       help="Whehter use blank image when reasoning based on gerated image caption data."
    )
    parser.add_argument(
        "--smart-info-v2-csv",
        type=str,
        default="./dataset/SMART_info_v2.csv",
        help="The smart101 information csv file (include puzzle difficulty and puzzle type).",
    )
    parser.add_argument(
        "--generate-model",
        type=str,
        default="gpt-4o",
        help="The model used to generate sub questions and captions.",
    )

    args = parser.parse_args()

    print(args)

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

    smart_info_v2_csv = args.smart_info_v2_csv
    if not os.path.exists(smart_info_v2_csv):
      raise ValueError("input smart csv file path {} not exist.".format(smart_info_v2_csv))

    # load env from .env file
    load_dotenv()

    main(args)
