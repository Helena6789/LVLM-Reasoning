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

def output_puzzle_counts(data_frame):
    #pd.set_option('display.max_rows', 150)
    sub_puzzle_count_df = data_frame.groupby('puzzle_id')['sub_puzzle_id'].nunique().reset_index().rename(columns={"sub_puzzle_id": "Q"})
    sub_branch_df = data_frame.groupby('puzzle_id')['sub_question_id'].nunique().reset_index().rename(columns={"sub_question_id": "B"})
    return sub_puzzle_count_df.merge(sub_branch_df)
    
def process_filter_df(output_csv_path):
    frames = []
    for input_csv_file in glob.glob(os.path.join(output_csv_path, 'generated_tot_sub_questions_data*.csv')):
        logger.info("loading file:{}".format(input_csv_file))
        df = pd.read_csv(input_csv_file)
        frames.append(df)
    cmb_df = pd.concat(frames)
    count_df_total = output_puzzle_counts(cmb_df)
    cmb_df.to_csv(os.path.join(output_csv_path, "generated_tot_sub_questions_filter_data_all.csv"), index=False)

    # add NO_FINAL_ANSWER records for all the puzzle without sub question
    last_records = cmb_df.groupby('sub_question_id').tail(1)
    last_records_without_answers = last_records[~last_records['sub_question'].isna()]
    last_records_without_answers.loc[:, 'sub_question'] = np.nan
    last_records_without_answers.loc[:, 'answer'] = 'NO_FINAL_ANSWER'
    last_records_without_answers.loc[:, 'level'] =  np.nan
    cmd_df_all = pd.concat([cmb_df, last_records_without_answers]).sort_index(kind='stable').reset_index(drop=True)
    cmd_df_all.to_csv(os.path.join(output_csv_path, "generated_tot_sub_questions_filter_data_all_with_answer.csv"), index=False)

    # filter branch with final answers
    sub_question_ids = cmb_df[cmb_df['sub_question'].isna()]['sub_question_id'].to_list() 
    filtered_df = cmb_df[cmb_df['sub_question_id'].isin(sub_question_ids)]
    count_df_final_anwser = output_puzzle_counts(filtered_df)
    filtered_df.to_csv(os.path.join(output_csv_path, "generated_tot_sub_questions_filter_data_with_final_answer.csv"), index=False)
    
    # output filter with correct answer only
    correct_answer_sub_question_ids = filtered_df[(filtered_df['true_answer'] == filtered_df['answer']) & filtered_df['sub_question'].isna()]['sub_question_id'].to_list()
    correct_filtered_df = filtered_df[filtered_df['sub_question_id'].isin(correct_answer_sub_question_ids)]
    count_df_correct_answer = output_puzzle_counts(correct_filtered_df)
    correct_filtered_df.to_csv(os.path.join(output_csv_path, "generated_tot_sub_questions_filter_data_with_correct_answer.csv"), index=False)
       
    # output status
    count_merge_all = pd.merge(count_df_total, count_df_final_anwser, on='puzzle_id', suffixes=('_T', '_F'))
    count_merge_all = pd.merge(count_merge_all, count_df_correct_answer, on='puzzle_id').rename(columns={'Q' : 'Q_C', 'B' : 'B_C'})
    count_merge_all['Q_F/Q_T(%)'] = (count_merge_all['Q_F'] /count_merge_all['Q_T']).round(2)
    count_merge_all['Q_C/Q_F(%)'] = (count_merge_all['Q_C'] /count_merge_all['Q_F']).round(2)
    count_merge_all['B_F/B_T(%)'] = (count_merge_all['B_F'] /count_merge_all['B_T']).round(2)
    count_merge_all['B_C/B_F(%)'] = (count_merge_all['B_C'] /count_merge_all['B_F']).round(2)
    print(count_merge_all[['puzzle_id', 'Q_T', 'Q_F', 'Q_C', 'Q_F/Q_T(%)', 'Q_C/Q_F(%)', 'B_T', 'B_F', 'B_C', 'B_F/B_T(%)', 'B_C/B_F(%)']])

    # output correct puzzle stats
    logger.info("-------puzzle stats-----------")
    total_puzzle_with_final_answer = len(set(filtered_df[filtered_df['sub_question'].isna()]['puzzle_id'].tolist()))
    total_puzzle_with_correct_answer = len(set(correct_filtered_df['puzzle_id'].tolist()))
    logger.info("Total puzzle_id with final answer - sum(P_T): {}".format(total_puzzle_with_final_answer))
    logger.info("Total puzzle_id with correct answer - sum(P_C): {}".format(total_puzzle_with_correct_answer))
    logger.info("sum(P_C)/sum(P_T): {:.2f}%".format(total_puzzle_with_correct_answer/total_puzzle_with_final_answer*100))

    # output correct sub_puzzle stats
    logger.info("-------sub_puzzle stats-----------")
    sub_puzzle_ids = filtered_df[filtered_df['sub_question'].isna()].drop_duplicates(['puzzle_id', 'sub_puzzle_id'])['sub_puzzle_id'].tolist()
    total_sub_puzzle_with_final_answer = len(sub_puzzle_ids)
    correct_sub_puzzle_ids = correct_filtered_df[correct_filtered_df['sub_question'].isna()].drop_duplicates(['puzzle_id', 'sub_puzzle_id'])['sub_puzzle_id'].tolist()
    total_sub_puzzle_with_correct_answer = len(correct_sub_puzzle_ids)
    logger.info("Total sub_puzzle with final answer - sum(Q_T): {}".format(total_sub_puzzle_with_final_answer))
    logger.info("Total sub_puzzle with correct answer - sum(Q_C): {}".format(total_sub_puzzle_with_correct_answer))
    logger.info("sum(Q_C)/sum(Q_T): {:.2f}%".format(total_sub_puzzle_with_correct_answer/total_sub_puzzle_with_final_answer*100))

    # output correct branch stats
    logger.info("-------branch stats-----------")
    total_branch_with_final_answer = len(sub_question_ids)
    total_branch_with_correct_answer = len(correct_answer_sub_question_ids)
    logger.info("Total branch with final answer - sum(B_F): {}".format(total_branch_with_final_answer))
    logger.info("Total branch with correct answer - sum(B_C): {}".format(total_branch_with_correct_answer))
    logger.info("sum(B_C)/sum(B_F): {:.2f}%".format(total_branch_with_correct_answer/total_branch_with_final_answer*100))
    logger.info("\n")
    
def get_accuracy(df, accuracy_type):
    if accuracy_type == "branch":
        column_key = "sub_question_id"
    elif accuracy_type == "sub_puzzle":
        column_key = "sub_puzzle_id"
    else:
        raise "Unknown accuracy type:{}".format(accuracy_type)

    # filter with final answers
    if accuracy_type  == "branch":
        sub_ids = df[df['sub_question'].isna()][column_key].to_list()
    else:
        sub_ids = df[df['sub_question'].isna()].drop_duplicates(['puzzle_id', 'sub_puzzle_id'])[column_key].to_list()

    filtered_df = df[df[column_key].isin(sub_ids)]
    # filter with correct answer only
    correct_answer_df = filtered_df[(filtered_df['true_answer'] == filtered_df['answer']) & filtered_df['sub_question'].isna()]

    if accuracy_type == 'branch':
        correct_answer_sub_ids = correct_answer_df[column_key].to_list()
    else:
        correct_answer_sub_ids = correct_answer_df.drop_duplicates(['puzzle_id', 'sub_puzzle_id'])[column_key].to_list()
       
    # output status
    total_sub_ids = len(sub_ids)
    total_sub_ids_with_correct_answer = len(correct_answer_sub_ids)
    return [total_sub_ids_with_correct_answer/total_sub_ids*100, total_sub_ids_with_correct_answer, total_sub_ids]

# # plot accuracy by different groups like type, difficulty, puzzle_id.
def plot_accuracy(generated_data_csv_file, smart_info_csv_file, output_figure_path, accuracy_type):
    if not os.path.exists(output_figure_path):
      os.makedirs(output_figure_path)
    
    df = pd.read_csv(generated_data_csv_file)
    df_smart_info_csv = pd.read_csv(smart_info_csv_file)
    difficulty_list = df_smart_info_csv['difficulty'].to_list()
    type_list = df_smart_info_csv['type'].to_list()

    # add two columns to the output csv
    df['difficulty'] = df['puzzle_id'].apply(lambda x : difficulty_list[x-1])
    df['type'] = df['puzzle_id'].apply(lambda x : type_list[x-1])

    # plot by difficulty
    plot_accuracy_by_group(df, difficulty_list, 'difficulty', 10, 
                           output_figure_path, accuracy_type, True)
    # plot by type
    plot_accuracy_by_group(df, type_list, 'type', 20, 
                           output_figure_path, accuracy_type, True)
    # plot by puzzle_id
    plot_accuracy_by_group(df, df['puzzle_id'].to_list(), 'puzzle_id', 32, 
                           output_figure_path, accuracy_type, True)

# plot accuracy by different group
# - group_vals: the values in the group to plot
# - group_name: filter out records group by the group_name key.
def plot_accuracy_by_group(df, group_vals, 
                           group_name, 
                           figure_width, 
                           output_figure_path,
                           accuracy_type,
                           add_text_value=False):
    accuracy = []
    accuracy_total = []
    accuracy_correct = []
    group_vals = sorted(list(set(group_vals)))
    for v in group_vals:
        # filter out all the records by the filter_column equals v
        total_df = df[df[group_name] == v]
        if len(total_df) == 0:
            accuracy.append(0)
            accuracy_correct.append(0)
            accuracy_total.append(0)
        else:
            accurary_data = get_accuracy(total_df, accuracy_type)
            accuracy.append(accurary_data[0])
            accuracy_correct.append(accurary_data[1])
            accuracy_total.append(accurary_data[2])

    group_vals = [str(v) for v in group_vals]
    # Create a bar chart
    fig = plt.figure(figsize=(figure_width, 5))
    ax = fig.add_subplot(111)
    ax.set_xticks(range(0, len(group_vals)), labels=group_vals)
    plt.xticks(fontsize=8)
    # add text value to graph
    if add_text_value:
        for i, v in enumerate(accuracy):
            if v == 0:
                plt.text(i, v, '{:.2f}%'.format(v), ha = 'center')
            else:
                plt.text(i, v, '{}/{}={:.2f}%'.format(accuracy_correct[i], accuracy_total[i], v), ha = 'center')
    plt.bar(group_vals, accuracy)
    plt.xlabel(group_name)
    plt.ylabel('Accuracy (%)')
    plt.title('GPT-4o grouped by {} (avg. acc= {:.2f}%)'.format(group_name, get_accuracy(df, accuracy_type)[0]))
    output_figure = "{}/accuracy_group_by_{}_{}.png".format(output_figure_path, group_name, accuracy_type)
    logger.info("Save plot to: {}".format(output_figure))
    plt.savefig(output_figure)
    plt.close()

def main(args):    
    process_filter_df(args.output_root)
    # plot 4o data accuracy
    plot_accuracy(os.path.join(args.output_root, "generated_tot_sub_questions_filter_data_all.csv"),
                  args.smart_info_v2_csv,
                  os.path.join(args.output_root, "plot_output"),
                  args.accuracy_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A tool for generate tot process subquestion.")
    parser.add_argument(
        "--output-root",
        type=str,
        default="./output/instance_split",
        help="The output folder to save all the results.",
    )
    parser.add_argument(
        "--accuracy-type",
        type=str,
        default="branch",
        help="Output plot accuracy type: either branch or sub_puzzel.",
    )
    parser.add_argument(
        "--smart-info-v2-csv",
        type=str,
        default="./dataset/SMART101-release-v1/SMART101-Data/SMART_info_v2.csv",
        help="The smart101 information csv file (include puzzle difficulty and puzzle type).",
    )
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')

    args = parser.parse_args()

    smart_info_v2_csv = args.smart_info_v2_csv
    if not os.path.exists(smart_info_v2_csv):
      raise ValueError("input smart csv file path {} not exist.".format(smart_info_v2_csv))

    # setup logging
    logging_level = logging.INFO
    if args.debug:
      logging_level = logging.DEBUG

    logging.basicConfig(
      level=logging_level,
      format="%(asctime)s [%(levelname)s] %(message)s",
      handlers=[
          RotatingFileHandler(os.path.join(args.output_root,"tot_data_stats-{}.log".format(time.strftime("%Y-%m-%d-%H-%M-%S"))),
                                      maxBytes=1024 * 1024 * 10),
          logging.StreamHandler()
      ]
    )
    logger.info(args)

    main(args)