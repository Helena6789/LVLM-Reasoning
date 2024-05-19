#!/usr/bin/env python3

import argparse
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt

def read_output_csv(output_csv_file):
    records = []
    with open(output_csv_file, newline="") as csvfile:
        datareader = csv.DictReader(csvfile)
        for row in datareader:
            records.append(row)
    return records

# get total average accuracy
def get_accuracy(df):
    filtered_df = df[df['true_answer'] == df['predit_answer']]
    acc = filtered_df.size / df.size * 100
    return acc

# # plot accuracy by different groups like type, difficulty, puzzle_id.
def plot_accuracy(output_csv_file, smart_info_csv_file, puzzle_max, output_figure_path, puzzle_subset_size):
    df = pd.read_csv(output_csv_file)
    df_smart_info_csv = pd.read_csv(smart_info_csv_file)
    difficulty_list = df_smart_info_csv['difficulty'].to_list()
    type_list = df_smart_info_csv['type'].to_list()

    # add two columns to the output csv
    df['difficulty'] = df['puzzle_id'].apply(lambda x : difficulty_list[x-1])
    df['type'] = df['puzzle_id'].apply(lambda x : type_list[x-1])

    img_suffix = "all"
    # plot by difficulty
    plot_accuracy_by_group(df, difficulty_list, 'difficulty', 10, 
                           output_figure_path, puzzle_subset_size, img_suffix, True)
    # plot by type
    plot_accuracy_by_group(df, type_list, 'type', 20, 
                           output_figure_path, puzzle_subset_size, img_suffix, True)
    # plot by puzzle_id
    plot_accuracy_by_group(df, range(1, puzzle_max + 1), 'puzzle_id', 32, 
                           output_figure_path, puzzle_subset_size, img_suffix)

    # get the empty image
    puzzle_id_with_empty_image = df_smart_info_csv[df_smart_info_csv['image'].isnull()]['id'].to_list()
    df_without_empty_image = df[~df['puzzle_id'].isin(puzzle_id_with_empty_image)]
    
    # plot accuracy without empty images.
    img_suffix = "excl_blank_img"
    # plot by difficulty
    plot_accuracy_by_group(df_without_empty_image, difficulty_list, 'difficulty', 10, 
                           output_figure_path, puzzle_subset_size, img_suffix, True)
    # plot by type
    plot_accuracy_by_group(df_without_empty_image, type_list, 'type', 20,
                           output_figure_path, puzzle_subset_size, img_suffix, True)
    # plot by puzzle_id
    plot_accuracy_by_group(df_without_empty_image, range(1, puzzle_max + 1), 'puzzle_id', 32,
                           output_figure_path, puzzle_subset_size, img_suffix)

# plot accuracy by different group
# - group_vals: the values in the group to plot
# - group_name: filter out records group by the group_name key.
def plot_accuracy_by_group(df, group_vals, 
                           group_name, 
                           figure_width, 
                           output_figure_path, 
                           puzzle_subset_size,
                           img_suffix, 
                           add_text_value=False):
    accuracy = []
    group_vals = sorted(list(set(group_vals)))
    for v in group_vals:
        # filter out all the records by the filter_column equals v
        total_df = df[df[group_name] == v]
        if len(total_df) == 0:
            accuracy.append(0)
        else:
            correct_df = total_df[total_df['true_answer'] == total_df['predit_answer']]
            accuracy.append(len(correct_df) / len(total_df) * 100)

    group_vals = [str(v) for v in group_vals]
    # Create a bar chart
    fig = plt.figure(figsize=(figure_width, 5))
    ax = fig.add_subplot(111)
    ax.set_xticks(range(0, len(group_vals)), labels=group_vals)
    plt.xticks(fontsize=8)
    # add text value to graph
    if add_text_value:
        for i, v in enumerate(accuracy):
            plt.text(i, v, '{:.2f}'.format(v), ha = 'center')
    plt.bar(group_vals, accuracy)
    plt.xlabel(group_name)
    plt.ylabel('Accuracy (%)')
    plt.title('Llava grouped by {} (avg. acc= {:.2f}%)'.format(group_name, get_accuracy(df)))
    output_figure = "{}/accuracy_group_by_{}_{}_{}.png".format(output_figure_path, group_name, img_suffix, puzzle_subset_size)
    print("Save plot to: {}".format(output_figure))
    plt.savefig(output_figure)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A tool for Llava baseline.")
    parser.add_argument(
        "--puzzle-max", 
        default=101, 
        type=int,
        help="The subfolder of the puzzle to process from 1 to max value (101)."
    )
    parser.add_argument(
        "--output-csv-file",
        type=str,
        default="output/output_100.csv",
        help="The input parameter of plot_accuracy function: output csv file.",
    )
    parser.add_argument(
        "--smart-info-v2-csv",
        type=str,
        default="/home/hq/LVLM/LVLM-Reasoning/dataset/SMART_info_v2.csv",
        help="The smart101 information csv file (include puzzle difficulty and puzzle type).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="output",
        help="The output folder to save all the figures.",
    )
    parser.add_argument(
        "--puzzle-subset-size", 
        default=100, 
        type=int,
        help="The size of subset puzzle to process for each puzzle folder, max value(2000)."
    )
    args = parser.parse_args()
    print(args)

    
    puzzle_subfolder_max = args.puzzle_max 
    if puzzle_subfolder_max < 1 or puzzle_subfolder_max > 101:
      raise ValueError("puzzle max can only range from 1 to 101.")
    
    output_csv_file = args.output_csv_file
    if not os.path.exists(output_csv_file):
      raise ValueError("csv file path {} not exist.".format(output_csv_file))

    smart_info_v2_csv = args.smart_info_v2_csv
    if not os.path.exists(smart_info_v2_csv):
      raise ValueError("smart csv file path {} not exist.".format(smart_info_v2_csv))

    output_root = args.output_root
    if not os.path.exists(output_root):
      os.makedirs(output_root)

    puzzle_subset_size = args.puzzle_subset_size 
    if puzzle_subset_size < 0 or puzzle_subset_size > 2000:
      raise ValueError("puzzle subset size can only range from 1 to 2000.")

    plot_accuracy(args.output_csv_file,
                  args.smart_info_v2_csv,
                  args.puzzle_max,
                  args.output_root,
                  args.puzzle_subset_size)

