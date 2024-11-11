#!/usr/bin/env python3

import os
import argparse
import json
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.cm as cm
import numpy as np
import time
import logging
import glob
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)

def generate_accuracy_plot(stages_accuracy_list, output_path, puzzle_id, model_type, step):

    fig, axs = plt.subplots(figsize=(15, 6))

    accuracy_list = stages_accuracy_list[0]
    labels = []
    values = []
    categories = []
    colors = []

    indices = np.linspace(0, 1, 10)
    cmap = cm.get_cmap('viridis')
    colors_stages = cmap(indices)

    for entry in accuracy_list:
        for key, value in entry.items():
            first_value, second_value, ratio = value
            labels.append(f"{first_value} / {second_value} = {ratio:.2f}")
            values.append(ratio)
            
            if key.startswith('pretrain-inference'):
                colors.append('tab:blue')
                categories.append('PI')
            elif key.startswith('pretrain-cot'):
                colors.append('tab:orange')
                categories.append('PC')
            elif key.startswith('finetune-inference'):
                colors.append('tab:green')
                categories.append('FI')
            elif key.startswith('finetune-cot'):
                colors.append('tab:cyan')
                categories.append('FC')
            elif key.startswith('implicit-cot'):
                cidx = int(key[-1])
                colors.append(colors_stages[cidx])
                categories.append('IC_S{}'.format(cidx))

    num_catagory = len(categories)
    # Plotting
    axs.bar(np.arange(num_catagory), values, color=colors)
    axs.set_ylabel("Accuracy")
    axs.set_xticklabels([])
    # axs[i].set_title('{}'.format(key), fontsize=20)
    axs.set_ylim([0, 1])

    # Annotate bars with the custom labels
    for bar, label in zip(axs.patches, labels):
        axs.text(
            bar.get_x() + bar.get_width() / 2,  # X position of text
            bar.get_height(),                   # Y position of text
            label,                              # Text to display
            ha='center', va='bottom',            # Center alignment
            fontsize=8
        )

    fig.suptitle("{} accuracy baseline for puzzle {}".format(model_type, puzzle_id),fontsize=12, weight='bold')
    # Add the legend below the suptitle, above the subplots
    legend_handles = [Patch(color=colors[i], label=categories[i]) for i in range(num_catagory)]
    fig.legend(handles=legend_handles, loc='upper center', ncol=num_catagory, bbox_to_anchor=(0.5, 0.95), fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # Save the plot to a file
    plot_output = os.path.join(output_path, 'accuracy_{}_{}_shift_{}.png'.format(model_type, puzzle_id, step))
    logger.info('save plot to {}'.format(plot_output))
    plt.savefig(plot_output)  


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A tool for generate infer accuracy.")
    parser.add_argument(
        "--infer-results-path",
        type=str,
        help="The infer resulst folder.",
    )
    parser.add_argument(
        "--puzzle-id",
        required=True,
        type=int,
        help="The puzzle to process from 1 to max value (101)."
    )
    parser.add_argument(
        "--infer-types",
        required=True,
        type=str,
        help="The inference types: 'pretrain-inference', 'pretrain-cot', 'finetune-inference', 'finetune-cot', 'implicit-cot'.",
    )
    parser.add_argument(
        "--model-type",
        required=True,
        type=str,
        help="The model type to inference.",
    )
    parser.add_argument(
        "--step",
        required=True,
        default=1,
        type=int,
        help="The implicit cot finetune jump step."
    )
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    infer_results_path = args.infer_results_path
    if not os.path.exists(infer_results_path):
      raise ValueError("input infer_results folder {} not exist.".format(infer_results_path))

    output_path = os.path.join(infer_results_path, "plots_output")
    if not os.path.exists(output_path):
      os.makedirs(output_path, exist_ok=True)
    
    infer_types = args.infer_types.split(',')
    for infer_type in infer_types:
        if infer_type not in ['pretrain-inference', 'pretrain-cot', 'finetune-inference', 'finetune-cot', 'implicit-cot']:
            raise ValueError("unknown infer_type {}. Valid values are pretrain-inference, pretrain-cot, finetune-inference, finetune-cot, implicit-cot.".format(infer_type))

    # setup logging
    logging_level = logging.INFO
    if args.debug:
      logging_level = logging.DEBUG

    logging.basicConfig(
      level=logging_level,
      format="%(asctime)s [%(levelname)s] %(message)s",
      handlers=[
          RotatingFileHandler(os.path.join(output_path, "plots_{}.log".format(time.strftime("%Y-%m-%d-%H-%M-%S"))),
                                      maxBytes=1024 * 1024 * 10),
          logging.StreamHandler()
      ]
    )

    logger.info(args)

    accuracy_list = []
    start_stage = 0
    max_stage = 9
    for infer_type in infer_types:
        result_folder = os.path.join(infer_results_path, infer_type)
        if not os.path.exists(result_folder):
            print('skip process infer_type:{}, no results found.'.format(infer_type))
            continue
        
        file_suffix = infer_type.replace('-', '_')
        accuracy_json_file_to_process = []
        if infer_type == 'implicit-cot':
            for stage in range(start_stage, max_stage + 1):
                json_file_pattern = os.path.join(result_folder, '*_shift_{}_stage_{}_{}.jsonl'.format(args.step, stage, file_suffix))
                accuracy_json_file_list = glob.glob(json_file_pattern)
                if len(accuracy_json_file_list) == 0:
                    logger.info('no file found for pattern:{}, skip.'.format(json_file_pattern))
                    continue
                logger.debug('infer type {}, stage {}: all founded infer results :{}'.format(infer_type, stage, accuracy_json_file_list))
                accuracy_json_file = sorted(accuracy_json_file_list)[-1]
                logger.debug('infer type {}, stage {}: using accuracy_json_file:{}'.format(infer_type, stage, accuracy_json_file))
                accuracy_json_file_to_process.append(('{}_{}'.format(infer_type, stage), accuracy_json_file))
                stage += 1
        else:
            json_file_pattern = os.path.join(result_folder, '*_{}.jsonl'.format(file_suffix))
            accuracy_json_file_list = glob.glob(json_file_pattern)
            if len(accuracy_json_file_list) == 0:
                    logger.info('no file found for pattern:{}, skip.'.format(json_file_pattern))
                    continue
            logger.debug('infer type {}: all founded infer results :{}'.format(infer_type, accuracy_json_file_list))
            accuracy_json_file = sorted(accuracy_json_file_list)[-1]
            logger.debug('infer type {}: using accuracy_json_file:{}'.format(infer_type, accuracy_json_file))
            accuracy_json_file_to_process.append((infer_type, accuracy_json_file))

        for file_type, accuracy_json_file in accuracy_json_file_to_process:
            logger.info("processing file:{}".format(accuracy_json_file))
            df = pd.read_json(path_or_buf=accuracy_json_file, lines=True)
            correct = df[df['response']==df['label']]
            ratio = len(correct)/ len(df)
            logger.info('{}, accuracy: {}/{}={:.4f}'.format(file_type, len(correct), len(df), ratio))
            accuracy_list.append({file_type : [len(correct),  len(df), ratio]})

    # plots
    logger.debug(accuracy_list)
    generate_accuracy_plot([accuracy_list], output_path, args.puzzle_id, args.model_type, args.step)
            
                
