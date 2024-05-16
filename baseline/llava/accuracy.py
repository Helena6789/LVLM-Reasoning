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

def get_accuracy(df):
    filtered_df = df[df['true_answer'] == df['predit_answer']]
    acc = filtered_df.size / df.size * 100
    return acc

def get_group_puzzle_accuracy(output_file, puzzle_max, output_figure_path):
    df = pd.read_csv(output_file)

    avg_acc = get_accuracy(df)

    puzzle_ids = range(1, puzzle_max + 1)
    accuracy = []
    for id in puzzle_ids:
        total_df = df[df['puzzle_id'] == id]
        if total_df.size == 0:
            accuracy.append(0)
        else:
            correct_df = total_df[total_df['true_answer'] == total_df['predit_answer']]
            accuracy.append(correct_df.size / total_df.size * 100)

    # Create a bar chart
    fig = plt.figure(figsize=(32, 2))
    ax = fig.add_subplot(111)
    ax.set_xticks(puzzle_ids)
    plt.xticks(fontsize=8)
    plt.bar(puzzle_ids, accuracy)
    plt.xlabel('puzzle id')
    plt.ylabel('Accuracy')
    plt.title('Llava grouped by puzzle_id (avg. acc= {:.2f}%)'.format(avg_acc))
    plt.savefig(
        "%s/accuracy_group.png" % (output_figure_path)
    )
    plt.close()