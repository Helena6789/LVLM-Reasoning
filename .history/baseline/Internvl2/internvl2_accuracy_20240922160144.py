#计算accuracy
import csv

csv_file_path = '/home/yyf/data/llava_data/finetune_dataset/internvl2_sft_preidct_18.csv'


total_count = 0
correct_count = 0
with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
      csvreader = csv.reader(csvfile)

      next(csvreader)
      for row in csvreader:

                  true_answer = row[3].strip()
                  predit_answer = row[4].strip() 
                  
                  if true_answer == predit_answer:
                        correct_count += 1
                  total_count += 1

# 计算准确率
accuracy = correct_count / total_count if total_count > 0 else 0

print(f"Total predictions: {total_count}")
print(f"Correct predictions: {correct_count}")
print(f"Accuracy: {accuracy:.2%}")
print("")
