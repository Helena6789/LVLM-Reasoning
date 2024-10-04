## Llava baseline
This is the tool to run baseline for [Llava model](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf) for dataset [SMART101](https://github.com/merlresearch/SMART).

### Dataset
Download and extract the dataset.
```
    wget https://zenodo.org/record/7775984/files/SMART101-release-v1.zip?download=1 -P dataset
    cd dataset
    unzip SMART101-release-v1.zip -d . >/dev/null
```

### Run baseline

```
python baseline.py --puzzle-max 1 --subset-size 1 --seed 123
```

* `--puzzle-max`: The subfolder of the puzzle to process from 1 to max value (101).
* `--subset-size`: The size of subset puzzle to process for each puzzle folder, max value(2000).

## Generate TOT data

To generate dataset using tree of thought(TOT) by the following command:

```
 python tot_data_generator.py --subset-size 2000 --seed 123 --data-root <your-smart101-original-dataset> --output-root <your-output-path> --max-depth 10 --split-type instance
```

An example script can be found at `baseline/llava/scripts/run_tot_data_generator.sh`.

* Provide the original smart101 dataset path and the output path.
* For the tot `--mat-depth`, we choose as `10`. 
* Instance split type including puzzle id: (18, order), (61, algebra), (62, logic), (69, spatial), (71, math), (73, path), (77, pattern), (94, measure), (99, counting).
* The generate data will save as csv file look like: `generated_tot_sub_questions_data_18.csv`.

A sample output will be like following:

```
puzzle_id,sub_puzzle_id,image_path,origin_question,true_answer,sub_question,answer,level,sub_question_id
18,1343,./dataset/SMART101-release-v1/SMART101-Data/18/img/puzzle_18_e_1343.png,"On a table, five polygon cards are stacked as shown, and each card is numbered according to the color shown at the bottom. The cards are removed one by one from the top of the stack. What is the possible order of the cards being removed? A: 4-5-1-2-3,B: 5-4-1-3-2,C: 1-2-4-3-5,D: 3-5-1-2-4,E: 1-5-2-4-3",D,What is the number of the top card?,5,0.0,18-1343-1
18,1343,./dataset/SMART101-release-v1/SMART101-Data/18/img/puzzle_18_e_1343.png,"On a table, five polygon cards are stacked as shown, and each card is numbered according to the color shown at the bottom. The cards are removed one by one from the top of the stack. What is the possible order of the cards being removed? A: 4-5-1-2-3,B: 5-4-1-3-2,C: 1-2-4-3-5,D: 3-5-1-2-4,E: 1-5-2-4-3",D,What is the number of the second card from the top?,4,1.0,18-1343-1
18,1343,./dataset/SMART101-release-v1/SMART101-Data/18/img/puzzle_18_e_1343.png,"On a table, five polygon cards are stacked as shown, and each card is numbered according to the color shown at the bottom. The cards are removed one by one from the top of the stack. What is the possible order of the cards being removed? A: 4-5-1-2-3,B: 5-4-1-3-2,C: 1-2-4-3-5,D: 3-5-1-2-4,E: 1-5-2-4-3",D,What is the number of the third card from the top?,1,2.0,18-1343-1
18,1343,./dataset/SMART101-release-v1/SMART101-Data/18/img/puzzle_18_e_1343.png,"On a table, five polygon cards are stacked as shown, and each card is numbered according to the color shown at the bottom. The cards are removed one by one from the top of the stack. What is the possible order of the cards being removed? A: 4-5-1-2-3,B: 5-4-1-3-2,C: 1-2-4-3-5,D: 3-5-1-2-4,E: 1-5-2-4-3",D,What is the number of the fourth card from the top?,3,3.0,18-1343-1
18,1343,./dataset/SMART101-release-v1/SMART101-Data/18/img/puzzle_18_e_1343.png,"On a table, five polygon cards are stacked as shown, and each card is numbered according to the color shown at the bottom. The cards are removed one by one from the top of the stack. What is the possible order of the cards being removed? A: 4-5-1-2-3,B: 5-4-1-3-2,C: 1-2-4-3-5,D: 3-5-1-2-4,E: 1-5-2-4-3",D,What is the number of the fifth card from the top?,2,4.0,18-1343-1
18,1343,./dataset/SMART101-release-v1/SMART101-Data/18/img/puzzle_18_e_1343.png,"On a table, five polygon cards are stacked as shown, and each card is numbered according to the color shown at the bottom. The cards are removed one by one from the top of the stack. What is the possible order of the cards being removed? A: 4-5-1-2-3,B: 5-4-1-3-2,C: 1-2-4-3-5,D: 3-5-1-2-4,E: 1-5-2-4-3",D,,B,,18-1343-1
```
* `sub_question_id`: means <puzzle_id>-<sub_question_id>-<branch_id>.
* `level`: means the tot branch depth.
* The last record of every branch will be the original question and its corresponding final answer.


### Generate TOT data stats 

To get the train and test dataset using the following command:

```
python tot_data_stats.py --output-root [TOT_DATA_PATH]
```

This processor will output the statistics of the gpt-4o generated sub questions output by previous step. 
It will also generate accuracy plots for the generated data. 

```
 puzzle_id   Q_T   Q_F  Q_C  Q_F/Q_T(%)  Q_C/Q_F(%)   B_T   B_F  B_C  B_F/B_T(%)  B_C/B_F(%)
0         18  2000  1683  542        0.84        0.32  6000  3131  683        0.52        0.22

-------puzzle stats-----------
Total puzzle_id with final answer - sum(P_T): 1
Total puzzle_id with correct answer - sum(P_C): 1
sum(P_C)/sum(P_T): 100.00%

-------sub_puzzle stats-----------
Total sub_puzzle with final answer - sum(Q_T): 1683
Total sub_puzzle with correct answer - sum(Q_C): 542
sum(Q_C)/sum(Q_T): 32.20%

-------branch stats-----------
Total branch with final answer - sum(B_F): 3131
Total branch with correct answer - sum(B_C): 683
sum(B_C)/sum(B_F): 21.81%


Save plot to: ./output/llava-v1.6-mistral-7b_gpt4o_tot_generate_data_by_question_decomposition_instance_split_2000/18/plot_output/accuracy_group_by_difficulty_branch.png
Save plot to: ./output/llava-v1.6-mistral-7b_gpt4o_tot_generate_data_by_question_decomposition_instance_split_2000/18/plot_output/accuracy_group_by_type_branch.png
Save plot to: ./output/llava-v1.6-mistral-7b_gpt4o_tot_generate_data_by_question_decomposition_instance_split_2000/18/plot_output/accuracy_group_by_puzzle_id_branch.png
```
* `Q_T`: the total number of sub questions with provided puzzle_id. i.e. for each puzzle id, it has 2000 sub questions. 
* `Q_F`: the total number of sub questions that can generate the final answer when we generated sub question data using tot. Not all subquestions can generate the final answer, some of them cannot generate the final answer. Note: We have 3 branches for each subquestion, if any branch can generate the final answer, we count it as `Q_F`.  
* `Q_C`: the total number of sub questions that can generate the correct final answer when generated sub suqestion data using tot.  Note: We have 3 branches for each subquestion, if any branch can generate the correct final answer, we count it as `Q_C`.  
* `B_T`: the total number of branches with provided puzzle_id. i.e. for each puzzle id, it has 2000 sub questions and each sub question has 3 branches, the total number of branches is 6000.  
* `B_F`: the total number of branches that can generate the final answer when we generated sub question data using tot. Not all branches can generate the final answer, some of them cannot generate the final answer.
* `B_C`: the total number of branches that can generate the correct final answer when generated sub suqestion data using tot.

## Generate finetune dataset

```
python finetune_data_generator --output-root  [OUTPUT_SAVE_PATH] --input-data [COMBINATION_DATA] --smart101-data-root [SMART101_DATA_PATH] --clip-image-root [PATH_SAVE_CLIP_IMAGES] --puzzle-id [PUZZLE_ID] --seed 42
```

An example script can be found at `baseline/llava/scripts/run_finetune_data_generator.sh`.

We split the data as `train`, `validation` and `test` data as 80-5-15 for original finetune, but the train and validation dataset will merge into one train file as `puzzle_<puzzle_id>_train_<stage>.json` for implicit-cot. We will give the ratio of validation set when finetune, its around 0.59 = validation/(train+validation).

> Note: We removed all the records when the sub answer of the sub questions are mssing or NA. 

In order to finetune llava using [ms-swift](https://github.com/modelscope/ms-swift), the train dataset will generate as the following format, see details in [dataset customization](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/LLM/Customization.md).

```json
[
    {
        "messages": [{"role": "user", "content": "What is the color corresponding to 1?"}, {"role": "assistant", "content": "Green"}],
        "images": ["./dataset/SMART101-release-v1/SMART101-Data/18/img/puzzle_18_e_1.png"]
    }
]
```

We also generated a data entry file `smart101.json` for ms-swift to finetune.