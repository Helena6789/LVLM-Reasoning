### Llava baseline
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

### Run Tree of Thoughts generate data

* Standard random split
```
 python tot_data_generator.py --puzzle-count 1 --subset-size 1 --seed 123 --data-root [SMART101_DATA_PATH] --output-root output/tot_output --max-depth 10 --split-type standard
```

* Puzzle split
```
 python tot_data_generator.py --subset-size 1 --seed 123 --data-root [SMART101_DATA_PATH] --output-root output/tot_output --max-depth 10 --split-type puzzle
```

* Instance split
```
 python tot_data_generator.py --subset-size 1 --seed 123 --data-root [SMART101_DATA_PATH] --output-root output/tot_output --max-depth 10 --split-type instance
```


### Run Tree of Thoughts generate datastats

```
 python tot_data_generator.py --output-root [TOT_DATA_PATH]
```

### Run generate finetune dataset for different types

```
python finetune_data_generator --output-root  [OUTPUT_SAVE_PATH] --input-data [COMBINATION_DATA] --smart101-data-root [SMART101_DATA_PATH] --clip-image-root [PATH_SAVE_CLIP_IMAGES] --puzzle-id [PUZZLE_ID] --seed 42
```