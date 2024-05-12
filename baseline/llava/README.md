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