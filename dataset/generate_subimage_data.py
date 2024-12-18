import json
import os
import numpy as np
from tqdm import tqdm

def process_entry(entry, step):
    entry['messages'] = entry['messages'][step*2:]
    return entry

def process_dataset(json_path, output_path, step):
    """Processes a JSON dataset file, updating each entry with relevance scores for subimages."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    for entry in tqdm(data, desc="Processing dataset"):
        entry = process_entry(entry, step)
    
    data = [entry for entry in tqdm(data) if entry.get("messages")]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

def main():
    data_root = './dataset/finetune_dataset_subimage/'
    smart101_path = os.path.join(data_root, "smart101.json")
    with open(smart101_path, 'r') as f:
        smart101_data = json.load(f)
    
    updated_smart101_data = {}

    for idx, (entry_name, entry_data) in tqdm(enumerate(smart101_data.items()), total=len(smart101_data)):
        # only process smart101_implicit_cot_*_train_0 dataset.
        if not entry_name.startswith('smart101_implicit_cot') or ('train_0' not in entry_name):
            continue

        for step in range(1, 10):
            dataset_path = entry_data["dataset_path"]
            new_dataset_path = dataset_path.replace("train_0", "train_{}".format(step))
            full_dataset_path = os.path.join(data_root, dataset_path)
            new_full_dataset_path = os.path.join(data_root, new_dataset_path)
            
            process_dataset(full_dataset_path, new_full_dataset_path, step)
            
            updated_smart101_data[entry_name.replace("train_0", "train_{}".format(step))] = {
                "dataset_path": new_dataset_path,
                "tags": entry_data["tags"]
            }

    smart101_data.update(updated_smart101_data)
    with open(smart101_path, 'w') as f:
        json.dump(smart101_data, f, indent=2)

if __name__ == "__main__":
    main()
