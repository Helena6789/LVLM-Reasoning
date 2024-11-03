import json
import os
from PIL import Image
import torch
from transformers import BlipProcessor, BlipModel
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
import numpy as np
from typing import List
import torch.nn.functional as F

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Load BLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")

# Define transform with 448x448 input size
input_size = 448
transform = T.Compose([
    T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
    T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
    T.ToTensor(),
    # T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

def calculate_blip_score(image, text, model, processor, device):
    """Calculates the relevance score between an image and a text prompt using BLIP."""
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        output = model(**inputs)
    
    # Extract embeddings and calculate cosine similarity
    image_features = F.normalize(output.image_embeds, p=2, dim=-1)
    text_features = F.normalize(output.text_embeds, p=2, dim=-1)
    similarity = (image_features @ text_features.T).squeeze().item()
    
    return similarity

def add_noise_to_image(image, target_box):
    """Adds noise to all parts of the image except the target subimage region."""
    noisy_image = image.copy()
    noise = Image.effect_noise((image.width, image.height), 10).convert("RGB")  # Create Gaussian noise
    
    # Blend noise with the original image everywhere except the target_box
    mask = Image.new("L", image.size, 128)  # Partial transparency mask
    mask.paste(255, target_box)  # Full opacity in the target region
    noisy_image = Image.composite(noisy_image, noise, mask)
    # noisy_image.save("noisy_image.png")
    
    return noisy_image

def normalize_and_shift_scores(scores, parent_score, target_mean=0, target_max=0.5):
    """Normalizes scores around the parent score with specified bounds."""
    scores = np.array(scores)
    scores = (scores - scores.mean()) / (scores.ptp() if scores.ptp() != 0 else 1)
    scores = scores * target_max + parent_score
    return scores.tolist()

def hierarchical_scoring(image, text, model, processor, device, levels=3):
    """Performs hierarchical scoring with each level of granularity."""
    
    def split_and_score(image, parent_score, grid_size, level):
        """Recursive function to split and score image regions."""
        width, height = image.size
        patch_width = width // grid_size
        patch_height = height // grid_size
        
        # Calculate scores for each patch in the current grid
        scores = []
        boxes = []
        for i in range(2):  # Each level divides into a 2x2 grid
            for j in range(2):
                left = j * patch_width
                upper = i * patch_height
                right = left + patch_width
                lower = upper + patch_height
                target_box = (left, upper, right, lower)
                
                # Store the box for recursive use in the final level
                boxes.append(target_box)
                
                # Apply noise to everything except the target `8x8` area
                modified_image = add_noise_to_image(image, target_box)
                
                # Calculate the relevance score with BLIP
                score = calculate_blip_score(modified_image, text, model, processor, device)
                scores.append(score)
                # print("score", score)
                # input()
        
        # Normalize scores starting from the second level
        if level > 1:
            scores = normalize_and_shift_scores(scores, parent_score, target_mean=0, target_max=0.05 * (4 ** (levels - level)))
        # print("parent_score", parent_score)
        # print("scores", scores)
        # input()
        final_scores = []
        if level < levels:
            # Recurse if we haven't reached the last level, for each quadrant
            for idx, (quadrant_score, box) in enumerate(zip(scores, boxes)):
                subimage = image.crop(box)
                sub_scores = split_and_score(subimage, quadrant_score, grid_size * 2, level + 1)
                final_scores.extend(sub_scores)
        else:
            # Last level, replicate scores directly into `16x16` grid
            for quadrant_score in scores:
                final_scores.extend([quadrant_score] * 2)  # Replicate in `2x2` cells

        return final_scores

    # Start with the entire image and an initial parent score
    initial_score = calculate_blip_score(image, text, model, processor, device)
    return split_and_score(image, initial_score, 2, 1)

def process_entry(entry, model, transform, device):
    """Processes a single entry to add relevance scores for each `8x8`-based subimage."""
    question = entry['messages'][0]['content'].replace("<image>\n", "")
    formatted_text = f"Question: {question} Answer: {entry['messages'][1]['content']}"
    
    image_path = entry['images'][0]
    image = Image.open(image_path)
    image = transform(image)
    image = T.ToPILImage()(image)
    
    scores = hierarchical_scoring(image, formatted_text, model, processor, device)

    scores_8x16 = [scores[i * 16:(i + 1) * 16] for i in range(8)]
    
    # Convert `8x16` scores to `16x16` by duplicating each row
    scores_16x16 = []
    for row in scores_8x16:
        scores_16x16.append(row)  # Original row
        scores_16x16.append(row)  # Duplicate row to make 16x16
    
    # Flatten `16x16` grid to row-major order
    flattened_scores = [score for row in scores_16x16 for score in row]
    
    # print("scores", len(flattened_scores))
    # input() 
    entry['subimage_scores'] = flattened_scores
    return entry

def process_dataset(json_path, output_path, model, transform, device):
    """Processes a JSON dataset file, updating each entry with relevance scores for subimages."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    for entry in data:
        entry = process_entry(entry, model, transform, device)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

def main():
    with open("/scratch/czr/LVLM-Reasoning/dataset/finetune_dataset/smart101.json", 'r') as f:
        smart101_data = json.load(f)
    
    updated_smart101_data = {}

    for entry_name, entry_data in smart101_data.items():
        dataset_path = entry_data["dataset_path"]
        full_dataset_path = os.path.join("/scratch/czr/LVLM-Reasoning/dataset/finetune_dataset", dataset_path)
        new_dataset_path = os.path.join("/scratch/czr/LVLM-Reasoning/dataset/finetune_dataset_subimage", dataset_path)
        
        process_dataset(full_dataset_path, new_dataset_path, model, transform, device)
        
        updated_smart101_data[entry_name] = {
            "dataset_path": dataset_path,
            "tags": entry_data["tags"]
        }
    
    new_smart101_path = "/scratch/czr/LVLM-Reasoning/dataset/finetune_dataset_subimage/smart101.json"
    with open(new_smart101_path, 'w') as f:
        json.dump(updated_smart101_data, f, indent=2)

if __name__ == "__main__":
    main()
