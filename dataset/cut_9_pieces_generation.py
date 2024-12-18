# import os
# import pandas as pd
# from PIL import Image
# import torch
# import clip
# # from transformers import CLIPProcessor, CLIPModel

# # Load the CLIP model and processor
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# # 已有的函数，保持不变
# def calculate_clip_score(image, text, model, preprocess, device):
#     max_tokens = 77
#     words = text.split()
#     truncated_text = " ".join(words[:max_tokens])
#     tokenized_text = clip.tokenize([truncated_text]).to(device)

#     image_input = preprocess(image).unsqueeze(0).to(device)
#     text_input = clip.tokenize([text]).to(device)
    
#     with torch.no_grad():
#         image_features = model.encode_image(image_input)
#         text_features = model.encode_text(text_input)
#         image_features /= image_features.norm(dim=-1, keepdim=True)
#         text_features /= text_features.norm(dim=-1, keepdim=True)
#         similarity = (image_features @ text_features.T).item()
    
#     return similarity

# def process_image(image_path, sub_question, sub_answer, output_dir):
#     device = get_device()
    
#     # Load CLIP model
#     model, preprocess = clip.load("ViT-B/32", device=device)
    
#     # Load image
#     image = Image.open(image_path)
#     width, height = image.size
    
#     # Calculate the size of each grid
#     grid_width = width // 3
#     grid_height = height // 3
    
#     max_score = -float('inf')
#     best_grid_image = None
#     best_grid_index = None
    
#     # Create the output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Iterate over 9 grids
#     for i in range(3):
#         for j in range(3):
#             left = j * grid_width
#             upper = i * grid_height
#             right = left + grid_width
#             lower = upper + grid_height
            
#             grid_image = image.crop((left, upper, right, lower))
#             text = f"{sub_question} {sub_answer}"
#             score = calculate_clip_score(grid_image, text, model, preprocess, device)
            
#             if score > max_score:
#                 max_score = score
#                 best_grid_image = grid_image
#                 best_grid_index = i * 3 + j + 1  # Index of the grid (1 to 9)
    
#     # Define the output file name and path
#     base_name = os.path.splitext(os.path.basename(image_path))[0]
#     output_file_name = f"{base_name}_grid{best_grid_index}.png"
#     output_file_path = os.path.join(output_dir, output_file_name)
    
#     # Save the best grid image if it doesn't already exist
#     if not os.path.exists(output_file_path):
#         best_grid_image.save(output_file_path)
    
#     return output_file_path, max_score

# def get_device():
#     return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 处理CSV文件并更新"subquestion_image_path"列
# def update_csv_with_image_paths(csv_path, output_dir):
#     # 读取CSV文件
#     df = pd.read_csv(csv_path)
    
#     # 添加新的列 "subquestion_image_path"
#     df["subquestion_image_path"] = ""

#     # 遍历每一行
#     # 遍历每一行
#     for index, row in df.iterrows():
#         # 检查 "subquestion_image_path" 列是否已有值
#         if pd.notna(row["subquestion_image_path"]) is not None:
#             continue  # 如果有值，跳过此行
        
#         # 获取 image_path 并处理
#         original_image_path = row["image_path"]
#         parts = original_image_path.split('/')[-5:]  # 取最后五个部分
#         relative_image_path = os.path.join(*parts)
#         real_image_path = os.path.join("/net/scratch/zhaorun/yiqiao", relative_image_path)
        
#         # 获取 sub_question 和 answer
#         sub_question = row["sub_question"]
#         sub_answer = row["answer"]
        
#         # 调用 process_image 函数
#         best_image_path, score = process_image(real_image_path, sub_question, sub_answer, output_dir)
#         print(f"question of {original_image_path} best score get {score}, save sub image {best_image_path}")
        
#         # 将结果写入 "subquestion_image_path" 列
#         df.at[index, "subquestion_image_path"] = best_image_path

    
#     # 保存更新后的CSV文件
#     df.to_csv(csv_path, index=False)

# # 使用示例
# csv_path = "/net/scratch/zhaorun/yiqiao/LLaMA/puzzle_18_61_info.csv"
# output_dir = "/net/scratch/zhaorun/yiqiao/LLaMA/save_9_picture"  # 需要替换为实际的输出目录

# update_csv_with_image_paths(csv_path, output_dir)

import os
import pandas as pd
from PIL import Image
import torch
import clip

def calculate_clip_score(image, sub_question, sub_answer, model, preprocess, device):
    try:
        text = f"{sub_question} {sub_answer}"
        tokenized_text = clip.tokenize([text]).to(device)
    except RuntimeError as e:
        if "too long for context length" in str(e):
            print(f"Text is too long, using only sub_question: {sub_question}")
            text = sub_question  # 仅使用 sub_question
            tokenized_text = clip.tokenize([text]).to(device)
        else:
            raise  # 其他错误，抛出异常

    image_input = preprocess(image).unsqueeze(0).to(device)
    text_input = clip.tokenize([text]).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T).item()
    
    return similarity

def extract_custom_basename(image_path):
    # 分割路径获取所有部分
    path_parts = image_path.split(os.path.sep)
    
    # 获取文件名部分（不包括扩展名）
    file_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 取得目录名中的数字作为编号
    dir_number = path_parts[-3]  # 假设数字总是在倒数第三部分，如上例中的 '18'
    
    # 修改文件名中的数字为目录中的数字
    new_base_name = file_name.replace(file_name.split('_')[1], dir_number)
    
    return dir_number, new_base_name

def process_image(image_path, sub_question, sub_answer, output_dir):
    # Load image
    image = Image.open(image_path)
    width, height = image.size
    
    # Calculate the size of each grid
    grid_width = width // 3
    grid_height = height // 3
    
    max_score = -float('inf')
    best_grid_image = None
    best_grid_index = None
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate over 9 grids
    for i in range(3):
        for j in range(3):
            left = j * grid_width
            upper = i * grid_height
            right = left + grid_width
            lower = upper + grid_height
            
            grid_image = image.crop((left, upper, right, lower))
            
            score = calculate_clip_score(grid_image, sub_question, sub_answer, model, preprocess, device)
            
            if score > max_score:
                max_score = score
                best_grid_image = grid_image
                best_grid_index = i * 3 + j + 1  # Index of the grid (1 to 9)

    # Calculate the size of the intersection points' grid
    intersection_grid_width = grid_width // 2
    intersection_grid_height = grid_height // 2

    # Iterate over 4 intersection grids
    for i in range(2):
        for j in range(2):
            left = (j + 1) * grid_width - intersection_grid_width // 2
            upper = (i + 1) * grid_height - intersection_grid_height // 2
            right = left + intersection_grid_width
            lower = upper + intersection_grid_height
            
            intersection_image = image.crop((left, upper, right, lower))
            
            score = calculate_clip_score(intersection_image, sub_question, sub_answer, model, preprocess, device)
            
            if score > max_score:
                max_score = score
                best_grid_image = intersection_image
                best_grid_index = 9 + i * 2 + j + 1  # Index of the intersection grid (10 to 13)

    # Define the output file name and path
    # base_name = os.path.splitext(os.path.basename(image_path))[0]
    dir_num, base_name = extract_custom_basename(image_path)
    output_file_name = f"{base_name}_grid{best_grid_index}.png"
    output_file_dir = os.path.join(output_dir, f"{dir_num}")
    if not os.path.exists(output_file_dir):
        os.mkdir(output_file_dir)
    output_file_path = os.path.join(output_file_dir, output_file_name)
    
    # Save the best grid image if it doesn't already exist
    if not os.path.exists(output_file_path):
        best_grid_image.save(output_file_path)
    
    return output_file_path, max_score

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def update_csv_with_image_paths(csv_path, output_dir):
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 初始化 "subquestion_image_path" 和 "best_score" 列为空值
    if "subquestion_image_path" not in df.columns:
        df["subquestion_image_path"] = None
    if "best_score" not in df.columns:
        df["best_score"] = None

    # 遍历每一行
    for index, row in df.iterrows():
        if row["puzzle_id"] == 71:
            continue  # 跳过处理，进行下一行
        # 检查 "subquestion_image_path" 列是否已有值
        if pd.isna(row["subquestion_image_path"]):
            # 获取 image_path 并处理
            original_image_path = row["image_path"]
            parts = original_image_path.split('/')[-5:]  # 取最后五个部分
            relative_image_path = os.path.join(*parts)
            real_image_path = os.path.join("/net/scratch/zhaorun/yiqiao", relative_image_path)
            
            # 获取 sub_question 和 answer
            sub_question = row["sub_question"]
            sub_answer = row["answer"]
            
            # 调用 process_image 函数
            best_image_path, max_score = process_image(real_image_path, sub_question, sub_answer, output_dir)
            print(f"question of {original_image_path} best score get {max_score}, save sub image {best_image_path}")
            
            # 将结果写入 "subquestion_image_path" 和 "best_score" 列
            df.at[index, "subquestion_image_path"] = best_image_path
            df.at[index, "best_score"] = max_score
    
            # 保存更新后的CSV文件
            df.to_csv(csv_path, index=False)

# Load CLIP model
device = get_device()
model, preprocess = clip.load("ViT-B/32", device=device)

# 使用示例
# csv_path = "/net/scratch/zhaorun/yiqiao/LLaMA/puzzle_18_61_info.csv"
# csv_path = "/net/scratch/zhaorun/yiqiao/LLaVA_0/puzzle_18_61_62_69.csv"
csv_path = "/net/scratch/zhaorun/yiqiao/LLaVA_0/puzzle_71_73_77_94_99.csv"
# output_dir = "/net/scratch/zhaorun/yiqiao/LLaMA/save_9_picture/new"
output_dir = "/net/scratch/zhaorun/yiqiao/LLaMA/save_9_picture/new_71_99"

update_csv_with_image_paths(csv_path, output_dir)
