from PIL import Image
import os

# Define the directory where the patches are stored
input_dir = "/scratch/czr/LVLM-Reasoning/debug"
output_image_path = os.path.join(input_dir, "reconstructed_image.png")

# Define the size of each patch and the grid dimensions
patch_size = 14  # Size of each patch
grid_size = 448 // 14          # Number of patches along one dimension (16x16 grid)

# Create a blank image to hold the reconstructed image
whole_image = Image.new('RGB', (patch_size * grid_size, patch_size * grid_size))

# Iterate through each patch and paste it into the correct position
for i in range(grid_size):
    for j in range(grid_size):
        # Load the patch
        patch_path = os.path.join(input_dir, f"patch_{i}_{j}.png")
        patch = Image.open(patch_path)
        
        # Calculate the position where this patch should go in the whole image
        x = j * patch_size
        y = i * patch_size
        
        # Paste the patch into the whole image
        whole_image.paste(patch, (x, y))

# Save the reconstructed whole image
whole_image.save(output_image_path)
print(f"Reconstructed image saved at {output_image_path}")
