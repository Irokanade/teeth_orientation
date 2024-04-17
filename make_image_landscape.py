import os
from PIL import Image

def makeImgLandscape(image):
    # Check if the image is in portrait orientation (height > width)
    if image.height > image.width:
        # Swap width and height to make it landscape
        image = image.transpose(Image.Transpose.ROTATE_90)
    return image

# Path to the dataset/sample folder
result_folder_path = 'dataset/sample'
output_folder_path = 'dataset/rotated'

# Iterate over each subfolder inside the dataset/sample folder
for folder_name in os.listdir(result_folder_path):
    folder_path = os.path.join(result_folder_path, folder_name)
    
    # Check if the path is a directory
    if os.path.isdir(folder_path):
        print(f"Processing folder: {folder_name}")
        
        # Create the corresponding output subfolder in the rotated folder
        output_subfolder_path = os.path.join(output_folder_path, folder_name)
        os.makedirs(output_subfolder_path, exist_ok=True)
        
        # Iterate over each image file in the subfolder
        for file_name in os.listdir(folder_path):
            # Construct the full file paths
            input_image_path = os.path.join(folder_path, file_name)
            output_image_path = os.path.join(output_subfolder_path, file_name)
            
            # Check if the file is an image
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                # Open the image
                with Image.open(input_image_path) as image:
                    # Rotate the image if needed
                    rotated_image = makeImgLandscape(image)
                    
                    # Save the rotated image to the output folder
                    rotated_image.save(output_image_path)
                    
                    print(f"Processed: {output_image_path}")

print("All images processed successfully!")
