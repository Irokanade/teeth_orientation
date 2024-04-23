import os
import pandas as pd
import csv
from PIL import Image
from getFace import getFace

def makeImgLandscape(image):
    # Check if the image is in portrait orientation (height > width)
    if image.height > image.width:
        # Swap width and height to make it landscape
        image = image.transpose(Image.Transpose.ROTATE_90)
    return image

def getFaceFromCsv(imageName, imageClassificationDf):
    # Remove file extension from imageName
    imageName_base, _ = os.path.splitext(imageName)

    for index, row in imageClassificationDf.iterrows():
        currentImageName = row.iloc[0]  # Assuming the image filename is in the first column

        # Remove file extension from currentImageName
        currentImageName_base, _ = os.path.splitext(currentImageName)

        if imageName_base == currentImageName_base:
            label = row.iloc[1]  # Assuming the label is in the second column
            return label
        
    # else return none
    return None

# # Path to the dataset/sample folder
# result_folder_path = 'dataset/sample'
# output_folder_path = 'dataset/rotated'

# # Iterate over each subfolder inside the dataset/sample folder
# for folder_name in os.listdir(result_folder_path):
#     folder_path = os.path.join(result_folder_path, folder_name)
    
#     # Check if the path is a directory
#     if os.path.isdir(folder_path):
#         print(f"Processing folder: {folder_name}")
        
#         # Create the corresponding output subfolder in the rotated folder
#         output_subfolder_path = os.path.join(output_folder_path, folder_name)
#         os.makedirs(output_subfolder_path, exist_ok=True)
        
#         # Iterate over each image file in the subfolder
#         for file_name in os.listdir(folder_path):
#             # Construct the full file paths
#             input_image_path = os.path.join(folder_path, file_name)
#             output_file_name = f"rotated_{file_name}"
#             output_image_path = os.path.join(output_subfolder_path, output_file_name)
            
#             # Check if the file is an image
#             if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
#                 # Open the image
#                 with Image.open(input_image_path) as image:
#                     # Rotate the image if needed
#                     rotated_image = makeImgLandscape(image)
                    
#                     # Save the rotated image to the output folder
#                     rotated_image.save(output_image_path)
                    
#                     print(f"Processed: {output_image_path}")


getFace()
print('got faces in result folder')

# mirror images base on face
# Path to the 'result' folder
result_folder_path = './result'
output_folder_path = './dataset/rotatedAndMirrored'

for folder_name in os.listdir(result_folder_path):
    folder_path = os.path.join(result_folder_path, folder_name)
    print(folder_name)

    if os.path.isdir(folder_path):
        sample_folder_path = os.path.join(folder_path, 'sample')
        # mask_folder_path = os.path.join(folder_path, 'mask')
        csv_file_path = os.path.join(folder_path, 'imageClassification.csv')

        # Skip first line as the data is 2D or 3D
        # also add custom labels
        df = pd.read_csv(csv_file_path, skiprows=1, names=['image', 'angle'])

        if os.path.exists(sample_folder_path):
            # Iterate over image files in the 'sample' folder
            for image_filename in os.listdir(sample_folder_path):
                image_path = os.path.join(sample_folder_path, image_filename)

                # Check if it's a file and has a valid image extension
                if os.path.isfile(image_path) and image_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image = Image.open(image_path)
                    Cutimage = image.copy()

                    # Print image information
                    print(f"Processing image: {image_filename}, Size: {image.size}")
                    label = getFaceFromCsv(image_filename, df)
                    print(f"Image: {image_filename}, Label: {label}")

                    # if no label the skip
                    #if label is None:
                    #    continue

                    # process image
                    if label != 'Face':
                        # if not face then it is mirrored
                        image = image.transpose(Image.FLIP_LEFT_RIGHT)

                    # Create the output folder structure mirroring the input structure
                    output_subfolder_path = os.path.join(output_folder_path, folder_name)
                    os.makedirs(output_subfolder_path, exist_ok=True)

                    # Save the processed image to the 'rotated and mirrored' folder
                    output_image_path = os.path.join(output_subfolder_path, image_filename)
                    image.save(output_image_path)