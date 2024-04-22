import os
import pandas as pd
from PIL import Image, ImageDraw
import csv
import numpy as np
from matplotlib.pyplot import *
from matplotlib import pyplot as plt
import math

def rotate(x,y,angle): #rotate x,y around 0,0 by angle (rad)
    xr=math.cos(angle)*x-math.sin(angle)*y
    yr=math.sin(angle)*x+math.cos(angle)*y
    return [xr,yr]

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

# Path to the 'result' folder
result_folder_path = './result'
output_folder_path = './orientation'

for folder_name in os.listdir(result_folder_path):
    folder_path = os.path.join(result_folder_path, folder_name)
    print(folder_name)

    if os.path.isdir(folder_path):
        sample_folder_path = os.path.join(folder_path, 'sample')
        mask_folder_path = os.path.join(folder_path, 'mask')
        csv_file_path = os.path.join(folder_path, 'imageClassification.csv')
        node_folder_path = os.path.join(folder_path, 'node')
        box_folder_path = os.path.join(folder_path, 'boundingBox')

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

                    imgHeight = image.height
                    imgWidth = image.width
                    # make image landscape
                    image = makeImgLandscape(image)

                    # process image
                    if label != 'Face':
                        # if not face then it is mirrored
                        image = image.transpose(Image.FLIP_LEFT_RIGHT)

                    # Create the output folder structure mirroring the input structure
                    output_subfolder_path = os.path.join(output_folder_path, folder_name, '0_sample')
                    os.makedirs(output_subfolder_path, exist_ok=True)

                    # Save the processed image to the 'orientation' folder
                    output_image_path = os.path.join(output_subfolder_path, image_filename)
                    image.save(output_image_path)


                    # Load bounding boxes' positions
                    csv_path = os.path.join(box_folder_path, f"box_{image_filename[:-3]}csv")
                    readFile = open(csv_path,mode='r',newline='')
                    csvreader = csv.reader(readFile)
                    boxArray = []
                    for row in csvreader:
                        nArray = []
                        for item in row:
                            nArray.append(float(item))
                        boxArray.append(nArray)
                    
                    # Load nodes' positions
                    csv_path = os.path.join(node_folder_path, f"node_{image_filename[:-3]}csv")
                    readFile = open(csv_path,mode='r',newline='')
                    csvreader = csv.reader(readFile)
                    nodeArray = []
                    for row in csvreader:
                        nArray = []
                        for item in row:
                            nArray.append(float(item))
                        nodeArray.append(nArray)

                    # Find nodes far away from others
                    newnodeArray = []
                    for i in range(len(nodeArray[0])):
                        newnodeArray.append([nodeArray[0][i],nodeArray[1][i]])

                    column_name = ['x','y']
                    node_df = pd.DataFrame(newnodeArray)
                    node_df.columns = column_name

                    node_df['x'] = np.log1p(node_df['x'])
                    node_df['y'] = np.log1p(node_df['y'])

                    Q1 = np.percentile(node_df['x']**2+node_df['y']**2-Cutimage.width**2-Cutimage.height**2, 25)
                    Q3 = np.percentile(node_df['x']**2+node_df['y']**2-Cutimage.width**2-Cutimage.height**2, 75)
                    IQR = Q3 - Q1
                    n = 1.0

                    upper = (node_df['x']**2+node_df['y']**2-Cutimage.width**2-Cutimage.height**2) >= (Q3+n*IQR)
                    outputArray = np.array(np.where(upper))
                    outputArray[0] = sorted(outputArray[0], reverse=True)
                    # Delete nodes and bounding boxes
                    if len(outputArray[0]) > 0:
                        nodeArray[0] = np.delete(nodeArray[0],outputArray[0])
                        nodeArray[1] = np.delete(nodeArray[1],outputArray[0])
                        for i in outputArray[0]:
                            del boxArray[i]

                    lower = (node_df['x']**2+node_df['y']**2-Cutimage.width**2-Cutimage.height**2) <= (Q1-n*IQR)
                    outputArray = np.array(np.where(lower))
                    outputArray[0] = sorted(outputArray[0], reverse=True)
                    # Delete nodes and bounding boxes
                    if len(outputArray[0]) > 0:
                        nodeArray[0] = np.delete(nodeArray[0],outputArray[0])
                        nodeArray[1] = np.delete(nodeArray[1],outputArray[0])
                        for i in outputArray[0]:
                            del boxArray[i]


                    # Rotate nodes, bounding boxes, and image 90 degrees
                    if Cutimage.height > Cutimage.width:
                        for box in boxArray:
                            x0, y0, x1, y1 = box
                            box[0] = y0
                            box[1] = Cutimage.width - x1
                            box[2] = y1
                            box[3] = Cutimage.width - x0
                        temp = nodeArray[0]
                        nodeArray[0] = nodeArray[1]
                        nodeArray[1] = temp
                        for i in range(len(nodeArray[1])):
                            nodeArray[1][i] = imgWidth - nodeArray[1][i]
                    Cutimage = makeImgLandscape(Cutimage)

                    # Flip nodes, bounding boxes, and image horizontally
                    if label != 'Face':
                        Cutimage = Cutimage.transpose(Image.FLIP_LEFT_RIGHT)
                        for box in boxArray:
                            x0, y0, x1, y1 = box
                            box[0] = Cutimage.width - x1
                            box[2] = Cutimage.width - x0
                        for i in range(len(nodeArray[0])):
                            nodeArray[0][i] = image.width - nodeArray[0][i]
                    
                    # Create polynomial line
                    polyLine = np.polyfit(nodeArray[0],nodeArray[1],2)
                    p = np.poly1d( polyLine )
                    
                    # Find critical points
                    bounds = [0, Cutimage.width-1]
                    crit_points = bounds + [x for x in p.deriv().r if x.imag == 0 and bounds[0] < x.real < bounds[1]]
                    #print(crit_points)

                    # Find two points on the line
                    p1 = 0
                    p2 = 0
                    if len(crit_points) == 3:
                        if crit_points[2] + 50 > Cutimage.width-1:
                            p2 = Cutimage.width-1
                        else:
                            p2 = crit_points[2] + 50
                        if p(crit_points[2]) < p(p2):
                            Cutimage = Cutimage.transpose(Image.Transpose.ROTATE_180)
                            for box in boxArray:
                                x0, y0, x1, y1 = box
                                box[0] = Cutimage.width - x1
                                box[1] = Cutimage.height - y1
                                box[2] = Cutimage.width - x0
                                box[3] = Cutimage.height - y0
                            for i in range(len(nodeArray[1])):
                                nodeArray[0][i] = Cutimage.width - nodeArray[0][i]
                                nodeArray[1][i] = Cutimage.height - nodeArray[1][i]
                            polyLine = np.polyfit(nodeArray[0],nodeArray[1],2)
                            p = np.poly1d( polyLine )
                            p2 = Cutimage.width - p2
                            rangeArray = np.arange(Cutimage.width,0,-0.01)
                        else:
                            rangeArray = np.arange(0,Cutimage.width,0.01)
                        for i in rangeArray:
                            if (p(i)-p(p2)<=0.01 and p(i)-p(p2)>=0) or (p(p2)-p(i)<=0.01 and p(p2)-p(i)>=0):
                                p1 = i
                                break
                    else:
                        p2 = crit_points[1]
                        p1 = crit_points[1] - 10
                        if p(crit_points[1]) < p(crit_points[0]):
                            Cutimage = Cutimage.transpose(Image.Transpose.ROTATE_180)
                            for box in boxArray:
                                x0, y0, x1, y1 = box
                                box[0] = Cutimage.width - x1
                                box[1] = Cutimage.height - y1
                                box[2] = Cutimage.width - x0
                                box[3] = Cutimage.height - y0
                            for i in range(len(nodeArray[1])):
                                nodeArray[0][i] = Cutimage.width - nodeArray[0][i]
                                nodeArray[1][i] = Cutimage.height - nodeArray[1][i]
                            polyLine = np.polyfit(nodeArray[0],nodeArray[1],2)
                            p = np.poly1d( polyLine )
                    pltImage = np.array(Cutimage)
                    plt.imshow(pltImage)
                    # Plot nodes
                    plt.scatter(nodeArray[0],nodeArray[1])

                    # Plot new line
                    x_base = np.linspace(0,Cutimage.width,Cutimage.width)
                    plt.plot(x_base, p(x_base),color = 'red')
                    plt.scatter([p1,p2], [p(p1),p(p2)])

                    # Save img with nodes and line
                    output_subfolder_path = os.path.join(output_folder_path, folder_name, '2_line')
                    os.makedirs(output_subfolder_path, exist_ok=True)

                    output_image_path = os.path.join(output_subfolder_path, f"line_{image_filename}")
                    plt.savefig(output_image_path)
                    plt.clf()

                    # The angle to rotate
                    slope = (p(p2)-p(p1)) / (p2-p1)
                    angle = np.rad2deg(np.arctan(slope))

                    Cutimage = Cutimage.rotate(angle,expand=1)

                    # Save rotated image
                    output_subfolder_path = os.path.join(output_folder_path, folder_name, '1_rotate')
                    os.makedirs(output_subfolder_path, exist_ok=True)

                    output_image_path = os.path.join(output_subfolder_path, f"rotate_{image_filename}")
                    Cutimage.save(output_image_path)

                     
                    # Rotate nodes and bounding boxes
                    for i in range(len(nodeArray[0])):
                        nodeArray[0][i], nodeArray[1][i] = rotate(nodeArray[0][i], nodeArray[1][i], -np.arctan(slope))
                        if slope < 0:
                            nodeArray[0][i] += (math.sin(-np.arctan(slope)) * image.height)
                        elif slope > 0:
                            nodeArray[1][i] += (math.sin(np.arctan(slope)) * image.width)
                    for box in boxArray:
                        x0, y0, x1, y1 = box
                        boxwidth = x1 - x0
                        boxheight = y1 - y0
                        box[0], box[1] = rotate(x0, y0, -np.arctan(slope))
                        box[2], box[3] = rotate(x1, y1, -np.arctan(slope))
                        if slope < 0:
                            box[0] += math.sin(-np.arctan(slope)) * image.height
                            box[0] -= math.sin(-np.arctan(slope)) * boxheight
                        elif slope > 0:
                            box[1] += math.sin(np.arctan(slope)) * image.width
                            box[1] -= math.sin(np.arctan(slope)) * boxwidth
                        if slope < 0:
                            box[2] += math.sin(-np.arctan(slope)) * image.height
                            box[2] += math.sin(-np.arctan(slope)) * boxheight
                        elif slope > 0:
                            box[3] += math.sin(np.arctan(slope)) * image.width
                            box[3] += math.sin(np.arctan(slope)) * boxwidth
                    ''''''

                    pltImage = np.array(Cutimage)
                    plt.imshow(pltImage)
                    # Plot nodes
                    plt.scatter(nodeArray[0],nodeArray[1])

                    # Create a new line after rotating
                    polyLine = np.polyfit(nodeArray[0],nodeArray[1],2)
                    p = np.poly1d( polyLine )
                    
                    # Find the new critical points
                    bounds = [0, Cutimage.width-1]
                    crit_points = bounds + [x for x in p.deriv().r if x.imag == 0 and bounds[0] < x.real < bounds[1]]

                    # Plot new line
                    x_base = np.linspace(0,Cutimage.width,Cutimage.width)
                    plt.plot(x_base, p(x_base),color = 'red')

                    # Save img with nodes and line
                    output_subfolder_path = os.path.join(output_folder_path, folder_name, '2_line_after_rotate')
                    os.makedirs(output_subfolder_path, exist_ok=True)

                    output_image_path = os.path.join(output_subfolder_path, f"line_after_rotate_{image_filename}")
                    plt.savefig(output_image_path)
                    plt.clf()

                    # Draw bounding boxes
                    Cutimg = Cutimage.copy()
                    draw = ImageDraw.Draw(Cutimg)
                    for box in boxArray:
                        x1, y1, x2 ,y2 = box
                        color = tuple(np.random.choice(range(256), size=3))
                        detLineScale = 0.005
                        draw.line([(x1,y1),(x2,y1),(x2,y2),(x1,y2),(x1,y1)], fill=color, width=int(Cutimage.width*detLineScale))

                    # Save img with bounding boxes
                    output_subfolder_path = os.path.join(output_folder_path, folder_name, '2_det')
                    os.makedirs(output_subfolder_path, exist_ok=True)

                    output_image_path = os.path.join(output_subfolder_path, f"det_{image_filename}")
                    Cutimg.save(output_image_path)

                    # Find the area to crop
                    minX = -1
                    maxX = -1
                    minY = 100000
                    maxY = 0
                    minYx = 0
                    maxYx = 0
                    for box in boxArray:
                        x1, y1, x2 ,y2 = box
                        if minY > y1:
                            minY = y1
                            minYx = (x1+x2)/2
                        if maxY < y2:
                            maxY = y2
                            maxYx = (x1+x2)/2
                    minY =  max(0,minY-20)
                    maxY = min(Cutimage.height, maxY+20)
                    rangeArray = np.arange(0,Cutimage.width,1)
                    for x in rangeArray:
                        if minX == -1 and p(x) <= 5 and p(x) >= 0:
                            if x - 30 >= 0:
                                minX = x - 30
                            else:
                                minX = 0
                            if len(boxArray) >= 14 and len(boxArray) <= 16:
                                minY = minY
                            else:
                                minY = 0
                            if len(crit_points) == 3:
                                if maxY < p(crit_points[2])+20 and maxYx > crit_points[2]-70 and maxYx < crit_points[2]+70:
                                    continue
                                elif maxY > p(crit_points[2])+20:
                                    continue
                                if p(crit_points[2]) + 20 <= Cutimage.height:
                                    maxY = p(crit_points[2]) + 20
                                else:
                                    maxY = Cutimage.height
                            else:
                                if maxY < p(crit_points[1])+20 and maxYx > crit_points[1]-70 and maxYx < crit_points[1]+70:
                                    continue
                                elif maxY > p(crit_points[1])+20:
                                    continue
                                if p(crit_points[1]) + 20 <= Cutimage.height:
                                    maxY = p(crit_points[1]) + 20
                                else:
                                    maxY = Cutimage.height
                        elif minX == -1 and (Cutimage.height - p(x)) <= 5 and (Cutimage.height - p(x)) >= 0:
                            if x - 30 >= 0:
                                minX = x - 30
                            else:
                                minX = 0
                            if len(crit_points) == 3:
                                if minY > p(crit_points[2])-20 and minYx > crit_points[2]-70 and minYx < crit_points[2]+70:
                                    minY = minY
                                elif minY < p(crit_points[2])-20:
                                    minY = minY
                                if p(crit_points[2]) - 20 >= 0:
                                    minY = p(crit_points[2]) - 20
                                else:
                                    minY = 0
                            else:
                                if minY > p(crit_points[1])-20 and minYx > crit_points[1]-70 and minYx < crit_points[1]+70:
                                    minY = minY
                                elif minY < p(crit_points[1])-20:
                                    minY = minY
                                if p(crit_points[1]) - 20 >= 0:
                                    minY = p(crit_points[1]) - 20
                                else:
                                    minY = 0
                            if len(boxArray) >= 14 and len(boxArray) <= 16:
                                continue
                            else:
                                maxY = Cutimage.height
                        elif ((p(x) <= 5 and p(x) >= 0) or ((Cutimage.height - p(x)) <= 5 and (Cutimage.height - p(x)) >= 0)) and x - minX > 100:
                            if x + 30 <= Cutimage.width:
                                maxX = x + 30
                            else:
                                maxX = Cutimage.width

                    if image_filename == "IMG_4382.png" or image_filename == "IMG_4381.png" or image_filename == "IMG_4494.png":
                            print(minX, ", ", maxX)
                    if minX < 0 or maxX <= 0:
                        if image_filename == "IMG_4382.png" or image_filename == "IMG_4381.png" or image_filename == "IMG_4494.png":
                            print("next")
                        minX = 100000
                        minY = 100000
                        maxX = 0
                        maxY = 0
                        for box in boxArray:
                            x1, y1, x2 ,y2 = box
                            minX = min(x1, minX)
                            minY = min(y1, minY)
                            maxX = max(x2, maxX)
                            maxY = max(y2, maxY)

                        minX = max(0,minX-30)
                        minY =  max(0,minY-30)
                        maxX = min(Cutimage.width, maxX+30)
                        maxY = min(Cutimage.height, maxY+30)
                    
                    if (maxY - minY) * 1.5 < (maxX - minX):
                        if (maxY + (((maxX-minX)-(maxY-minY)*1.5)/1.5)/2) < Cutimage.height:
                            maxY += (((maxX-minX)-(maxY-minY)*1.5)/1.5) / 2
                            if (minY - (((maxX-minX)-(maxY-minY)*1.5)/1.5)/2) > 0:
                                minY -= (((maxX-minX)-(maxY-minY)*1.5)/1.5)/2
                            else:
                                maxY -= ((((maxX-minX)-(maxY-minY)*1.5)/1.5)/2 - minY)
                                minY = 0
                        else:
                            minY -= (((maxX-minX)-(maxY-minY)*1.5)/1.5) / 2
                            minY += ((maxY + (((maxX-minX)-(maxY-minY)*1.5)/1.5)/2) - Cutimage.height)
                            maxY = Cutimage.height
                    elif (maxY - minY) * 1.5 > (maxX - minX):
                        if (maxX + ((maxY-minY)*1.5-(maxX-minX))/2) < Cutimage.width:
                            maxX += ((maxY-minY)*1.5-(maxX-minX)) / 2
                            if (minX - ((maxY-minY)*1.5-(maxX-minX))/2) > 0:
                                minX -= ((maxY-minY)*1.5-(maxX-minX)) / 2
                            else:
                                maxX -= (((maxY-minY)*1.5-(maxX-minX))/2 - minX)
                                minX = 0
                        else:
                            minX -= ((maxY-minY)*1.5-(maxX-minX)) / 2
                            minX += ((maxX + ((maxY-minY)*1.5-(maxX-minX))/2) - Cutimage.width)
                            maxX = Cutimage.width
                    ''''''

                    # Crop image
                    Cutimage = Cutimage.crop((minX,minY,maxX,maxY))
                    #print()

                    # Save cropped image
                    output_subfolder_path = os.path.join(output_folder_path, folder_name, '3_crop')
                    os.makedirs(output_subfolder_path, exist_ok=True)

                    output_image_path = os.path.join(output_subfolder_path, f"crop_{image_filename}")
                    Cutimage.save(output_image_path)



        if os.path.exists(mask_folder_path):
            # Iterate over image files in the 'sample' folder
            for image_filename in os.listdir(mask_folder_path):
                image_path = os.path.join(mask_folder_path, image_filename)

                # Check if it's a file and has a valid image extension
                if os.path.isfile(image_path) and image_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image = Image.open(image_path)

                    image_filename_without_prefix = image_filename.replace("mask_", "")
                    print(f"Processing image: {image_filename}, Size: {image.size}")
                    label = getFaceFromCsv(image_filename_without_prefix, df)
                    print(f"Image: {image_filename}, Label: {label}")

                    # if no label the skip
                    if label is None:
                        continue

                    # make image landscape
                    image = makeImgLandscape(image)

                    # process image
                    if label != 'Face':
                        # if not face then it is mirrored
                        image = image.transpose(Image.FLIP_LEFT_RIGHT)

                    # Create the output folder structure mirroring the input structure
                    output_subfolder_path = os.path.join(output_folder_path, folder_name, '0_mask')
                    os.makedirs(output_subfolder_path, exist_ok=True)

                    # Save the processed image to the 'orientation' folder
                    output_image_path = os.path.join(output_subfolder_path, image_filename)
                    image.save(output_image_path)

        