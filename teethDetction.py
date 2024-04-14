import os
import csv
import cv2
import torch
import numpy as np

import PIL.Image
import PIL.ImageOps
from PIL import Image, ImageDraw

from matplotlib import pyplot as plt
from maskrcnn import my_get_model_instance_segmentation


def absoluteToScale(teethLocation,imageWidth,imageHeight):

    x1 = teethLocation.x1
    x2 = teethLocation.x2
    y1 = teethLocation.y1
    y2 = teethLocation.y2
    return [ ((x1+x2)/2)/imageWidth,((y1+y2)/2)/imageHeight,((x2-x1))/imageWidth,((y2-y1))/imageHeight ]

class ImageFile:
    def __init__(self,FileName):
        self.fileName = FileName
        self.is3D = True
        self.photoImageSet = []
        self.missingLabelId = []

class PhotoImage:
    def __init__(self,imageName,gradient,teethNum,grayData,teethLocationSet,width,height,image,imageTeethNodeSet, p,polyLine):   
        self.imageName = imageName
        self.gradient = gradient
        self.absGradient = abs(gradient)
        self.gradientRank = -1
        self.teethNum = teethNum
        self.teethNumRank = -1
        self.image = image
        self.grayData = grayData
        self.useFlag = False
        self.teethLocationSet = teethLocationSet
        self.width = width
        self.height = height
        self.teethScaleSet = [absoluteToScale(teethLocation,width,height) for teethLocation in teethLocationSet] 
        self.view = 'Unknown'
        self.teethNodeSet = imageTeethNodeSet
        self.polyLine = polyLine

        self.regression = p

class TeethNode:
    def __init__(self,mask,box):   
        self.mask = mask.astype(np.uint8)
        self.box = box
        self.labelId = -1

class TeethLocation:
    def __init__(self,x1,y1,x2,y2):   
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

def exif_transpose(img):
    if not img:
        return img

    exif_orientation_tag = 274

    if hasattr(img, "_getexif") and isinstance(img._getexif(), dict) and exif_orientation_tag in img._getexif():
        exif_data = img._getexif()
        orientation = exif_data[exif_orientation_tag]

        if orientation == 1:

            pass
        elif orientation == 2:

            img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:

            img = img.rotate(180)
        elif orientation == 4:

            img = img.rotate(180).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 5:

            img = img.rotate(-90, expand=True).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 6:

            img = img.rotate(-90, expand=True)
        elif orientation == 7:

            img = img.rotate(90, expand=True).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 8:

            img = img.rotate(90, expand=True)

    return img

def sortGradient(infoSet):
    infoSet = sorted(infoSet,key = lambda x : x.absGradient,reverse = True)
    for i in range( len(infoSet) ):
        infoSet[ i ].gradientRank = i+1

def sortTeeth(infoSet):
    infoSet = sorted(infoSet,key = lambda x : x.teethNum,reverse = True)
    for i in range( len(infoSet) ):
        infoSet[ i ].teethRank = i+1

def checkFlag3D(pltImage, resizeScale, proportion, imageWidth, imageHeight):
    resizedImage = cv2.resize(pltImage, dsize=(int(imageWidth * resizeScale), int(imageHeight * resizeScale)), interpolation=cv2.INTER_CUBIC)
    white = (255, 255, 255)
    mask = cv2.inRange(resizedImage, white, white)
    whitePixelCnt = cv2.countNonZero(mask)
    totalPixel = resizedImage.shape[0] * resizedImage.shape[1]

    return whitePixelCnt < totalPixel * proportion

def pilSave(image,path,fileLabel,prefix,imageName):
    check = imageName.split('.')
    if prefix != "" :
        prefix += "_"
    if len(check[-1]) < 4 :
        image.save(f"{path}/{fileLabel}/{prefix}{imageName[:-3]}png")
    else:
        image.save(f"{path}/{fileLabel}/{prefix}{imageName[:-4]}png")

def pltSave(path,fileLabel,prefix,imageName):
    check = imageName.split('.')
    if prefix != "" :
        prefix += "_"
    if len(check[-1]) < 4 :
        plt.savefig(f"{path}/{fileLabel}/{prefix}{imageName[:-3]}png")
    else:
        plt.savefig(f"{path}/{fileLabel}/{prefix}{imageName[:-4]}png")

def makeResultDirProcess(fileName):
    redir = 'result'
    file = os.path.join(redir)
    os.makedirs(file, exist_ok=True)
    file = os.path.join(redir, fileName)
    os.makedirs(file, exist_ok=True)
    createFile = ['det', 'regression', 'raw_images', 'mask', 'boundingBox', 'node']
    for name in createFile:
        tmp = os.path.join(redir, fileName, name)
        os.makedirs(tmp, exist_ok=True)

def initialize_system():
    print("Cuda is available = ", torch.cuda.is_available())
    isGPU = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device -> ", device)
    model = my_get_model_instance_segmentation(2)

    param_dict = torch.load("./ckpts/label_two_class_best_model.pth", device)
    model.load_state_dict(param_dict)
    model.to(device)
    model.eval()
    return model, device

def load_images(root, samdir):
    imgdir = list(sorted(os.listdir(os.path.join(root, samdir))))
    imgs = {}
    for fileName in imgdir:
        imgSet = list(sorted(os.listdir(os.path.join(root, samdir, fileName))))
        imgs[fileName] = imgSet
    return imgs

def process_images(imgs, model, device):
    bb_thre = 0.80
    mask_thre = 0.55
    for fileName, imgSet in imgs.items():
        print('\nProcess : ', fileName)
        makeResultDirProcess(fileName)
        imageFileInfo = ImageFile(fileName)
        imageInfoSet = []
        for imageName in imgSet:
            image, pltImage = load_and_preprocess_image(fileName, imageName, device)
            imageWidth, imageHeight = image.width, image.height
            img, _, _ = initialize_image_drawing(image)
            teethLocationSet, imageTeethNodeSet, xList, yList, npimg, npblack_img = detect_teeth(
                model, device, image, imageName, fileName, bb_thre, mask_thre, imageWidth, imageHeight)
            polyLine = analyze_image(xList, yList, pltImage, imageName, fileName, imageWidth, imageHeight)
            imageGray = cv2.cvtColor(pltImage, cv2.COLOR_BGR2GRAY)
            imageInfo = PhotoImage(imageName, polyLine[0], len(teethLocationSet), imageGray, teethLocationSet,
                                   imageWidth, imageHeight, pltImage, imageTeethNodeSet, np.poly1d(polyLine), polyLine)
            imageInfoSet.append(imageInfo)
            check_3D_flag(imageFileInfo, pltImage, imageName, fileName, image, npimg, npblack_img, imageWidth, imageHeight)
        classify_images(imageFileInfo, imageInfoSet, fileName)

def load_and_preprocess_image(fileName, imageName, device):
    image = Image.open(f"./dataset/Sample/{fileName}/{imageName}")
    image = exif_transpose(image)
    if image.height > image.width:
        image = image.rotate(90, expand=True)
    idealSize = 1024
    image.thumbnail((idealSize, idealSize))
    pltImage = np.array(image)
    pilSave(image, f"./result/{fileName}", "raw_images", "", imageName)
    return image, pltImage

def initialize_image_drawing(image):
    img = image.copy()
    black_img = Image.new('RGB', img.size, (0, 0, 0))
    return img, np.array(img, dtype=np.uint8), np.array(black_img, dtype=np.uint8)

def detect_teeth(model, device, image, imageName, fileName, bb_thre, mask_thre, imageWidth, imageHeight):
    images_tensor = torch.as_tensor(np.array(image)[...,:3]/255, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    images_tensor = images_tensor.to(device)
    output = model(images_tensor)[0]
    draw = ImageDraw.Draw(image)

    teethLocationSet, imageTeethNodeSet, xList, yList = [], [], [], []
    npblack_img = np.zeros((imageHeight, imageWidth, 3), dtype=np.uint8)

    bbox_file_path = f"./result/{fileName}/boundingBox/box_{imageName[:-4]}.csv"
    with open(bbox_file_path, "w", newline="") as bboxFile:
        csvWriter = csv.writer(bboxFile)

        for box, mask, score in zip(output["boxes"], output["masks"], output["scores"]):
            if score.item() > bb_thre:
                box_coords = [round(b.item()) for b in box]
                x1, y1, x2, y2 = box_coords
                teethLocationSet.append(TeethLocation(x1, y1, x2, y2))
                csvWriter.writerow([x1, y1, x2, y2]) 

                xList.append((x1 + x2) / 2)
                yList.append((y1 + y2) / 2)

                mask_np = mask.squeeze().cpu().detach().numpy()
                mask_binary = np.where(mask_np > mask_thre, 255, 0).astype(np.uint8)
                imageTeethNodeSet.append(TeethNode(mask_binary, box_coords))

                for i in range(3):
                    npblack_img[:, :, i] = np.where(mask_binary == 255, mask_np * 255, npblack_img[:, :, i])

                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

    node_file_path = f"./result/{fileName}/node/node_{imageName[:-4]}.csv"
    with open(node_file_path, "w", newline="") as nodeFile:
        csvWriter = csv.writer(nodeFile)
        csvWriter.writerow(xList)
        csvWriter.writerow(yList)

    return teethLocationSet, imageTeethNodeSet, xList, yList, image, npblack_img


def draw_bounding_box(npimg, teethLocation):
    cv2.rectangle(npimg, (teethLocation.x1, teethLocation.y1), (teethLocation.x2, teethLocation.y2), (0, 255, 0), 2)

def process_box(box, imageWidth, imageHeight):
    x1, y1, x2, y2 = [max(int(b), 0) for b in box[:2]] + [min(int(b), imageWidth) for b in box[2:]]
    return TeethLocation(x1, y1, x2, y2)

def process_mask(mask, mask_thre, imageTeethNodeSet, teethLocation, imageWidth, draw):
    mask = mask.detach().squeeze().cpu().numpy()
    mask = np.where(mask > mask_thre, 255, 0).astype(np.uint8)
    imageTeethNodeSet.append(TeethNode(mask, teethLocation))
    color = tuple(np.random.choice(range(256), size=3))
    detLineScale = 0.005
    draw.line([(teethLocation.x1, teethLocation.y1), (teethLocation.x2, teethLocation.y1),
               (teethLocation.x2, teethLocation.y2), (teethLocation.x1, teethLocation.y2),
               (teethLocation.x1, teethLocation.y1)], fill=color, width=int(imageWidth * detLineScale))

def analyze_image(xList, yList, pltImage, imageName, fileName, imageWidth, imageHeight):
    xArray = np.array(xList)
    yArray = np.array(yList)
    plt.xlim(0, imageWidth)
    plt.ylim(imageHeight, 0)
    polyLine = np.polyfit(xArray, yArray, 2)
    plt.imshow(pltImage)
    plt.scatter(xArray, yArray)
    p = np.poly1d(polyLine)
    x_base = np.linspace(0, imageWidth, imageWidth)
    plt.plot(x_base, p(x_base), color='red')
    pltSave(f"./result/{fileName}", "regression", "regression", imageName)
    plt.clf()
    return polyLine

def check_3D_flag(imageFileInfo, pltImage, imageName, fileName, img, npimg, npblack_img, imageWidth, imageHeight):
    resizeScale = 0.1
    whiteProportion = 1 / 100

    if not checkFlag3D(pltImage, resizeScale, whiteProportion, imageWidth, imageHeight):
        imageFileInfo.is3D = False

    pilSave(img, f"./result/{fileName}", "det", "det", imageName)

    mask_image = Image.fromarray(npblack_img)
    pilSave(mask_image, f"./result/{fileName}", "mask", "mask", imageName)

def classify_images(imageFileInfo, imageInfoSet, fileName):
    writeFile = open(f"./result/{fileName}/imageClassification.csv", mode="w", newline="")
    csvWriter = csv.writer(writeFile)

    if imageFileInfo.is3D:
        csvWriter.writerow(["3D"])
    else:
        csvWriter.writerow(["2D"])

    sortGradient(imageInfoSet)
    sortTeeth(imageInfoSet)

    for info in imageInfoSet:
        if not info.useFlag:
            if info.teethRank == 1 and info.gradientRank >= 3:
                csvWriter.writerow([info.imageName, "Face"])
                info.view = 'Face'
                info.useFlag = True
            elif info.teethRank != 1 and info.gradientRank <= 2:
                view = 'Up' if info.gradient >= 0 else 'Below'
                csvWriter.writerow([info.imageName, view])
                info.view = view
                info.useFlag = True

    classify_side_views(imageFileInfo, imageInfoSet, csvWriter)

    writeFile.close()

def classify_side_views(imageFileInfo, imageInfoSet, csvWriter):
    leftRightCnt = 0
    for info in imageInfoSet:
        if not info.useFlag:
            leftRightCnt += 1

    if leftRightCnt == 2:
        for info in imageInfoSet:
            if not info.useFlag:
                classify_left_right(imageFileInfo, info, csvWriter)
    else:
        print("~~~!!! Classification Error !!!~~~")

def classify_left_right(imageFileInfo, info, csvWriter):
    if imageFileInfo.is3D:
        classify_3D_view(info, csvWriter)
    else:
        classify_2D_view(info, csvWriter)

def classify_3D_view(info, csvWriter):
    info.teethLocationSet = sorted(info.teethLocationSet, key=lambda loc: (loc.x1 + loc.x2) / 2)
    teethCheckNum = 3
    leftCnt = sum((loc.x2 - loc.x1) / (loc.y2 - loc.y1) for loc in info.teethLocationSet[:teethCheckNum])
    rightCnt = sum((loc.x2 - loc.x1) / (loc.y2 - loc.y1) for loc in info.teethLocationSet[-teethCheckNum:])
    view = 'Right' if leftCnt > rightCnt else 'Left'
    csvWriter.writerow([info.imageName, view])
    info.view = view
    info.useFlag = True

def classify_2D_view(info, csvWriter):
    oriH, oriW = len(info.grayData), len(info.grayData[0])
    resizeScale = 0.1
    grayDataSmall = cv2.resize(info.grayData, dsize=(int(oriW * resizeScale), int(oriH * resizeScale)),
                               interpolation=cv2.INTER_CUBIC)
    h, w = len(grayDataSmall), len(grayDataSmall[0])
    w3, leftCnt, rightCnt = int(w / 3), 0, 0
    for row in range(h):
        leftCnt += sum(grayDataSmall[row][:w3])
        rightCnt += sum(grayDataSmall[row][w3 + w3:w])
    leftAverage = leftCnt / (h * w3)
    rightAverage = rightCnt / (h * (w - w3 - w3))
    view = 'Left' if leftAverage > rightAverage else 'Right'
    csvWriter.writerow([info.imageName, view])
    info.view = view
    info.useFlag = True

def main():
    model, device = initialize_system()

    root = 'dataset'
    samdir = 'Sample'
    imgs = load_images(root, samdir)

    process_images(imgs, model, device)

if __name__ == "__main__":
    main()