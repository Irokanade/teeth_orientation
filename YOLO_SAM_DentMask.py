import os
import torch
import pickle
import time

from tqdm import tqdm
from ultralytics import YOLO


import PIL.Image
import PIL.ImageOps
from PIL import Image, ImageDraw, ImageFont

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
###### 標號 ######
import csv
import cv2
import json
###### SAM #####
import sys

sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

'''def tensor2image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor.permute(1, 2, 0), dtype=np.uint8)
    return Image.fromarray(tensor)'''

def viewOfficial(view):
    if view == 'Up':
        return 'upper_occlusal'
    elif view == 'Below':
        return 'lower_occlusal'
    elif view == 'Left':
        return 'left_buccal'
    elif view == 'Right':
        return 'right_buccal'
    elif view == 'Face':
        return 'frontal'
    else:
        return 'Unknown'

def absoluteToScale(teethLocation,imageWidth,imageHeight):
    # x,y,w,h
    x1 = teethLocation.x1
    x2 = teethLocation.x2
    y1 = teethLocation.y1
    y2 = teethLocation.y2
    return [ ((x1+x2)/2)/imageWidth,((y1+y2)/2)/imageHeight,((x2-x1))/imageWidth,((y2-y1))/imageHeight ]

'''def scaleToAbsolute(teethScale,imageWidth,imageHeight):
    scaleX = teethScale[0]
    scaleY = teethScale[1]
    scaleW = teethScale[2]
    scaleH = teethScale[3]

    x = scaleX*imageWidth
    y = scaleY*imageHeight
    w = scaleW*imageWidth
    h = scaleH*imageHeight

    return TeethLocation(x-(w/2),y-(h/2),x+(w/2),y+(h/2))'''

def edge_detection(info):
    gray = cv2.cvtColor(info.image, cv2.COLOR_BGR2GRAY) if len(info.image.shape) == 3 else info.image

    h, w = gray.shape
    x_start = int(w * 0.3)   # Left 30%
    x_end = int(w * 0.67)    # Right 67%
    y_start = int(h * 0.20)  # Top 20%
    y_end = int(h * 0.92)    # Bottom 92%

    cropped_gray = gray[y_start:y_end, x_start:x_end]

    edges = cv2.Canny(cropped_gray, 50, 150)
    edge_count = cv2.countNonZero(edges)

    return edge_count

def get_teeth_area(teeth):
    return abs(teeth.y2 - teeth.y1) * abs(teeth.x2 - teeth.x1)

def frontal_rotate(info):
    middle_teeth = sorted(info.teethLocationSet, key=lambda t: (t.x1 + t.x2) / 2)[int(len(info.teethLocationSet)/2) - 4:int(len(info.teethLocationSet)/2) + 4]

    top_4_teeth = middle_teeth[:4]
    bottom_4_teeth = middle_teeth[4:]

    top_4_avg_area = sum(get_teeth_area(teeth) for teeth in top_4_teeth) / 4
    bottom_4_avg_area = sum(get_teeth_area(teeth) for teeth in bottom_4_teeth) / 4

    if bottom_4_avg_area > top_4_avg_area:
        info.image = np.rot90(info.image, 2)

    return info

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
        self.teethScaleSet = [absoluteToScale(teethLocation,width,height) for teethLocation in teethLocationSet] #x,y,w,h
        self.view = 'Unknown'
        self.teethNodeSet = imageTeethNodeSet
        self.polyLine = polyLine
        #### mark ####
        self.regression = p
        #### edge detection ####
        self.edge_count = edge_detection(self)

class TeethNode:
    def __init__(self,mask,box):
        self.mask = mask.astype(np.uint8)
        self.box = box
        self.labelId = -1

    def dump(self):
        return [self.mask, self.box, self.labelId]

    @staticmethod
    def build_teethNode(attributes):
        return TeethNode(*attributes)

class TeethLocation:
    def __init__(self,x1,y1,x2,y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

# class NumpyEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return json.JSONEncoder.default(self, obj)

def exif_transpose(img):
    if not img:
        return img

    exif_orientation_tag = 274

    # Check for EXIF data (only present on some files)
    if hasattr(img, "_getexif") and isinstance(img._getexif(), dict) and exif_orientation_tag in img._getexif():
        exif_data = img._getexif()
        orientation = exif_data[exif_orientation_tag]

        # Handle EXIF Orientation
        if orientation == 1:
            # Normal image - nothing to do!
            pass
        elif orientation == 2:
            # Mirrored left to right
            img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            # Rotated 180 degrees
            img = img.rotate(180)
        elif orientation == 4:
            # Mirrored top to bottom
            img = img.rotate(180).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 5:
            # Mirrored along top-left diagonal
            img = img.rotate(-90, expand=True).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 6:
            # Rotated 90 degrees
            img = img.rotate(-90, expand=True)
        elif orientation == 7:
            # Mirrored along top-right diagonal
            img = img.rotate(90, expand=True).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 8:
            # Rotated 270 degrees
            img = img.rotate(90, expand=True)

    return img

'''def to_serializable(val):
    if hasattr(val, '__dict__'):
        return val.__dict__
    elif hasattr(val, "tolist"):
        return val.tolist()
    return val'''

def sortGradient(infoSet):
    infoSet = sorted(infoSet,key = lambda x : x.absGradient, reverse = True)
    for index, info in enumerate(infoSet):
        info.gradientRank = index + 1


def sortTeeth(infoSet):
    infoSet = sorted(infoSet, key = lambda x : x.teethNum, reverse = True)
    for index, info in enumerate(infoSet):
        info.teethRank = index + 1

def checkFlag3D(oriPltImage,resizeScale,proportion):
    pltImage = cv2.resize(oriPltImage, dsize=(int(imageWidth*resizeScale),int(imageHeight*resizeScale)), interpolation=cv2.INTER_CUBIC)
    hRGB = len(pltImage)
    wRGB = len(pltImage[0])
    totalPixel = hRGB * wRGB


    white = (255,255,255)
    mask = cv2.inRange(pltImage,white,white)
    whitePixelCnt = cv2.countNonZero(mask)

    # for row in range(hRGB):
    #     for col in range(wRGB):
    #         whiteFlag = True
    #         for i in range(3):
    #             if pltImage[row][col][i] != 255:
    #                 whiteFlag = False
    #                 break
    #         if whiteFlag == True:
    #             whitePixelCnt += 1
    if whitePixelCnt  < totalPixel*proportion :
        return False
    else:
        return True

'''def findTeethScaleByName(imageInfoSet,imgName):
    for imgInfo in imageInfoSet:
        if imgInfo.imageName == imgName:
            return imgInfo.teethScaleSet'''

'''def findInfoByName(imageInfoSet,imgName):
    for imgInfo in imageInfoSet:
        if imgInfo.imageName == imgName:
            return imgInfo'''

'''def teethLocScale(teethLoc,scale):
    teethWidth  = teethLoc.x2 - teethLoc.x1
    teethHeight = teethLoc.y2 - teethLoc.y1
    xMiddle = (teethLoc.x2 + teethLoc.x1)/2
    yMiddle = (teethLoc.y2 + teethLoc.y1)/2
    teethWidth = teethWidth * scale
    teethHeight = teethHeight * scale
    return TeethLocation(xMiddle-teethWidth/2,yMiddle-teethHeight/2,xMiddle+teethWidth/2,yMiddle+teethHeight/2)'''

'''def absoluteXYWHtoLoc(x,y,w,h):
    return TeethLocation(x-w/2,y-h/2,x+w/2,y+h/2)'''

'''def countIouScale(boxA,boxB,scale):
    return countIou(teethLocScale(boxA,scale),teethLocScale(boxB,scale))'''

'''def countIou(boxA,boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA.x1, boxB.x1)
	yA = max(boxA.y1, boxB.y1)
	xB = min(boxA.x2, boxB.x2)
	yB = min(boxA.y2, boxB.y2)
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA.x2 - boxA.x1 + 1) * (boxA.y2 - boxA.y1 + 1)
	boxBArea = (boxB.x2 - boxB.x1 + 1) * (boxB.y2 - boxB.y1 + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou'''

# def labelToImageInfo(baseTeethNodeSet,fillTeethNodeSet):
#     #base為要更新的class
#     #fill為前面記錄的,之後labelId要填入class裡
#     for fill in fillTeethNodeSet:
#         for base in baseTeethNodeSet:
#             if fill.box == base.box:
#                 base.labelId = fill.labelId

'''def doubleCheckPivot(missingLabelId,imageInfo):

    #print('double check pivot')
    #print('double missing',missingLabelId)
    leftDimension = -1
    rightDimension = -1
    if imageInfo.view == 'Up':
        leftDimension = 1
        rightDimension = 2
    elif imageInfo.view == 'Below':
        leftDimension = 4
        rightDimension = 3

    checkTeethNum = 4

    for checkLabelId in range(leftDimension*10+1,leftDimension*10+1+checkTeethNum ): #left 象限1/4
        for lableId in missingLabelId:
            if lableId == checkLabelId:
                return False
    for checkLabelId in range(rightDimension*10+1,rightDimension*10+1+checkTeethNum ): #right 象限2/3
        for lableId in missingLabelId:
            if lableId == checkLabelId:
                return False
    return True'''

'''def labelOffset(teethNodeSet,offset):
    for teethNode in teethNodeSet:
        teethNode.labelId += offset'''

def boxMiddle(box):
    return (box.x1 + box.x2)/2 , (box.y1 + box.y2)/2

'''def missingLabel(view,leftTeethNodeSet,rightTeethNodeSet):
    missingLabelId = []

    leftDimension = -1
    rightDimension = -1
    if view == 'Up':  #up面觀存在1/2象限
        leftDimension = 1
        rightDimension = 2
    elif view == 'Below':
        leftDimension = 3
        rightDimension = 4

    for dimension in range(leftDimension, rightDimension+1):
        for checkLabel in range( dimension*10+1,dimension*10+1+8 ):
            check = False
            for teethNode in leftTeethNodeSet:
                if teethNode.labelId == checkLabel:
                    check = True
                    break
            for teethNode in rightTeethNodeSet:
                if teethNode.labelId == checkLabel:
                    check = True
                    break
            if check == False:
                missingLabelId.append(checkLabel)
    return missingLabelId'''

'''def extract_box_brightness(image, box):
    x1 = max(int(box.x1), 0)
    y1 = max(int(box.y1), 0)
    x2 = min(int(box.x2), image.shape[1])
    y2 = min(int(box.y2), image.shape[0])
    #print('box = ',x1,y1,x2,y2)
    bounding_box_region = image[y1:y2, x1:x2, :]

    #print(bounding_box_region.shape)

    brightness = np.mean(bounding_box_region)
    return brightness'''

'''def extract_mask_brightness(image, mask):
    masked_pixels = image[np.where(mask > 0)]

    brightness = np.mean(masked_pixels)

    return brightness'''


'''def check5Missing(teethNodeSet,sixBoxes):
    iouThreshold = 0.8
    for i in range(len(teethNodeSet)):
        if teethNodeSet[i].labelId % 10 == 5: #check 5 missing
            for sixBox in sixBoxes:
                if countIou(teethNodeSet[i].box,sixBox) > iouThreshold:
                    dimension = teethNodeSet[i].labelId//10
                    for k in range(len(teethNodeSet)):
                        if teethNodeSet[k].labelId // 10 == dimension and teethNodeSet[k].labelId%10 >= 5:
                            teethNodeSet[k].labelId += 1
                    print('5Missing!!!',file = specialLogFile)
                    break
    return teethNodeSet'''


'''def doubleCheckPosition(imageInfo,leftTeethNodeSet,rightTeethNodeSet):
    targetOffset = -1
    checkNum = 3
    minDev = 1e9
    for offset in range(-1,1+1):
        positionTeethNodeSet = leftTeethNodeSet[:checkNum-offset] + rightTeethNodeSet[:checkNum+offset]
        dev = np.std(  list(map( lambda teethNode : ( (teethNode.box.y1+teethNode.box.y2)/2.0 ),positionTeethNodeSet)) )
        if dev < minDev:
            minDev = dev
            targetOffset = offset
    copyLeft = leftTeethNodeSet[:]
    copyRight = rightTeethNodeSet[:]
    if targetOffset == -1:
        if imageInfo.view == 'Up':
            copyLeft[0].labelId = 20
        elif imageInfo.view == 'Below':
            copyLeft[0].labelId = 30
        leftTeethNodeSet  = copyLeft[1:]
        rightTeethNodeSet = copyLeft[:1]+copyRight[:]
        labelOffset(leftTeethNodeSet, -1)
        labelOffset(rightTeethNodeSet, 1)
    elif targetOffset == 1:
        if imageInfo.view == 'Up':
            copyRight[0].labelId = 10
        elif imageInfo.view == 'Below':
            copyRight[0].labelId = 40
        leftTeethNodeSet  = copyRight[:1]+copyLeft[:]
        rightTeethNodeSet = copyRight[1:]
        labelOffset(leftTeethNodeSet, 1)
        labelOffset(rightTeethNodeSet, -1)


    #print('offset = ',targetOffset)
    #print('after = ',leftTeethNodeSet)
    #print('after = ',rightTeethNodeSet)'''

'''def slidingTeeth(regression,baseTeethLoc,slidingOverlapRatio,xStep,state):
    teethWidth  = (baseTeethLoc.x2 - baseTeethLoc.x1)
    teethHeight = (baseTeethLoc.y2 - baseTeethLoc.y1)
    Xstart = int((baseTeethLoc.x2 + baseTeethLoc.x1)/2)
    while countIou( absoluteXYWHtoLoc(Xstart,regression(Xstart),teethWidth,teethHeight),baseTeethLoc ) > slidingOverlapRatio :
        if state == 'Left':
            Xstart -= xStep
        else:
            Xstart += xStep
    return absoluteXYWHtoLoc(Xstart,regression(Xstart),teethWidth,teethHeight)'''

'''def teethMatch(baseTeethLoc,teethNodeSet,dimension,xStep,slidingOverlapRatio,teethOverlapRatio,state,teethScaleRatio,xAverage,yAverage,imageInfo):
    #print("wow!!!!!!!!!!!!!!!!!",dimension)
    missingLabelId = []
    if len(teethNodeSet) == 0:
        for labelId in range(dimension*10+1,dimension*10+1+8):
            missingLabelId.append(labelId)
        return missingLabelId,teethNodeSet

    brightDiffLock = 70
    #print('image Name = ',imageInfo.imageName)
    for labelId in range(dimension*10+1,dimension*10+1+8+(1)):

        maxIou = 0
        matchTeethNode = teethNodeSet[0]
        for teethNode in teethNodeSet:
            if  teethNode.labelId == -1 and countIouScale(baseTeethLoc,teethNode.box,teethScaleRatio) > maxIou :
                if labelId%10>=6 and extract_mask_brightness(imageInfo.image,teethNode.mask)-extract_box_brightness(imageInfo.image,baseTeethLoc)>brightDiffLock:
                    print('Handle BRIGHT DETECT ERROR !!  =>',imageInfo.imageName)
                else:
                    maxIou = countIouScale(baseTeethLoc,teethNode.box,teethScaleRatio)
                    matchTeethNode = teethNode

        #print("maxIou!! ", maxIou)

        if maxIou > teethOverlapRatio: #match next teeth
            matchTeethNode.labelId = labelId
            baseTeethLoc = matchTeethNode.box
        else: #lack of teeth
            missingLabelId.append(labelId)
            if labelId%10 >= 6:
                molarEnlargeRatio = 1.2
                baseTeethLoc = absoluteXYWHtoLoc((baseTeethLoc.x1 + baseTeethLoc.x2)/2,(baseTeethLoc.y1 + baseTeethLoc.y2)/2,xAverage*molarEnlargeRatio,yAverage*molarEnlargeRatio)
            baseTeethLoc = slidingTeeth(imageInfo.regression,baseTeethLoc,slidingOverlapRatio,xStep,state) #移動baseLoc到剛好slidingOverlapRatio Iou比例

    return missingLabelId,teethNodeSet'''



'''def positionMissing(imageInfo):
    leftTeethNodeSet = [] #第1/4象限
    rightTeethNodeSet = [] #第2/3象限
    missingLabelId = [] 

    teethOverlapRatio = 0.0 #每顆牙要重疊IOU比例
    slidingOverlapRatio = 0.09 #每顆牙滑動必須重疊IOU比例
    teethScaleRatio = 1.12 #每顆牙齒放大比例,算有無重疊用
    xStep = 3

    teethNodeSet = imageInfo.teethNodeSet

    xAverage = (sum((teethNode.box.x2-teethNode.box.x1) for teethNode in teethNodeSet)/len(teethNodeSet))
    yAverage = (sum((teethNode.box.y2-teethNode.box.y1) for teethNode in teethNodeSet)/len(teethNodeSet))

    # ax^2 + bx + c = 0
    a = imageInfo.polyLine[0]
    b = imageInfo.polyLine[1]
    c = imageInfo.polyLine[2]

    xVertex = (-1.0)*( b/(2*a) )
    yVertex = ((4.0*a*c)-(b*b))/(4.0*a)

    for teethNode in teethNodeSet:
        Xmiddle = (teethNode.box.x1 + teethNode.box.x2)/2.0
        if Xmiddle < xVertex:
            leftTeethNodeSet.append(teethNode)
        else:
            rightTeethNodeSet.append(teethNode)


    leftDimension = -1
    rightDimension = -1
    #前三個用x排序，剩下用y排序
    if imageInfo.view == 'Up':
        leftDimension = 1
        rightDimension = 2
        leftTeethNodeSet  = sorted(leftTeethNodeSet ,key = lambda x : ((x.box.y2+x.box.y1)/2.0),reverse = False)
        rightTeethNodeSet = sorted(rightTeethNodeSet,key = lambda x : ((x.box.y2+x.box.y1)/2.0),reverse = False)
    elif imageInfo.view == 'Below':
        leftDimension = 4
        rightDimension = 3
        leftTeethNodeSet  = sorted(leftTeethNodeSet ,key = lambda x : ((x.box.y2+x.box.y1)/2.0),reverse = True)
        rightTeethNodeSet = sorted(rightTeethNodeSet,key = lambda x : ((x.box.y2+x.box.y1)/2.0),reverse = True)

    leftTeethNodeSet[:3]  = sorted(leftTeethNodeSet[0:3]  ,key = lambda x : ((x.box.x2+x.box.x1)/2.0),reverse = True)
    rightTeethNodeSet[:3] = sorted(rightTeethNodeSet[0:3] ,key = lambda x : ((x.box.x2+x.box.x1)/2.0),reverse = False)

    baseTeethLoc = TeethLocation(xVertex-xAverage/2,yVertex-yAverage/2,xVertex+xAverage/2,yVertex+ yAverage/2)

    missingTmp,leftTeethNodeSet = teethMatch(baseTeethLoc,leftTeethNodeSet,leftDimension,xStep,slidingOverlapRatio,teethOverlapRatio,'Left',teethScaleRatio,xAverage,yAverage,imageInfo)
    missingLabelId += missingTmp

    baseTeethLoc = TeethLocation(xVertex-xAverage/2,yVertex-yAverage/2,xVertex+xAverage/2,yVertex+ yAverage/2)

    missingTmp,rightTeethNodeSet = teethMatch(baseTeethLoc,rightTeethNodeSet,rightDimension,xStep,slidingOverlapRatio,teethOverlapRatio,'Right',teethScaleRatio,xAverage,yAverage,imageInfo)
    missingLabelId += missingTmp

    return missingLabelId,leftTeethNodeSet,rightTeethNodeSet'''

'''def XIou(boxA,boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA.x1, boxB.x1)
	xB = min(boxA.x2, boxB.x2)
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA.x2 - boxA.x1 + 1)
	boxBArea = (boxB.x2 - boxB.x1 + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou'''

'''def YIou(boxA,boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
	yA = max(boxA.y1, boxB.y1)
	yB = min(boxA.y2, boxB.y2)
	# compute the area of intersection rectangle
	interArea = max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA.y2 - boxA.y1 + 1)
	boxBArea = (boxB.y2 - boxB.y1 + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou'''


'''def checkXYOverlap(teethNodeSet,XOverlapRatio,YOverlapRatio): #X重疊度超過overlapRatio 回傳False
    teethNodeSetTmp = teethNodeSet[:]
    teethNodeSetTmp = sorted(teethNodeSetTmp ,key = lambda x : ((x.box.x2+x.box.x1)/2.0),reverse = False)

    for i in range(len(teethNodeSetTmp)-1):
        if XIou(teethNodeSetTmp[i].box,teethNodeSetTmp[i+1].box) >= XOverlapRatio:
            return False
        if YIou(teethNodeSetTmp[i].box,teethNodeSetTmp[i+1].box) < YOverlapRatio:
            return False
    return True'''

'''def teethNodeCombine(teethNodeSet,i1,i2):
    # 取得要合併的兩個 TeethNode
    node1 = teethNodeSet[i1]
    node2 = teethNodeSet[i2]

    # 合併 mask

    # dilate
    kernel = np.ones( (3,3), np.uint8 )
    node1.mask = cv2.dilate(node1.mask, kernel, iterations = 10)
    node2.mask = cv2.dilate(node2.mask, kernel, iterations = 10)

    #merge
    mergedMask = node1.mask | node2.mask

    #erode
    mergedMask = cv2.erode(mergedMask, kernel, iterations = 10)


    # 合併 bounding box
    points = cv2.findNonZero(mergedMask)

    x,y,w,h = cv2.boundingRect(points)
    xmin = int(x)
    xmax = int(x+w)
    ymin = int(y)
    ymax = int(y+h)
    mergedBox = TeethLocation(xmin, ymin, xmax, ymax)

    # 建立新的合併後的 TeethNode
    mergedNode = TeethNode(mergedMask, mergedBox)
    mergedNode.labelId = node1.labelId

    # 刪除原本的兩個節點
    if i1 > i2:
        del teethNodeSet[i1]
        del teethNodeSet[i2]
    else:
        del teethNodeSet[i2]
        del teethNodeSet[i1]

    # 將合併後的節點加入到列表中
    teethNodeSet.append(mergedNode)

    return teethNodeSet'''

'''def toothErrorDetectionCombine(teethNodeSet,XOverlapRatio):
    lock = True
    while lock:
        lock = False
        n = len(teethNodeSet)
        for i in range(n):
            for k in range(i+1,n):
                if XIou(teethNodeSet[i].box,teethNodeSet[k].box) >= XOverlapRatio:
                    # print(XIou(teethNodeSet[i].box,teethNodeSet[k].box))
                    # cv2.imshow('mask ',teethNodeSet[i].mask)
                    # cv2.waitKey()
                    # cv2.imshow('mask2 ',teethNodeSet[k].mask)
                    # cv2.waitKey()
                    print('Error Dection Combine Teeth!!', file=specialLogFile)
                    #print('ori len = ',len(teethNodeSet))
                    teethNodeSet = teethNodeCombine(teethNodeSet,i,k)
                    #print('after len = ',len(teethNodeSet))
                    lock = True
                    break
            if lock :
                break
    return teethNodeSet'''


'''def leftRightLabel(imageInfo):
    teethNodeSet = imageInfo.teethNodeSet
    teethNodeSet = sorted(teethNodeSet ,key = lambda x : ((x.box.y2+x.box.y1)/2.0),reverse = False)
    minDev = 1e9
    upTeethNodeSet = []
    belowTeethNodeSet = []
    XOverlapRatio = 0.3 # 同排牙齒的X重疊程度，必須小於XoverlapRatio
    YOverlapRatio = 0.2 # 同排牙齒的Y重疊程度，必須大於YoverlapRatio
    for upNum in range(1,min(8+1,len(teethNodeSet))):
        upTeethNodeSetTmp = teethNodeSet[:upNum]
        belowTeethNodeSetTmp = teethNodeSet[upNum:]
        upDev = np.std(  list(map( lambda teethNode : ( (teethNode.box.y1+teethNode.box.y2)/2.0 ),upTeethNodeSetTmp)) )
        belowDev = np.std(  list(map( lambda teethNode : ( (teethNode.box.y1+teethNode.box.y2)/2.0 ),belowTeethNodeSetTmp)) )

        if upDev + belowDev < minDev and checkXYOverlap(upTeethNodeSetTmp,XOverlapRatio,YOverlapRatio) and checkXYOverlap(belowTeethNodeSetTmp,XOverlapRatio,YOverlapRatio):
            #print('use XY')
            minDev = upDev + belowDev
            upTeethNodeSet = upTeethNodeSetTmp
            belowTeethNodeSet = belowTeethNodeSetTmp

    if len(upTeethNodeSet)==0 and len(belowTeethNodeSet)==0: #改用迴歸直線分上下界線
        #print('use regreesion')
        for teethNode in teethNodeSet:
            if (teethNode.box.y1+teethNode.box.y2)/2 < imageInfo.regression((teethNode.box.x1+teethNode.box.x2)/2):
                upTeethNodeSet.append(teethNode)
            else:
                belowTeethNodeSet.append(teethNode)


    #牙齒跟牙根偵測錯誤(分離)，形成疊羅漢，同排必定X重疊程度<XoverlapRatio
    upTeethNodeSet = toothErrorDetectionCombine(upTeethNodeSet,XOverlapRatio)
    belowTeethNodeSet = toothErrorDetectionCombine(belowTeethNodeSet,XOverlapRatio)
    #####

    upDimension = -1
    belowDimension = -1
    if imageInfo.view == 'Left':
        upDimension = 2
        belowDimension = 3
        upTeethNodeSet  = sorted(upTeethNodeSet ,key = lambda x : ((x.box.x2+x.box.x1)/2.0),reverse = True)
        belowTeethNodeSet = sorted(belowTeethNodeSet,key = lambda x : ((x.box.x2+x.box.x1)/2.0),reverse = True)
    elif imageInfo.view == 'Right':
        upDimension = 1
        belowDimension = 4
        upTeethNodeSet  = sorted(upTeethNodeSet ,key = lambda x : ((x.box.x2+x.box.x1)/2.0),reverse = False)
        belowTeethNodeSet = sorted(belowTeethNodeSet,key = lambda x : ((x.box.x2+x.box.x1)/2.0),reverse = False)

    upIndex = 0
    for labelId in range(upDimension*10 + 8, upDimension*10+1 -1, -1): #up 象限1/2
        if upIndex >= len(upTeethNodeSet):
            break
        if(imageFileInfo.missingLabelId.count(labelId) == 1):
            #print(str(labelId)+"miss!")
            nothing = 'nothing'
        else:
            upTeethNodeSet[ upIndex ].labelId = labelId
            upIndex += 1

    belowIndex = 0
    for labelId in range(belowDimension*10 + 8, belowDimension*10+1 -1, -1): #below 象限4/3
        if belowIndex >= len(belowTeethNodeSet):
            break
        if(imageFileInfo.missingLabelId.count(labelId) == 1):
            #print(str(labelId)+"miss!")
            nothing = 'nothing'
        else:
            belowTeethNodeSet[ belowIndex ].labelId = labelId
            belowIndex += 1

    return upTeethNodeSet,belowTeethNodeSet'''

def pilSave(image,path,fileLabel,prefix,imageName):
    check = imageName.split('.')
    if prefix != "" :
        prefix += "_"
    if len(check[-1]) < 4 :
        image.save(f"{path}/{fileLabel}/{prefix}{imageName[:-3]}png")
    else:  #jpeg
        image.save(f"{path}/{fileLabel}/{prefix}{imageName[:-4]}png")

def pltSave(path,fileLabel,prefix,imageName):
    check = imageName.split('.')
    if prefix != "" :
        prefix += "_"
    if len(check[-1]) < 4 :
        plt.savefig(f"{path}/{fileLabel}/{prefix}{imageName[:-3]}png")
    else:  #jpeg
        plt.savefig(f"{path}/{fileLabel}/{prefix}{imageName[:-4]}png")

def makeResultDirProcess(fileName):
    redir = 'result'
    sampleDir = 'sample'
    file = os.path.join(redir)
    os.makedirs(file,exist_ok=True)
    file = os.path.join(sampleDir)
    os.makedirs(file,exist_ok=True)
    file = os.path.join(redir,  fileName)
    os.makedirs(file,exist_ok=True)
    file = os.path.join(sampleDir,  fileName)
    os.makedirs(file,exist_ok=True)
    createFile = ['regression'] #'det','seg','color','pre_processing','changeColor','mask','newNameSample', ,'sample','boundingBox', 'node'
    for name in createFile:
        tmp = os.path.join(redir, fileName, name)
        os.makedirs(tmp,exist_ok=True)

'''def getBoundingBoxes(model,threshold,imageInfo):
    torchImage = torch.as_tensor(np.array(imageInfo.image)[...,:3]/255, dtype=torch.float32).permute(2,0,1).unsqueeze(0)
    torchImage = torchImage.to(device)
    output = model(torchImage)[0]

    scores = output["scores"]
    boxes = output["boxes"]
    masks = output["masks"]
    classes = output["labels"]
    zippedData = zip(boxes,masks, scores,classes)
    zippedData = sorted(zippedData,key=lambda x:x[2],reverse=True)
    retBoxes = []

    leftBoxes = []
    rightBoxes = []
    leftScores = []
    rightScores = []
    for box,mask, score,label in zippedData:
        #print(score.item())
        if score.item() > threshold:
            box = [b.item() for b in box]
            x1, y1, x2 ,y2 = box
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            if (x1+x2)/2 < (imageInfo.width/2): #left
                leftBoxes.append(TeethLocation(x1,y1,x2,y2))
                leftScores.append(score.item())
            else:
                rightBoxes.append(TeethLocation(x1,y1,x2,y2))
                rightScores.append(score.item())


    #找左右分數最大，各一顆
    if len(leftBoxes) > 0:
        retBoxes.append(  (max(zip(leftBoxes,leftScores), key=lambda box: box[1] ))[0]  )
    if len(rightBoxes) > 0:
        retBoxes.append(  (max(zip(rightBoxes,rightScores), key=lambda box: box[1]))[0] )
    drawImage = Image.fromarray(imageInfo.image.copy())
    draw = ImageDraw.Draw(drawImage)
    for box in retBoxes:
        x1 = box.x1
        x2 = box.x2
        y1 = box.y1
        y2 = box.y2
        detLineScale = 0.005 #det line width scale
        draw.line([(x1,y1),(x2,y1),(x2,y2),(x1,y2),(x1,y1)], fill=color, width=int(imageInfo.width*detLineScale))

    pilSave(drawImage,f"./result/{fileName}","det","det_upLowerSix",imageInfo.imageName)

    return retBoxes'''

def checkClassication(leftCnt, rightCnt , imageInfoSet):
    if leftCnt > 1 or rightCnt > 1:
        print('classification ERROR',file=specialLogFile)
        for imageInfo in imageInfoSet:  # force modify
            if imageInfo.view == 'Left':
                imageInfo.view = 'Right'
                return
            elif imageInfo.view == 'Right':
                imageInfo.view = 'Left'
                return

def constructDimensionDict(teethNodeSet):
    ret = {}
    for teethNode in teethNodeSet:
        if teethNode.labelId in ret:
            ret[teethNode.labelId].append(teethNode)
        else:
            ret[teethNode.labelId] = [teethNode]
    return ret

### SAM ###
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

'''def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25) '''

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def SAM(image, bbox):
    sam_checkpoint = "./ckpts/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    #x = torch.ones(1, device=device)
    x = torch.ones(1, device=device)
    print (x)

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    predictor.set_image(image)

    input_boxes = torch.tensor(bbox, device=predictor.device)

    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    masks.shape

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.close()
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box in input_boxes:
        show_box(box.cpu().numpy(), plt.gca())

    return masks
### SAM ###

if __name__ == "__main__":
    print( time.ctime() )
    print("Cuda is available = ",torch.cuda.is_available())
    print("Cuda version = ",torch.version.cuda)
    print("pytorch version = ",torch.__version__)
    isLabel = False
    isCuda = False
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        isCuda = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        isCuda = True
    else:
        device = torch.device("cpu")
    print("device -> ",device)
    model = YOLO('./ckpts/8Class_best.pt')



    # Test all image | path setting
    # load every file in ./dataset/Sample
    root = 'dataset'
    samdir = 'sample'

    # load direction
    imgdir = list(sorted(os.listdir(os.path.join(root, samdir))))
    imgs = {}

    print(imgdir)

    for fileName in imgdir:
        imgSet = list(sorted(os.listdir(os.path.join(root, samdir, fileName))))
        print(imgSet)

        imgs[ fileName ] = imgSet

    # confident threshold
    confident_thre = 0.60

    # put every image in path and show result in corresponding folder
    # det means detection, seg means segmentation
    pbar = tqdm( total = 100 )
    print( "\nfileNum = ", len(imgs) )

    with open('specialLog.log','w') as specialLogFile :
        for fileName,imgSet in imgs.items():
            print('\n\nfileName : ',fileName, file = specialLogFile)
            print( '\nProcess : ',fileName )
            makeResultDirProcess(fileName)

            imageFileInfo = ImageFile(fileName)

            imageInfoSet: list[PhotoImage] = []
            labelLoc = {}

            for imageName in imgSet:
                print(imageName,' => DETECT teeth ')
                image_path = os.path.join(root, samdir, fileName, imageName)
                image = Image.open(image_path)
                image = exif_transpose(image)

                # Rotate the image if height is greater than width
                if image.height > image.width:
                    image = image.rotate(90, expand=True)

                # Mirror flip all image
                image = image.transpose(Image.FLIP_LEFT_RIGHT)

                results = list(model(image, conf=confident_thre))
                result = results[0]
                pltImage = np.array(image)
                if result.masks is None:
                    continue

                pltImage = cv2.resize(pltImage, result.masks.data.shape[1:][::-1], interpolation=cv2.INTER_AREA)
                image = Image.fromarray(np.uint8(pltImage))
                #pilSave(image,f"./result/{fileName}","sample","",imageName)

                imageWidth = len(pltImage[0])
                imageHeight = len(pltImage)

                img = image.copy()
                black_img = Image.new('RGB', img.size, (0, 0, 0))
                draw = ImageDraw.Draw(img)

                masks = result.masks.data.cpu().numpy()
                boxes = result.boxes.cpu().numpy()

                bbox = []

                # processing bounding box
                xList = []
                yList = []

                ### SAM ###
                resize = 0.002
                
                # write to boundingBox folder
                '''
                bbox_file_path = f"./result/{fileName}/boundingBox/box_{imageName[:-4]}.csv"
                with open(bbox_file_path, "w", newline="") as bboxFile:
                    #csvWriter = csv.writer(bboxFile)
                '''
                #print( "length of boxes = ", len(boxes) )
                maxX, maxY = 0, 0
                minX, minY = img.width, img.height
                for box, mask in zip(boxes, masks):
                    points = cv2.findNonZero(mask)
                    
                    x,y,w,h = cv2.boundingRect(points)
                    x1 = int(x)
                    x2 = int(x+w)
                    y1 = int(y)
                    y2 = int(y+h)
                    one_box = [x1,y1,x2,y2]
                    bbox.append(one_box)
                    if x2 > maxX:
                        maxX = x2
                    if x1 < minX:
                        minX = x1
                    if y2 > maxY:
                        maxY = y2
                    if y1 < minY:
                        minY = y1

                    #csvWriter.writerow([x1, y1, x2, y2]) 
                    xList.append( (x1+x2)/2 )
                    yList.append( (y1+y2)/2 )

                #if imageName == 'DSCF3120.JPG':
                #    print( minX, ", ", minY, ", ", maxX, ", ", maxY )

                xArray = np.array(xList)
                yArray = np.array(yList)
                polyLine = np.polyfit(xArray,yArray,2)
                p = np.poly1d( polyLine )
                relx = 0
                for i in range(len(xList)):
                    relx += (abs(p(xList[i])-yList[i]))
                relx = relx / len(xList)
                polyLine = np.polyfit(yArray,xArray,2)
                p = np.poly1d( polyLine )
                rely = 0
                for i in range(len(xList)):
                    rely += (abs(p(yList[i])-xList[i]))
                rely = rely / len(yList)
                if relx > rely:
                    # Mirror flip back
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    for i in range(len(xList)):
                        xList[i] = image.width - 1 - xList[i]
                        bbox[i] = [image.width-1-bbox[i][2], bbox[i][1], image.width-1-bbox[i][0], bbox[i][3]]
                    '''for mask in masks:
                        new_msk = [[] for _ in range(image.height)]
                        for i in range( image.height ):
                            for j in range( image.width ):
                                new_msk[i].append(mask[i][image.width-1-j])
                        mask = new_msk'''
                    tmp = maxX
                    maxX = image.width - minX
                    minX = image.width - tmp

                    #if imageName == 'DSCF3120.JPG':
                    #    print( minX, ", ", minY, ", ", maxX, ", ", maxY )
                    
                    # Rotate the image if height is greater than width
                    image = image.rotate(90, expand=True)
                    for i in range(len(xList)):
                        tmp = xList[i]
                        xList[i] = image.width - 1 - yList[i]
                        yList[i] = tmp
                        x1, y1, x2, y2 = bbox[i][0], bbox[i][1], bbox[i][2], bbox[i][3]
                        bbox[i] = [y1, image.height-1-x2, y2, image.height-1-x1]
                    '''for mask in masks:
                        new_msk = [[] for _ in range(image.height)]
                        for i in range( image.height ):
                            for j in range( image.width ):
                                new_msk[i].append(mask[image.width-1-j][i])
                        mask = new_msk'''
                    tmp1, tmp2 = maxY, minY
                    minY = image.height - maxX
                    maxY = image.height - minX
                    minX = tmp2
                    maxX = tmp1

                    #if imageName == 'DSCF3120.JPG':
                    #    print( minX, ", ", minY, ", ", maxX, ", ", maxY )

                    # Mirror flip back
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    for i in range(len(xList)):
                        xList[i] = image.width - 1 - xList[i]
                        bbox[i] = [image.width-1-bbox[i][2], bbox[i][1], image.width-1-bbox[i][0], bbox[i][3]]
                    '''for mask in masks:
                        new_msk = [[] for _ in range(image.height)]
                        for i in range( image.height ):
                            for j in range( image.width ):
                                new_msk[i].append(mask[i][image.width-1-j])
                        mask = new_msk'''
                    tmp = maxX
                    maxX = image.width - minX
                    minX = image.width - tmp

                    #if imageName == 'DSCF3120.JPG':
                    #    print( minX, ", ", minY, ", ", maxX, ", ", maxY )

                    imageWidth = image.width
                    imageHeight = image.height
                
                xArray = np.array(xList)
                yArray = np.array(yList)
                polyLine = np.polyfit(xArray,yArray,2)
                p = np.poly1d( polyLine )
                curve = []
                for x in xList:
                    curve.append(p(x))
                mid = max(curve)
                #crop if image is too big
                if len(bbox) >= 9 or abs(mid-image.height/2)<=50:
                    minX = max( 0 , minX-100 )
                    minY = max( 0 , minY-100 )
                    maxX = min( maxX+100 , image.width )
                    maxY = min( maxY+100 , image.height )
                    #if imageName == 'DSCF3120.JPG':
                    #    print( minX, ", ", minY, ", ", maxX, ", ", maxY )
                    if (maxY - minY) * 1.429 < (maxX - minX):
                        if (maxY + (((maxX-minX)-(maxY-minY)*1.429)/1.429)/2) < image.height:
                            maxY += (((maxX-minX)-(maxY-minY)*1.429)/1.429) / 2
                            if (minY - (((maxX-minX)-(maxY-minY)*1.429)/1.429)/2) > 0:
                                minY -= (((maxX-minX)-(maxY-minY)*1.429)/1.429)/2
                            else:
                                maxY -= ((((maxX-minX)-(maxY-minY)*1.429)/1.429)/2 - minY)
                                minY = 0
                        else:
                            minY -= (((maxX-minX)-(maxY-minY)*1.429)/1.429) / 2
                            minY += ((maxY + (((maxX-minX)-(maxY-minY)*1.429)/1.429)/2) - image.height)
                            maxY = image.height
                    elif (maxY - minY) * 1.429 > (maxX - minX):
                        if (maxX + ((maxY-minY)*1.429-(maxX-minX))/2) < image.width:
                            maxX += ((maxY-minY)*1.429-(maxX-minX)) / 2
                            if (minX - ((maxY-minY)*1.429-(maxX-minX))/2) > 0:
                                minX -= ((maxY-minY)*1.429-(maxX-minX)) / 2
                            else:
                                maxX -= (((maxY-minY)*1.429-(maxX-minX))/2 - minX)
                                minX = 0
                        else:
                            minX -= ((maxY-minY)*1.429-(maxX-minX)) / 2
                            minX += ((maxX + ((maxY-minY)*1.429-(maxX-minX))/2) - image.width)
                            maxX = image.width
                    minX = int(max( 0 , minX ))
                    minY = int(max( 0 , minY ))
                    maxX = int(min( maxX , image.width ))
                    maxY = int(min( maxY , image.height ))
                    #if imageName == 'DSCF3120.JPG':
                    #    print( minX, ", ", minY, ", ", maxX, ", ", maxY )

                    image = image.crop((minX,minY,maxX,maxY))
                    img = img.crop((minX,minY,maxX,maxY))
                    black_img = black_img.crop((minX,minY,maxX,maxY))

                    for i in range(len(xList)):
                        xList[i] = xList[i] - minX
                        yList[i] = yList[i] - minY
                        bbox[i] = [bbox[i][0]-minX, bbox[i][1]-minY, bbox[i][2]-minX, bbox[i][3]-minY]

                    '''for mask in masks:
                        new_msk = [[] for _ in range(image.height)]
                        #print( len(new_msk) )
                        for i in range( minY, maxY ):
                            for j in range( minX, maxX ):
                                new_msk[i-minY].append(mask[i][j])
                        mask = new_msk
                '''
                '''
                # write to node folder
                node_file_path = f"./result/{fileName}/node/node_{imageName[:-4]}.csv"
                with open(node_file_path, "w", newline="") as nodeFile:
                    csvWriter = csv.writer(nodeFile)
                    csvWriter.writerow(xList)
                    csvWriter.writerow(yList)
                '''

                pltImage = np.array(image)
                #pltImage = cv2.resize(pltImage, result.masks.data.shape[1:][::-1], interpolation=cv2.INTER_AREA)
                sam_img = image
                #image = cv2.imread(f"./{root}/{samdir}/{fileName}/{imageName}")
                image = cv2.cvtColor(pltImage, cv2.COLOR_BGR2RGB)
                if isCuda:
                    image = cv2.resize(image, (imageWidth, imageHeight), interpolation=cv2.INTER_AREA)
                '''oriMask = masks
                masks = SAM(image, bbox)
                for mask in masks:
                    show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
                    mask = mask.cpu().numpy()'''

                plt.axis('off')

                xList = []
                yList = []

                teethCnt = 0

                teethLocationSet = []
                imageTeethNodeSet = []


                npimg = np.array(image.copy(), dtype=np.uint8)
                npblack_img = np.array(black_img, dtype=np.uint8)
                #for box,mask in zip(boxes,masks):
                for box in bbox:
                    #print('box score',score.item())
                    teethCnt += 1

                    # print(image.height,image.width)
                    # print(mask.shape)
                    #mask = mask.detach().squeeze().cpu().numpy()
                    #mask = np.where(mask > confident_thre, 255, 0).astype(np.uint8)

                    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                    '''points = cv2.findNonZero(mask)
                    x,y,w,h = cv2.boundingRect(points)
                    x1 = int(x)
                    x2 = int(x+w)
                    y1 = int(y)
                    y2 = int(y+h)'''

                    teethLocationSet.append( TeethLocation(x1,y1,x2,y2) )

                    xList.append( (x1+x2)/2 )
                    yList.append( (y1+y2)/2 )


                    '''teethNodetmp = TeethNode(mask,TeethLocation(x1,y1,x2,y2))
                    teethNodetmp.labelId = int(box.cls[0])
                    imageTeethNodeSet.append(teethNodetmp)'''
                    color = list(np.random.choice(range(256), size=3))
                    '''npimg[np.where(mask>0)] = color
                    npblack_img[np.where(mask>0)] = color'''

                    color = tuple(np.random.choice(range(256), size=3))
                    
                    detLineScale = 0.005 #det line width scale
                    draw.line([(x1,y1),(x2,y1),(x2,y2),(x1,y2),(x1,y1)], fill=color, width=int(imageWidth*detLineScale))
                    ''''''
                    # draw.text((x1,y1), f"{score.item():.4f}", font=fnt) # draw confidence

                xArray = np.array(xList)
                yArray = np.array(yList)
                plt.xlim( 0,imageWidth )#
                plt.ylim( imageHeight,0 )#
                polyLine = np.polyfit(xArray,yArray,2)

                plt.imshow(pltImage)#
                plt.scatter(xArray,yArray) #draw dot#

                p = np.poly1d( polyLine )
                xArray.sort()
                x_base = np.linspace(0,imageWidth,imageWidth)
                plt.plot(x_base, p(x_base),color = 'red') #draw regression#

                pltSave(f"./result/{fileName}","regression","regression",imageName)#
                plt.clf()#

                imageGray = cv2.cvtColor(pltImage,cv2.COLOR_BGR2GRAY)

                imageInfo = PhotoImage(imageName,polyLine[0],teethCnt,imageGray,teethLocationSet,imageWidth,imageHeight,pltImage,imageTeethNodeSet, p,polyLine)
                imageInfoSet.append(imageInfo)

                resizeScale = 0.1
                whiteProportion = 1/100
                if checkFlag3D(pltImage,resizeScale,whiteProportion) == False:
                    imageFileInfo.is3D = False

                #pilSave(img,f"./result/{fileName}","det","det",imageName)

                img = Image.fromarray(npimg)
                black_img = Image.fromarray(npblack_img)

                #pilSave(img,f"./result/{fileName}","seg","seg",imageName)
                #pilSave(black_img,f"./result/{fileName}","seg","mask",imageName)

            writeFile = open(f"./result/{fileName}/imageClassification.csv",mode="w",newline="")
            csvWriter = csv.writer(writeFile)

            sortGradient(imageInfoSet)
            sortTeeth(imageInfoSet)

            #判斷面觀

            if imageFileInfo.is3D :
                csvWriter.writerow(["3D"])
            else:
                csvWriter.writerow(["2D"])
            for info in imageInfoSet:
                if info.teethRank == 1 and info.gradientRank >= 3 and info.useFlag == False:
                    csvWriter.writerow([info.imageName, "Face"])
                    info.view = 'Face'
                    info.useFlag = True
                    info = frontal_rotate(info)
                    break
            compared_info = None
            for info in imageInfoSet:
                if info.teethRank != 1 and info.gradientRank <= 2 and info.useFlag == False:
                    if compared_info is None:
                        compared_info = info
                    else:
                        if compared_info.edge_count > info.edge_count:
                            compared_info.view = 'Below'
                            info.view = 'Up'
                        else:
                            compared_info.view = 'Up'
                            info.view = 'Below'

                        for i in [compared_info, info]:
                            if i.view == 'Below' and i.gradient >= 0:
                                i.image = np.fliplr(np.flipud(i.image))
                                sam_img.rotate(180, expand=True)
                                csvWriter.writerow([i.imageName, "Below Rotated"])
                            elif i.view == 'Below':
                                csvWriter.writerow([i.imageName, "Below"])
                            elif i.view == 'Up' and i.gradient < 0:
                                i.image = np.fliplr(np.flipud(i.image))
                                sam_img.rotate(180, expand=True)
                                csvWriter.writerow([i.imageName, "Up Rotated"])
                            else:
                                csvWriter.writerow([i.imageName, "Up"])

                            i.useFlag = True

                        compared_info = None

            leftRightCnt = 0
            for i in range( len(imageInfoSet) ):
                if imageInfoSet[i].useFlag == False:
                    leftRightCnt += 1

            if leftRightCnt !=2:
                print("~~~!!! Classification Error !!!~~~",file = specialLogFile)
            else:
                leftViewCnt = 0
                rightViewCnt = 0
                if imageFileInfo.is3D :
                    for i in range( len(imageInfoSet) ):
                        if imageInfoSet[i].useFlag == False:
                            info = imageInfoSet[i]
                            info.useFlag = True

                            info.teethLocationSet = sorted(info.teethLocationSet,key = lambda loc : (loc.x1+loc.x2)/2)

                            teethCheckNum = 3
                            leftCnt = 0 #check width/height
                            for k in range(teethCheckNum):
                                leftCnt +=  ( (info.teethLocationSet[k].x2 - info.teethLocationSet[k].x1)/(info.teethLocationSet[k].y2 - info.teethLocationSet[k].y1) )

                            rightCnt = 0
                            for k in range(info.teethNum-teethCheckNum,info.teethNum):
                                rightCnt += ( (info.teethLocationSet[k].x2 - info.teethLocationSet[k].x1)/(info.teethLocationSet[k].y2 - info.teethLocationSet[k].y1) )

                            if leftCnt > rightCnt:
                                imageInfoSet[i].view = 'Right'
                                rightViewCnt += 1
                                csvWriter.writerow([info.imageName,"Right"])
                            else:
                                imageInfoSet[i].view = 'Left'
                                leftViewCnt += 1
                                csvWriter.writerow([info.imageName,"Left"])
                else :
                    for i in range(len(imageInfoSet)):
                        if not imageInfoSet[i].useFlag:
                            info = imageInfoSet[i]
                            info.useFlag = True

                            teeth_with_metrics = []
                            max_teeth_location = None
                            image_width = info.image.shape[1]
                            for teethLocation in info.teethLocationSet:
                                height = abs(teethLocation.y2 - teethLocation.y1)
                                width = abs(teethLocation.x2 - teethLocation.x1)
                                area = height * width
                                if not ((teethLocation.x2 == image_width and area < 15000) or (teethLocation.x1 == 0 and area < 15000)):
                                    x_coordinates = (teethLocation.x1 + teethLocation.x2) / 2
                                    teeth_with_metrics.append((area, width, height, x_coordinates, teethLocation))

                            teeth_with_metrics.sort(key=lambda x: x[3])
                            teeth_edge = teeth_with_metrics[:2] + teeth_with_metrics[-2:]
                            height_greater_than_width = [teeth for teeth in teeth_edge if teeth[2] > teeth[1]]
                            max_teeth_location = max(height_greater_than_width, key=lambda x: x[1])[4]
                            max_y = max(max(teethLocation.y1, teethLocation.y2) for teethLocation in info.teethLocationSet)
                            min_y = min(min(teethLocation.y1, teethLocation.y2) for teethLocation in info.teethLocationSet)
                            average_y = (max_y + min_y) / 2
                            max_average_y = (max_teeth_location.y1 + max_teeth_location.y2) / 2

                            if max_average_y > average_y:
                                info.image = np.fliplr(np.flipud(info.image))
                            oriH = len(info.grayData)
                            oriW = len(info.grayData[0])

                            resizeScale = 0.1
                            grayDataSmall = cv2.resize(info.grayData, dsize=(int(oriW*resizeScale),int(oriH*resizeScale)), interpolation=cv2.INTER_CUBIC)

                            h = len(grayDataSmall)
                            w = len(grayDataSmall[0])

                            w3 = int(w/3)
                            leftCnt = 0
                            for row in range(h):
                                for col in range(w3):
                                    leftCnt += grayDataSmall[row][col]
                            leftAverage = leftCnt / (h*w3)

                            rightCnt = 0
                            for row in range(h):
                                for col in range(w3+w3,w):
                                    rightCnt += grayDataSmall[row][col]
                            rightAverage = rightCnt / (h*(w-w3-w3))

                            if leftAverage > rightAverage:
                                imageInfoSet[i].view = 'Left'
                                leftViewCnt += 1
                                csvWriter.writerow([info.imageName,"Left"])
                            else:
                                imageInfoSet[i].view = 'Right'
                                rightViewCnt += 1
                                csvWriter.writerow([info.imageName,"Right"])
                checkClassication(leftViewCnt,rightViewCnt,imageInfoSet)

            writeFile.close()
            imageFileInfo.photoImageSet = imageInfoSet[:]

            for imageInfo in imageFileInfo.photoImageSet:
                ###### UP/Below #########
                if imageInfo.view == 'Up':
                    dimensionDict = constructDimensionDict(imageInfo.teethNodeSet)
                    for labelId, dimensionTeethNodeSet in dimensionDict.items():
                        dimensionTeethNodeSet = sorted(dimensionTeethNodeSet ,key = lambda x : ((x.box.x2+x.box.x1)/2.0),reverse = False)
                        if len(dimensionTeethNodeSet) == 2:
                            dimensionTeethNodeSet[0].labelId += 1*10
                            dimensionTeethNodeSet[1].labelId += 2*10
                        else:
                            for teethNode in dimensionTeethNodeSet:
                                boxMidX,boxMidY = boxMiddle(teethNode.box)
                                if boxMidX < imageInfo.width/2:
                                    teethNode.labelId += 1*10
                                else:
                                    teethNode.labelId += 2*10

                elif imageInfo.view == 'Below':
                    dimensionDict = constructDimensionDict(imageInfo.teethNodeSet)
                    for labelId, dimensionTeethNodeSet in dimensionDict.items():
                        dimensionTeethNodeSet = sorted(dimensionTeethNodeSet ,key = lambda x : ((x.box.x2+x.box.x1)/2.0),reverse = False)
                        if len(dimensionTeethNodeSet) == 2:
                            dimensionTeethNodeSet[0].labelId += 4*10
                            dimensionTeethNodeSet[1].labelId += 3*10
                        else:
                            for teethNode in dimensionTeethNodeSet:
                                boxMidX,boxMidY = boxMiddle(teethNode.box)
                                if boxMidX < imageInfo.width/2:
                                    teethNode.labelId += 4*10
                                else:
                                    teethNode.labelId += 3*10

                elif imageInfo.view == 'Right':
                    dimensionDict = constructDimensionDict(imageInfo.teethNodeSet)
                    for labelId, dimensionTeethNodeSet in dimensionDict.items():
                        dimensionTeethNodeSet = sorted(dimensionTeethNodeSet ,key = lambda x : ((x.box.y2+x.box.y1)/2.0),reverse = False)
                        if len(dimensionTeethNodeSet) == 2:
                            dimensionTeethNodeSet[0].labelId += 1*10
                            dimensionTeethNodeSet[1].labelId += 4*10
                        else:
                            for teethNode in dimensionTeethNodeSet:
                                boxMidX,boxMidY = boxMiddle(teethNode.box)
                                if boxMidY < imageInfo.height/2:
                                    teethNode.labelId += 1*10
                                else:
                                    teethNode.labelId += 4*10
                elif imageInfo.view == 'Left':
                    dimensionDict = constructDimensionDict(imageInfo.teethNodeSet)
                    for labelId, dimensionTeethNodeSet in dimensionDict.items():
                        dimensionTeethNodeSet = sorted(dimensionTeethNodeSet ,key = lambda x : ((x.box.y2+x.box.y1)/2.0),reverse = False)
                        if len(dimensionTeethNodeSet) == 2:
                            dimensionTeethNodeSet[0].labelId += 2*10
                            dimensionTeethNodeSet[1].labelId += 3*10
                        else:
                            for teethNode in dimensionTeethNodeSet:
                                boxMidX,boxMidY = boxMiddle(teethNode.box)
                                if boxMidY < imageInfo.height/2:
                                    teethNode.labelId += 2*10
                                else:
                                    teethNode.labelId += 3*10
                elif imageInfo.view == 'Face':
                    teethNodeSet = imageInfo.teethNodeSet
                    upTeethNodeSet = [] #第一、二象限
                    belowTeethNodeSet = [] #第三、四象限

                    for teethNode in teethNodeSet:
                        Ymiddle = imageInfo.regression((teethNode.box.x1+teethNode.box.x2)/2)#以回歸線為中線
                        if Ymiddle > (teethNode.box.y1 + teethNode.box.y2)/2:
                            upTeethNodeSet.append(teethNode)
                        else:
                            belowTeethNodeSet.append(teethNode)

                    dimensionDict = constructDimensionDict(upTeethNodeSet)
                    for labelId, dimensionTeethNodeSet in dimensionDict.items():
                        dimensionTeethNodeSet = sorted(dimensionTeethNodeSet ,key = lambda x : ((x.box.x2+x.box.x1)/2.0),reverse = False)
                        if len(dimensionTeethNodeSet) == 2:
                            dimensionTeethNodeSet[0].labelId += 1*10
                            dimensionTeethNodeSet[1].labelId += 2*10
                        else:
                            for teethNode in dimensionTeethNodeSet:
                                boxMidX,boxMidY = boxMiddle(teethNode.box)
                                if boxMidX < imageInfo.width/2:
                                    teethNode.labelId += 1*10
                                else:
                                    teethNode.labelId += 2*10

                    dimensionDict = constructDimensionDict(belowTeethNodeSet)
                    for labelId, dimensionTeethNodeSet in dimensionDict.items():
                        dimensionTeethNodeSet = sorted(dimensionTeethNodeSet ,key = lambda x : ((x.box.x2+x.box.x1)/2.0),reverse = False)
                        if len(dimensionTeethNodeSet) == 2:
                            dimensionTeethNodeSet[0].labelId += 4*10
                            dimensionTeethNodeSet[1].labelId += 3*10
                        else:
                            for teethNode in dimensionTeethNodeSet:
                                boxMidX,boxMidY = boxMiddle(teethNode.box)
                                if boxMidX < imageInfo.width/2:
                                    teethNode.labelId += 4*10
                                else:
                                    teethNode.labelId += 3*10
                else:
                    print('classification error view',file=specialLogFile)

                saveName =  viewOfficial(imageInfo.view) + '_' + fileName + '.png'
                pilSave(Image.fromarray(imageInfo.image.copy()),f"./sample/{fileName}","","",saveName)

    print( time.ctime() )
    #########著色#########
    '''
            with open('./utils/teeth_rgb.json') as jf:
                with open('./utils/error_rgb.json') as errorJson:
                    colorData = json.load(jf)
                    errorData = json.load(errorJson)
                    for imageInfo in imageFileInfo.photoImageSet:

                        objectName =  viewOfficial(imageInfo.view) + '_' + fileName + '.text'
                        with open(f"./result/{fileName}/color/{objectName}", "wb") as file:
                            pickle.dump([i.dump() for i in imageInfo.teethNodeSet if str(i.labelId) in colorData[0]], file)

                        errorIndex = ord('a')
                        imageName = imageInfo.imageName
                        print(imageName,' => draw color')
                        img = imageInfo.image.copy()
                        black_img = Image.new('RGB', tuple([len(img[0]),len(img)]), (0, 0, 0))
                        npimg = np.array(img, dtype=np.uint8)
                        npblack_img = np.array(black_img, dtype=np.uint8)

                        for teethNode in imageInfo.teethNodeSet:
                            #print("label",teethNode.labelId)
                            if str(teethNode.labelId) not in colorData[0]: #新增找不到label
                                if isLabel:
                                    color = errorData[0][chr(errorIndex)].copy()
                                    color.reverse()  # rgb,bgr
                                    teethNode.labelId = chr(errorIndex)
                                    errorIndex += 1
                                    npimg[np.where(teethNode.mask>0)] = color
                                    npblack_img[np.where(teethNode.mask>0)] = color
                            else:
                                #print(teethNode.labelId," => ",extract_box_brightness(imageInfo.image,teethNode.box))
                                #print(teethNode.labelId," => ",extract_mask_brightness(imageInfo.image,teethNode.mask))
                                color = colorData[0][str(teethNode.labelId)].copy()  # lableId = -1, 要擋掉
                                color.reverse()  # rgb,bgr
                                npimg[np.where(teethNode.mask>0)] = color
                                npblack_img[np.where(teethNode.mask>0)] = color

                        img = Image.fromarray(npimg)
                        black_img = Image.fromarray(npblack_img)
                        saveName =  viewOfficial(imageInfo.view) + '_' + fileName + '.png'
                        pilSave(black_img,f"./result/{fileName}","changeColor","",saveName)
                        pilSave(Image.fromarray(imageInfo.image.copy()),f"./sample/{fileName}","","",saveName)
                        pilSave(Image.fromarray(imageInfo.image.copy()),f"./result/{fileName}","newNameSample","",saveName)
                        for teethNode in imageInfo.teethNodeSet:
                            if isLabel or (str(teethNode.labelId) in colorData[0]):
                                fontSize = int(len(imageInfo.image[0])*0.03)
                                x = int((teethNode.box.x1+teethNode.box.x2)/2 - fontSize/2)
                                y = int((teethNode.box.y1+teethNode.box.y2)/2 - fontSize/2)
                                draw = ImageDraw.Draw(img)
                                font = ImageFont.truetype("./utils/arial.ttf", fontSize)
                                draw.text((x,y),str(teethNode.labelId),font = font)
                                imgDraw = ImageDraw.Draw(img)
                        pilSave(Image.fromarray(np.hstack([imageInfo.image,np.array(img)])),f"./result/{fileName}","color","seg",saveName) 
                        pilSave(black_img,f"./result/{fileName}","color","mask",saveName)
    '''

