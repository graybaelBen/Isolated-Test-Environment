# Isolated Test Environment Main
import cv2
import numpy
import csv
import os

# import modules
from ORB_detector_module import ORB_detector # ins: img_filename, mask_filename; outs: kp, gray_img
from ORB_descriptor_module import ORB_descriptor # ins: kp, gray_img; outs: kp, des
from SIFT_detector_module import SIFT_detector # ins: img_filename, mask_filename; outs: kp, gray_img
from SIFT_descriptor_module import SIFT_descriptor
from FLANN_matcher_module import FLANN_matcher # ins: kp, des; outs:
from preprocessor  import processor
#from RootSIFT_descriptor_module import RootSIFT_descriptor

# assign modules
detector = ORB_detector
descriptor = ORB_descriptor
#detector = SIFT_detector
#descriptor = SIFT_descriptor
# descriptor = RootSIFT_descriptor
matcher = FLANN_matcher

#assign active directories
imgdir = 'Batches\Batch1\images'
maskdir = 'Batches\Batch1\masks'
graydir = 'Batches\Batch1\gray'
processedDir = 'Batches\Batch1\processed'
#process images
processor.threshold(imgdir, maskdir, processedDir)
imgdir = processedDir

imgDirArr = os.listdir(imgdir)
maskDirArr = os.listdir(maskdir)
grayDirArr = os.listdir(graydir)

# loop through image directory, generate keypoints and grayscale images
kpArray = []
kpCountArray = []
grayArray = []
for image in imgDirArr:
    img = os.path.join(imgdir, image)
    mask = os.path.join(maskdir, maskDirArr[imgDirArr.index(image)])
    kp, gray_img = detector.detect(img,mask)
    kpArray.append(kp)
    grayArray.append(gray_img)
    kpCountArray.append(len(kp))

    # save gray images - only needed on the first run of a batch
    cv2.imwrite(os.path.join(graydir, image), gray_img)
    
    #ORB
    #kp, gray_img = detector.ORB_detect(img,mask)
    #kpArray.append(kp)
    #grayArray.append(gray_img)
    #kpCountArray.append(len(kp))
    #cv2.imwrite(os.path.join(graydir, image), gray_img)

#debugging
#print('gray len: ', len(grayArray))
print('kpcount length :', len(kpCountArray))

# loop through grayscale directory, generate descriptors
desArray = []
grayDirArr = os.listdir(graydir)
#print(grayDirArr)
for gray in grayDirArr:
    gray_img = cv2.imread(os.path.join(graydir, gray))
    kp = kpArray[grayDirArr.index(gray)]
    kp, des = descriptor.descript(gray_img, kp)
    #ORB
    #kp, des = descriptor.ORB_descript(gray_img,kp)

    #print(type(des[0]))

    desArray.append(des)
#debugging
#print('descriptor length :', len(desArray))

# loop through all combinations and compare images
matchCountArray = []
# starts at image in index 0
matchCount = 0
for image in imgDirArr:
    #print(image)
    img1Index = imgDirArr.index(image)
    a = 0
    print('image', img1Index+1, '/', len(imgDirArr))
    # internal loop starts at image in the next index
    for compare in imgDirArr[imgDirArr.index(image)+1:]:
        #print(compare)
        #print(img1Index)
        
        img2Index = imgDirArr.index(compare)
        matchCount = matcher.match(desArray[img1Index], desArray[img2Index], image, compare, kpArray[img1Index], kpArray[img2Index])
        
        #ORB
        #print(img2Index)
        #matchCount += matcher.match(desArray[img1Index], desArray[img2Index] ,image, compare, kpArray[img1Index], kpArray[img2Index])
        matchCountArray.append(matchCount)
        a += 1
        print(a)

print(matchCount)
# print to csv
numpy.transpose(kpCountArray)
numpy.transpose(matchCountArray)

with open('results.csv', 'w', newline = '') as f:
    writer = csv.writer(f)
    writer.writerow(kpCountArray)
    for matchCount in matchCountArray:
        writer.writerow([matchCount])
print('done!')
