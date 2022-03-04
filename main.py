# Isolated Test Environment Main
import cv2
import numpy
import csv
import os

# import modules
from SIFT_detector_module import SIFT_detector # ins: img_filename, mask_filename; outs: kp, gray_img
from SIFT_descriptor_module import SIFT_descriptor # ins: kp, gray_img; outs: kp, des
from FLANN_matcher_module import FLANN_matcher # ins: kp, des; outs:
from preprocessor  import processor
#from RootSIFT_descriptor_module import RootSIFT_descriptor

# assign modules
detector = SIFT_detector
descriptor = SIFT_descriptor
# descriptor = RootSIFT_descriptor
matcher = FLANN_matcher

#assign active directories
imgdir = 'Batches\Batch1\images'
maskdir = 'Batches\Batch1\masks'
graydir = 'Batches\Batch1\gray'

#process images
processor.threshold(imgdir, maskdir)
imgdir = 'Batches\Batch1\processed'

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
    #cv2.imwrite(os.path.join(graydir, image), gray_img)
#debugging
print('kpcount length :', len(kpCountArray))

# loop through grayscale directory, generate descriptors
desArray = []
for gray in grayDirArr:
    gray_img = cv2.imread(os.path.join(graydir, gray))
    kp = kpArray[grayDirArr.index(gray)]
    kp, des = descriptor.descript(gray_img,kp)
    #RootSIFT
    #des /= (des.sum(axis=1, keepdims=True) + 1e-7)
    #des = numpy.sqrt(des)
    #RootSIFT end
    desArray.append(des)

#debugging
print('descriptor length :', len(desArray))

# loop through all combinations and compare images
matchCountArray = []
# starts at image in index 0
for image in imgDirArr:
    img1Index = imgDirArr.index(image)
    print('image', img1Index+1, '/', len(imgDirArr))
    # internal loop starts at image in the next index
    for compare in imgDirArr[imgDirArr.index(image)+1:]:
        img2Index = imgDirArr.index(compare)
        matchCount = matcher.match(desArray[img1Index], desArray[img2Index])
        matchCountArray.append(matchCount)

# print to csv
numpy.transpose(kpCountArray)
numpy.transpose(matchCountArray)

with open('results.csv', 'w', newline = '') as f:
    writer = csv.writer(f)
    writer.writerow(kpCountArray)
    for matchCount in matchCountArray:
        writer.writerow([matchCount])
print('done!')
