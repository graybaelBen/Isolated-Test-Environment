# Isolated Test Environment Main
import cv2
import numpy
import csv
import os

# import modules
from SIFT_detector_module import SIFT_detector # ins: img_filename, mask_filename; outs: kp, gray_img
from SIFT_descriptor_module import SIFT_descriptor # ins: kp, gray_img; outs: kp, des
from FLANN_matcher_module import FLANN_matcher # ins: kp, des; outs:
#from RootSIFT_descriptor_module import RootSIFT_descriptor

# assign modules
detector = SIFT_detector
descriptor = SIFT_descriptor
# descriptor = RootSIFT_descriptor
matcher = FLANN_matcher

imgdir = 'Batch2.1'
maskdir = 'Batch2.1M'
graydir = 'Batch2.1G'

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
    kp, gray_img = detector.SIFT_detect(img, mask)
    kpArray.append(kp)
    grayArray.append(gray_img)
    kpCountArray.append(len(kp))
    cv2.imwrite(os.path.join(graydir, image), gray_img)
#debugging
print('kpcount length :', len(kpCountArray))

# loop through grayscale directory, generate descriptors
desArray = []
for gray in grayDirArr:
    gray_img = cv2.imread(os.path.join(graydir, gray))
    kp = kpArray[grayDirArr.index(gray)]
    kp, des = descriptor.descript(gray_img, kp)
    desArray.append(des)
#debugging
print('descriptor length :', len(desArray))

# loop through all combinations and compare images
matchCountArray = []
for image in imgDirArr:
    img1Index = imgDirArr.index(image)
    print('image', img1Index+1, '/', len(imgDirArr))
    for compare in imgDirArr[imgDirArr.index(image)+1:]:
        img2Index = imgDirArr.index(compare)
        matchCount = matcher.FLANN_match(desArray[img1Index], desArray[img2Index])
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

# looping runner from flann.py
'''
kpCountArray = []
matchCountArray = []

loopcount = 0
for image in imgDirArr:
    img1 = os.path.join(imgdir, image)
    mask1 = os.path.join(maskdir, maskDirArr[imgDirArr.index(image)])
    for compare in imgDirArr[imgDirArr.index(image)+1:]:
        img2 = os.path.join(imgdir, compare)
        mask2 = os.path.join(maskdir, maskDirArr[imgDirArr.index(compare)])
        
        #print(img1, '\n', img2, '\n\n')
        
        kp1count, kp2count, matchCount = FLANN_match(img1, mask1, img2, mask2)
        kpCountArray.append(kp1count)
        kpCountArray.append(kp2count)
        matchCountArray.append(matchCount)
        
        print(matchCount)
        loopcount +=1

print(loopcount)
print(len(kpCountArray))
print(len(matchCountArray))

# print to csv
with open('results.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(kpCountArray)
    writer.writerow(matchCountArray)

print('done!')
'''