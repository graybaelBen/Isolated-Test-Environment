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
detector = SIFT_detector
descriptor = SIFT_descriptor
# descriptor = RootSIFT_descriptor
matcher = FLANN_matcher

#assign active directories
current_dir = os.path.join('Batch1','B1.1')
imgdir = os.path.join(current_dir,'images')
maskdir = os.path.join(current_dir,'masks')
processeddir = os.path.join(current_dir,'processed')

imgDirArr = os.listdir(imgdir)
maskDirArr = os.listdir(maskdir)

# Image Processing
for idx, image in enumerate(imgDirArr):
    img = os.path.join(imgdir, image)
    mask = os.path.join(maskdir, maskDirArr[idx])
    img = cv2.imread(img,0)
    mask = cv2.imread(mask,0)

    # processed = processor.grayscale(img)
    processed = processor.mask(img, mask)
    processed = processor.threshold(processed)

    cv2.imwrite(os.path.join(processeddir, image), processed)

#imgDirArr = os.listdir(processeddir)
#imgdir = processeddir

# loop through image directory, generate keypoints and grayscale images
kpArray = []
kpCountArray = []

for idx, image in enumerate(imgDirArr):
    img = cv2.imread(os.path.join(imgdir, image))
    mask = cv2.imread(os.path.join(maskdir, maskDirArr[idx]),0)
    print(os.path.join(imgdir, image))
    print(os.path.join(maskdir, maskDirArr[idx]))
    kp = detector.detect(img,mask)
    kpArray.append(kp)
    kpCountArray.append(len(kp))
    
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
#print(grayDirArr)
for idx, image in enumerate(imgDirArr):

    img = cv2.imread(os.path.join(imgdir, image))
    print(kpCountArray)
    kp = kpArray[idx]
    kp, des = descriptor.descript(img ,kp)
    #RootSIFT
    #des /= (des.sum(axis=1, keepdims=True) + 1e-7)
    #des = numpy.sqrt(des)
    #RootSIFT end
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
for idx1, img1 in enumerate(imgDirArr):

    # internal loop starts at image in the next index
    for idx2, img2 in enumerate(imgDirArr[idx1+1:]):
        #print(compare)
        #print(img1Index)
        
        image1 = cv2.imread(os.path.join(imgdir,img1))
        image2 = cv2.imread(os.path.join(imgdir,img2))
        matchCount, drawnMatches = matcher.match(desArray[idx1], desArray[idx1+idx2+1], image1, image2, kpArray[idx1], kpArray[idx1+idx2+1])
        compared_images = img1+img2
        results = os.path.join(current_dir,"results", compared_images)
        print(results)
        cv2.imwrite(results,drawnMatches)

        #ORB
        #print(img2Index)
        #matchCount += matcher.match(desArray[img1Index], desArray[img2Index] ,image, compare, kpArray[img1Index], kpArray[img2Index])
        
        matchCountArray.append(matchCount)
        print(idx2)

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
