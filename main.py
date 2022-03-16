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
#descriptor = RootSIFT_descriptor
matcher = FLANN_matcher

# assign active directories
current_dir = os.path.join('Demo','D1.1')

imgdir = os.path.join(current_dir,'images')
maskdir = os.path.join(current_dir,'masks')
processeddir = os.path.join(current_dir,'processed')
imgDirArr = os.listdir(imgdir)
maskDirArr = os.listdir(maskdir)

# Image Processing
for idx, img in enumerate(imgDirArr):
    image = cv2.imread(os.path.join(imgdir, img))
    mask = cv2.imread(os.path.join(maskdir, maskDirArr[idx]),0)
    cv2.imshow("Original Mask", mask)
    cv2.imshow("Original Image",image)
    cv2.waitKey(0)

    processed = image
    processed = processor.grayscale(processed)
    processed = processor.mask(processed, mask)
    processed = processor.threshold(processed)
    processed = processor.dilate(processed)
    processed = processor.erode(processed)
    
    cv2.imwrite(os.path.join(processeddir, img), processed)
    cv2.imshow("Processed Image",processed)
    cv2.waitKey(0)

# COMMENT OUT FOR NO PROCESSING
imgDirArr = os.listdir(processeddir)
imgdir = processeddir

#'''

# Keypoint Detection
kpArray = []
kpCountArray = []
for idx, img in enumerate(imgDirArr):
    image = cv2.imread(os.path.join(imgdir, img),0)
    mask = cv2.imread(os.path.join(maskdir, maskDirArr[idx]),0)
    print(os.path.join(imgdir, img))
    print(os.path.join(maskdir, maskDirArr[idx]))

    kp = detector.detect(image,mask)
    kpArray.append(kp)
    kpCountArray.append(len(kp))
    img_kp = processor.draw_cross_keypoints(image, kp, color=(120,157,187))
    cv2.imshow("KeyPoints",img_kp)
    cv2.waitKey(0)
    #ORB
    #kp, gray_img = detector.ORB_detect(img,mask)
    #kpArray.append(kp)
    #grayArray.append(gray_img)
    #kpCountArray.append(len(kp))
    #cv2.imwrite(os.path.join(graydir, image), gray_img)


#print('kpcount length :', len(kpCountArray))

# Keypoint Description
desArray = []
for idx, img in enumerate(imgDirArr):

    image = cv2.imread(os.path.join(imgdir, img))
    print(kpCountArray)
    kp = kpArray[idx]
    kp, des = descriptor.descript(image ,kp)

    #RootSIFT
    #des /= (des.sum(axis=1, keepdims=True) + 1e-7)
    #des = numpy.sqrt(des)
    #RootSIFT end
    #ORB
    #kp, des = descriptor.ORB_descript(gray_img,kp)

    desArray.append(des)

#print('descriptor length :', len(desArray))


# Image Comparison
matchCountArray = []
matchCount = 0
for idx1, img1 in enumerate(imgDirArr):
    # internal loop starts at image in the next index
    for idx2, img2 in enumerate(imgDirArr[idx1+1:]):
  
        image1 = cv2.imread(os.path.join(imgdir,img1))
        image2 = cv2.imread(os.path.join(imgdir,img2))
        matchCount, drawnMatches = matcher.match(desArray[idx1], desArray[idx1+idx2+1], image1, image2, kpArray[idx1], kpArray[idx1+idx2+1])
        compared_images = img1+img2
        results = os.path.join(current_dir,"results", compared_images)
        print(results)
        cv2.imwrite(results,drawnMatches)

        cv2.imshow("Comparison", drawnMatches)
        cv2.waitKey(0)
        #ORB
        #print(img2Index)
        #matchCount += matcher.match(desArray[img1Index], desArray[img2Index] ,image, compare, kpArray[img1Index], kpArray[img2Index])
        
        matchCountArray.append(matchCount)
        print(idx2)

# print to csv
numpy.transpose(kpCountArray)
numpy.transpose(matchCountArray)

with open('results.csv', 'w', newline = '') as f:
    writer = csv.writer(f)
    writer.writerow(kpCountArray)
    for matchCount in matchCountArray:
        writer.writerow([matchCount])

#'''
print('done!')
