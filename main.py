# Isolated Test Environment Main
import cv2
import numpy as np
import csv
import os


# import modules
from ORB_detector_module import ORB_detector # ins: img_filename, mask_filename; outs: kp, gray_img
from ORB_descriptor_module import ORB_descriptor # ins: kp, gray_img; outs: kp, des
from SIFT_detector_module import SIFT_detector # ins: img_filename, mask_filename; outs: kp, gray_img
from SIFT_descriptor_module import SIFT_descriptor
from FLANN_matcher_module import FLANN_matcher # ins: kp, des; outs:
from BLOB_detector import BLOB_detector
from preprocessor  import Processor
#from RootSIFT_descriptor_module import RootSIFT_descriptor

# assign modules
detector = SIFT_detector
descriptor = SIFT_descriptor
#descriptor = RootSIFT_descriptor
matcher = FLANN_matcher
# instatiate object of class
process = Processor()

# assign active directories
current_dir = os.path.join('Batch1','pristine2send')

imgdir = os.path.join(current_dir,'images')
maskdir = os.path.join(current_dir,'masks')
processeddir = os.path.join(current_dir,'processed')

if not os.path.exists(os.path.join(current_dir,'spotMask')):
    os.makedirs(os.path.join(current_dir,'spotMask'))
spotMaskdir = os.path.join(current_dir,'spotMask')

imgDirArr = os.listdir(imgdir)
maskDirArr = os.listdir(maskdir)

# Image Processing
for idx, img in enumerate(imgDirArr):
    image = cv2.imread(os.path.join(imgdir, img))
    mask = cv2.imread(os.path.join(maskdir, maskDirArr[idx]),0)
    
    processed = image
    processed = process.grayscale(processed)
    processed = process.mask(processed, mask)
    
    #processed = process.threshold(processed)
   
    processed = process.cluster_quantize(processed,6)   
    # image is inverted so erode and dilate are swapped from our perspective
    processed = process.erode(processed,3)
    processed = process.dilate(processed,10)
    #processed, spot_mask = process.spot_mask(processed)

    cv2.imwrite(os.path.join(processeddir, img), processed)

    #cv2.imwrite(os.path.join(spotMaskdir, maskDirArr[idx]), spot_mask)
    #cv2.imshow("Processed Image",processed)
    #cv2.waitKey(0)

# COMMENT OUT FOR NO PROCESSING
imgDirArr = os.listdir(processeddir)
#maskDirArr = os.listdir(spotMaskdir)
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

   
    kps = detector.detect(image,mask)

    kpArray.append(kps)
    kpCountArray.append(len(kps))

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
    kp, des = descriptor.descript(image, kp)

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

        #ORB
        #print(img2Index)
        #matchCount += matcher.match(desArray[img1Index], desArray[img2Index] ,image, compare, kpArray[img1Index], kpArray[img2Index])
        
        matchCountArray.append(matchCount)
        print(idx2)

# print to csv
np.transpose(kpCountArray)
np.transpose(matchCountArray)

with open('results.csv', 'w', newline = '') as f:
    writer = csv.writer(f)
    writer.writerow(kpCountArray)
    for matchCount in matchCountArray:
        writer.writerow([matchCount])

#'''
print('done!')
