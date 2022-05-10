import cv2
import numpy as np
import os

# import modules
from FLANN_matcher_module import FLANN_matcher # ins: kp, des; outs:
from preprocessor  import Processor
from HessianAffine import HessianAffine
HA = HessianAffine()

#descriptor = RootSIFT_descriptor
matcher = FLANN_matcher
# instatiate object of class
process = Processor()

def run(batch):
    # assign active directories
    current_dir = os.path.join('Batches',batch)

    imgdir = os.path.join(current_dir,'images')
    maskdir = os.path.join(current_dir,'masks')
    processeddir = os.path.join(current_dir,'processed')

    imgDirArr = os.listdir(imgdir)
    maskDirArr = os.listdir(maskdir)

    imgDirArr.sort()
    maskDirArr.sort()

    # Image Processing
    for idx, img in enumerate(imgDirArr):
        image = cv2.imread(os.path.join(imgdir, img))
        mask = cv2.imread(os.path.join(maskdir, maskDirArr[idx]),0)
        processed = image
        processed = process.grayscale(processed)
        processed = process.mask(processed, mask)
        
        #processed = process.threshold(processed)
        #processed = process.cluster_quantize(processed,6)   
        # image is inverted so erode and dilate are swapped from our perspective
        #processed = process.erode(processed,3)
        #processed = process.dilate(processed,10)

        if not os.path.exists(os.path.join(current_dir,'processed')):
            os.makedirs(os.path.join(current_dir,'processed'))
        cv2.imwrite(os.path.join(processeddir, img), processed)


    # COMMENT OUT FOR NO PROCESSING
    imgDirArr = os.listdir(processeddir)

    # Keypoint Detection Description
    kpArray = []
    kpCountArray = []
    desArray = []
    
    for idx, img in enumerate(imgDirArr):
        image = os.path.join(imgdir, img)
        mask = os.path.join(maskdir, maskDirArr[idx])
        kps, des = HA.detect_describe(image,mask)
                
        kps, des = HA.reapplyMask(image, mask, kps, des)

        desArray.append(des)
        kpArray.append(kps)
        kpCountArray.append(len(kps))

    #print('kpcount length :', len(kpCountArray))
    #print('descriptor length :', len(desArray))
    if not os.path.exists(os.path.join(current_dir,'results')):
        os.makedirs(os.path.join(current_dir,'results'))
        print("helloooo")

    # Image Comparison
    matchCountArray = []
    for idx1, img1 in enumerate(imgDirArr):
        # internal loop starts at image in the next index
        for idx2, img2 in enumerate(imgDirArr[idx1+1:]):
            matchCount = 0
            image1 = cv2.imread(os.path.join(imgdir,img1))
            # print(type(image))
            image2 = cv2.imread(os.path.join(imgdir,img2))
            matchCount, drawnMatches = matcher.match(desArray[idx1], desArray[idx1+idx2+1], image1, image2, kpArray[idx1], kpArray[idx1+idx2+1])
            compared_images = img1+img2
       
            results = os.path.join(current_dir,"results", compared_images)
            print(results)
            cv2.imwrite(results,drawnMatches)  
            matchCountArray.append(matchCount)
            
            # print(idx2)
    print(batch, " COMPLETE")
    return matchCountArray, kpCountArray
