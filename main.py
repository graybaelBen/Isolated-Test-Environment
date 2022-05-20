# Isolated Test Environment Main
from HessianAffine import HessianAffine
import cv2
import numpy as np
import os
from multiprocessing.pool import ThreadPool #ASIFT uses this


# import modules
from ORB_detector_module import ORB_detector # ins: img_filename, mask_filename; outs: kp, gray_img
from ORB_descriptor_module import ORB_descriptor # ins: kp, gray_img; outs: kp, des
from SIFT_detector_module import SIFT_detector # ins: img_filename, mask_filename; outs: kp, gray_img
from SIFT_descriptor_module import SIFT_descriptor
from FLANN_matcher_module import FLANN_matcher # ins: kp, des; outs:
from BLOB_detector import BLOB_detector
from preprocessor  import Processor
#from ASIFT_module import affine_detect, match_and_dont_draw
from HessianAffine import HessianAffine

#from RootSIFT_descriptor_module import RootSIFT_descriptor

HA = HessianAffine()

# assign modules
detector = SIFT_detector
descriptor = SIFT_descriptor

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
    # patcheddir = os.path.join(current_dir, 'patched')


    # if not os.path.exists(os.path.join(current_dir,'spotMask')):
    #     os.makedirs(os.path.join(current_dir,'spotMask'))
    # spotMaskdir = os.path.join(current_dir,'spotMask')

    imgDirArr = os.listdir(imgdir)
    maskDirArr = os.listdir(maskdir)
    #patchedDirArr = os.listdir(patcheddir)
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
        #processed, spot_mask = process.spot_mask(processed)

        cv2.imwrite(os.path.join(processeddir, img), processed)

        #cv2.imwrite(os.path.join(spotMaskdir, maskDirArr[idx]), spot_mask)
        #cv2.imshow("Processed Image",processed)
        #cv2.waitKey(0)

    # COMMENT OUT FOR NO PROCESSING
    imgDirArr = os.listdir(processeddir)
    # maskDirArr = os.listdir(spotMaskdir)
    imgdir = processeddir

    #'''

    # Keypoint Detection
    kpArray = []
    kpCountArray = []
    for idx, img in enumerate(imgDirArr):
        #if using HA, feed path instead of image
        if(detector == HA or descriptor == HA):
            image = os.path.join(imgdir, img)
            mask = os.path.join(maskdir, maskDirArr[idx])
        else:
            image = cv2.imread(os.path.join(imgdir, img),0)
            mask = cv2.imread(os.path.join(maskdir, maskDirArr[idx]),0)

        # print(os.path.join(imgdir, img))
        # print(os.path.join(maskdir, maskDirArr[idx]))

    
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
        # print(kpCountArray)
        kp = kpArray[idx]
        kp, des = descriptor.descript(image, kp)

        #RootSIFT
        #des /= (des.sum(axis=1, keepdims=True) + 1e-7)
        #des = numpy.sqrt(des)
        #RootSIFT end
        #ORB
        #kp, des = descriptor.ORB_descript(gray_img,kp)
        # print(type(maskDirArr[idx]))
        mask = os.path.join(maskdir, maskDirArr[idx])
        
        #kp, des = HA.reapplyMask(image, mask, kp, des)
        desArray.append(des)


    #print('descriptor length :', len(desArray))

    
    #ASIFT detection+description (currently have to use pre-masked/patched data)
    """
    pool=ThreadPool(processes = cv2.getNumberOfCPUs())
    #matchCountArr = []
    for patched in patchedDirArr:
        img1 = cv2.imread(os.path.join(patcheddir, patched), 0)
        kp1, desc1 = affine_detect(detector, img1, pool = pool)
        for compare in patchedDirArr[patchedDirArr.index(patched)+1:]:
            img2 = cv2.imread(os.path.join(patcheddir, compare), 0)
            kp2, desc2 = affine_detect(detector, img2, pool = pool)
            print('img1 - %d features, img2 - %d features' % (len(kp1), len(kp2)))
            ASIFT_matchCountArr=match_and_dont_draw(patched+"_"+compare, img1, img2, kp1, kp2, desc1, desc2)
            
    """


    # Image Comparison
    matchCountArray = []
    matchCount = 0
    for idx1, img1 in enumerate(imgDirArr):
        # internal loop starts at image in the next index
        for idx2, img2 in enumerate(imgDirArr[idx1+1:]):
    
            image1 = cv2.imread(os.path.join(imgdir,img1))
            # print(type(image))
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
            # print(idx2)
    print(batch, " COMPLETE")
    return matchCountArray, kpCountArray
#run()
