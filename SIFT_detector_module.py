import numpy as np
import cv2 

class SIFT_detector:

    def detect(img_filename, mask_filename):
        
        # for manual image usage:
        #img_filename = 'Batch1\B1.1\images\02_Station04_Camera1_2012-7-5_17-10-22(0).JPG'
        #mask_filename = 'Batch1\B1.1\masks\02_Station04_Camera1_2012-7-5_17-10-22(0).BMP'
        print(img_filename)
        img = cv2.imread(img_filename)
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        mask = cv2.imread(mask_filename)
        gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create(nfeatures = 0, #0
                                nOctaveLayers = 3, #3
                                contrastThreshold = 0.4, #0.4
                                edgeThreshold = 10, #10
                                sigma = 1.6 ) #1.6)
        kp = sift.detect(gray_img, gray_mask)
        print(len(kp))

        # draw keypoints
        #img = cv.drawKeypoints(gray,kp,img)
        #cv.imwrite('sift_keypoints.jpg',img)

        return kp, gray_img
