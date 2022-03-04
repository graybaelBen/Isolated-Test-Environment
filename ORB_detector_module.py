import numpy as np
import cv2 

class ORB_detector:
    def ORB_detect(img_filename, mask_filename):
        img = cv2.imread(img_filename)
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        mask = cv2.imread(mask_filename)
        gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(nfeatures = 750 ,WTA_K = 3, edgeThreshold = 31, patchSize = 31, fastThreshold = 40)
        kp = orb.detect(gray_img, gray_mask)
        #print(len(kp))
        return kp, gray_img
        