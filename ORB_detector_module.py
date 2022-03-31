import numpy as np
import cv2 

class ORB_detector:
    def detect(img, mask):
        # img = cv2.imread(img_filename)
        # gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # mask = cv2.imread(mask_filename)
        # gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create()
        kp = orb.detect(img, mask)
        #print(len(kp))
        return kp
        