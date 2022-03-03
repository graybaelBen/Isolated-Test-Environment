import numpy as np
import cv2 

class SURF_detector:

    def detect(img_filename, mask_filename):
        #filename = '02__Station32__Camera1__2012-8-31__5-17-22(1).JPG'
        #mask = '02__Station32__Camera1__2012-8-31__5-17-22(1).BMP'
        img = cv2.imread(img_filename)
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        mask = cv2.imread(mask_filename)
        gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        surf = cv2.xfeatures2d.SURF_create(400)
        surf.setExtended(True)
        kp = surf.detect(gray_img, gray_mask)
        print(len(kp))
        """ img = cv.drawKeypoints(gray,kp,img)
        cv.imwrite('sift_keypoints.jpg',img) """
        return kp, gray_img
