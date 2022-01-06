import numpy as np
import cv2 

class SIFT_detector:

    def SIFT_detect():
        filename = '02__Station32__Camera1__2012-8-31__5-17-22(1).JPG'
        maskfile = '02__Station32__Camera1__2012-8-31__5-17-22(1).BMP'
        mask = cv2.imread(maskfile)
        img = cv2.imread(filename)
        grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        grayMask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        kp = sift.detect(grayImg, grayMask)
        """ img = cv.drawKeypoints(gray,kp,img)
        cv.imwrite('sift_keypoints.jpg',img) """
        return kp, grayImg
