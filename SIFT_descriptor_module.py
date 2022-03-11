import numpy as np
import cv2
from SIFT_detector_module import SIFT_detector

class SIFT_descriptor:

    def descript(image, kp):


        sift = cv2.SIFT.create()
        kp, des = sift.compute(image,kp)
        print('SIFT: kp=%d, descriptors=%s' % (len(kp), des.shape))
        return kp, des


''' 
filename = ' '
img = cv.imread(filename)
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
sift = cv.SIFT_create()
kp = sift.detect(gray,None)
img = cv.drawKeypoints(gray,kp,img)
cv.imwrite('sift_keypoints.jpg',img) 

sift = cv2.SIFT.create()
detector = SIFT_detector
kp, gray = detector.SIFT_detect()
kp, des = sift.compute(gray,kp)
print('SIFT: kp=%d, descriptors=%s' % (len(kp), des.shape))
'''

#ORB
'''
class SIFT_descriptor:

    def descript(gray, kp):
        sift = cv2.SIFT.create(nfeatures = 0, #0
                                nOctaveLayers = 3, #3
                                contrastThreshold = 0.4, #0.4
                                edgeThreshold = 10, #10
                                sigma = 1.6)
        #
        kp, des = sift.compute(gray,kp)
        #print('SIFT: kp=%d, descriptors=%s' % (len(kp)))#, des.shape))
        return kp, des

#detector = SIFT_detector
#kp, gray = detector.SIFT_detect(img, mask)
'''