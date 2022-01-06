import numpy as np
import cv2
from SIFT_detector_module import SIFT_detector

""" filename = ' '
img = cv.imread(filename)
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
sift = cv.SIFT_create()
kp = sift.detect(gray,None)
img = cv.drawKeypoints(gray,kp,img)
cv.imwrite('sift_keypoints.jpg',img) """

sift = cv2.SIFT.create()
detector = SIFT_detector
kp, gray = detector.SIFT_detect()
kp, des = sift.compute(gray,kp)
print('SIFT: kp=%d, descriptors=%s' % (len(kp), des.shape))