import numpy as np
import cv2
from ORB_detector_module import ORB_detector

class ORB_descriptor:
    def descript(gray, kp):
        orb = cv2.ORB_create(nfeatures = 750,WTA_K = 3, edgeThreshold = 31, patchSize = 31, fastThreshold=40)
        kp, des = orb.compute(gray,kp)
        print('ORB: kp=%d, descriptors=%s' % (len(kp), des.shape))
        #print(des.shape)
        return kp, des