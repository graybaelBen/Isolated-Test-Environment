import numpy as np
import cv2
from ORB_detector_module import ORB_detector

class ORB_descriptor:
    def ORB_descript(gray, kp):
        orb = cv2.ORB_create()
        kp, des = orb.compute(gray,kp)
        print('ORB: kp=%d, descriptors=%s' % (len(kp), des.shape))
        return kp, des