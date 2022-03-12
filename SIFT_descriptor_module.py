import numpy as np
import cv2
from SIFT_detector_module import SIFT_detector

class SIFT_descriptor:

    def descript(image, kp):
        sift = cv2.SIFT.create()
        kp, des = sift.compute(image,kp)
        print('SIFT: kp=%d, descriptors=%s' % (len(kp), des.shape))
        return kp, des