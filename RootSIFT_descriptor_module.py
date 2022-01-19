# import the necessary packages
#from rootsift import RootSIFT
from SIFT_detector_module import SIFT_detector
import numpy as np
import cv2 


class RootSIFT_descriptor:
  
    def compute(self, kp, des, eps=1e-7):
        # compute SIFT descriptors
        #detector = SIFT_detector
        #(kp, des) = detector.detect(image, kp)
        # if there are no keypoints or descriptors, return an empty tuple
        if len(kp) == 0:
            return ([], None)
        # apply the Hellinger kernel by first L1-normalizing and taking the
        # square-root
        #des /= (des.sum(axis=1, keepdims=True) + eps)
        des = np.sqrt(des)
        #descs /= (np.linalg.norm(descs, axis=1, ord=2) + eps)
        # return a tuple of the keypoints and descriptors
        return kp, des
    def descript(self, img, mask):
        #kp, gray = SIFT_detector.SIFT_detect()
        # extract normal SIFT descriptors
        # extractor = cv2.DescriptorExtractor_create("SIFT")
        detector = SIFT_detector
        sift = cv2.SIFT_create()
        kp, gray = detector.SIFT_detect(img, mask)
        kp, des = sift.compute(gray, kp)
        #print('SIFT: kps=%d, descriptors=%s' % (len(kp), des.shape))
        #print(des[0,10:20])
        # extract RootSIFT descriptors
        kp, des = self.compute(self, kp, des)
        #print('RootSIFT: kp=%d, descriptor size=%s' % (len(kp), des.shape))
        #print(des[0,10:20])
        return kp, des

    

""" # load the image we are going to extract descriptors from and convert
# it to grayscale
image = cv2.imread("example.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# detect Difference of Gaussian keypoints in the image
detector = cv2.FeatureDetector_create("SIFT")
kps = detector.detect(gray) """




#filename = '02__Station04__Camera1__2012-7-5__12-17-42(0).JPG'
#maskfile = '02__Station04__Camera1__2012-7-5__12-17-42(0).BMP'
# load the image we are going to extract descriptors from and convert
# it to grayscale
#image = cv2.imread(filename, 0)
#mask = cv2.imread(maskfile, 0)
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#grayM = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

#rs = RootSIFT_descriptor


#kp1, des1 = rs.descript(rs, filename, maskfile)