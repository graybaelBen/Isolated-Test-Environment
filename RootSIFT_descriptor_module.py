# import the necessary packages
from rootsift import RootSIFT
from SIFT_detector_module import SIFT_detector
import cv2 

""" # load the image we are going to extract descriptors from and convert
# it to grayscale
image = cv2.imread("example.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# detect Difference of Gaussian keypoints in the image
detector = cv2.FeatureDetector_create("SIFT")
kps = detector.detect(gray) """

kp, gray = SIFT_detector.SIFT_detect()
# extract normal SIFT descriptors
extractor = cv2.DescriptorExtractor_create("SIFT")
(kp, descs) = extractor.compute(gray, kp)
print('SIFT: kps=%d, descriptors=%s' % (len(kp), descs.shape))
# extract RootSIFT descriptors
rs = RootSIFT()
(kp, descs) = rs.compute(gray, kp)
print('RootSIFT: kp=%d, descriptors=%s' % (len(kp), descs.shape))