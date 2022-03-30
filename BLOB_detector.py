#Binary Large Object Detector Module

import cv2

#https://learnopencv.com/blob-detection-using-opencv-python-c/

class BLOB_detector:
    def detect(image, mask=None):

        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = 1
        params.maxThreshold = 1000

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 100

        # Filter by Circularity
        params.filterByCircularity = False
        params.minCircularity = 0.1

        # Filter by Convexity
        params.filterByConvexity = False
        params.minConvexity = 0.87

        # Filter by Inertia
        params.filterByInertia = False
        params.minInertiaRatio = 0.001

        # Create a detector with the parameters
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(image, mask)
        return keypoints

    def mser(self,cv_image):

        delta = 5
        min_area = 60
        max_area = 14400

        max_variation = float(0.25),
        min_diversity = 2,
        max_evolution = 200,
        area_threshold = 1.01,
        min_margin = 0.003,
        edge_blur_size = 5 


        vis = cv_image.copy()
        mser = cv2.MSER_create(delta,min_area,max_area,0.25,.2,200,1.01,0.003,5)
        #regions, _ = mser.detectRegions(cv_image)
        kp = mser.detect(cv_image)
        return kp