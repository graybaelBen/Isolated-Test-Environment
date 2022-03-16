import cv2 

class SIFT_detector:

    def detect(image, mask):

        sift = cv2.SIFT_create() 
        kp = sift.detect(image, mask)
        print(len(kp))
        return kp