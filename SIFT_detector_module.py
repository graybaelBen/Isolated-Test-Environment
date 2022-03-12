import cv2 

class SIFT_detector:

    def detect(image, mask):

        sift = cv2.SIFT_create() # removing the parameters was the only change!!
        kp = sift.detect(image, mask)
        print(len(kp))
        return kp
