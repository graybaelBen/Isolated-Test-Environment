import numpy as np
import cv2 

class SIFT_detector:

    def detect(image, mask):
        #cv2.imshow("Erosion->Dialation", mask)
        #cv2.waitKey(0)
        # for manual image usage:
        #img_filename = 'Batch1\B1.1\images\02_Station04_Camera1_2012-7-5_17-10-22(0).JPG'
        #mask_filename = 'Batch1\B1.1\masks\02_Station04_Camera1_2012-7-5_17-10-22(0).BMP'
        # print(img_filename)
        #data = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        #data = (data - np.min(data)) * 255 / np.max(data)
        #image = data.astype(np.uint8)
       # print(image)

        sift = cv2.SIFT_create(nfeatures = 0, #0
                                nOctaveLayers = 3, #3
                                contrastThreshold = 0.4, #0.4
                                edgeThreshold = 10, #10
                                sigma = 1.6 ) #1.6)
        kp = sift.detect(image, mask)
        print(len(kp))

        # draw keypoints
        #img = cv.drawKeypoints(gray,kp,img)
        #cv.imwrite('sift_keypoints.jpg',img)

        return kp
