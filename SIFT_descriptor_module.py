import cv2

class SIFT_descriptor:

    def descript(image, kp):
        sift = cv2.SIFT.create()
        kp, des = sift.compute(image,kp)
        print('SIFT: kp=%d, descriptors=%s' % (len(kp), des.shape))
        return kp, des