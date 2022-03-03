import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

class processor:
    def threshold(imgdir, maskdir):
        saveDir = "Batches\Batch1\processed"
        imgDirArr = os.listdir(imgdir)
        maskDirArr = os.listdir(maskdir)

        for idx, image in enumerate(imgDirArr):
            print(os.path.join(maskdir, maskDirArr[idx]))
            print(os.path.join(imgdir, image))
            img = cv2.imread(os.path.join(imgdir, image), 0)
            mask = cv2.imread(os.path.join(maskdir, maskDirArr[idx]), 0)
            img = cv2.bitwise_and(img, img, mask=mask) 
            img = cv2.medianBlur(img,5)

            thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                        cv2.THRESH_BINARY,11,2)

            cv2.imwrite(os.path.join(saveDir, image), thresh )
            return saveDir