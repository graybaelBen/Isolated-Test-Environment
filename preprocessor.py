import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

class processor:
    def threshold(imgdir, maskdir, savedir):
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

            cv2.imwrite(os.path.join(savedir, image), thresh )

    def mask(imgdir, maskdir, savedir):

        imgDirArr = os.listdir(imgdir)
        maskDirArr = os.listdir(maskdir)
        saveDirArr = os.listdir(savedir)

        for image in imgDirArr:
            img = cv2.imread(os.path.join(imgdir, image),0)
            mask = cv2.imread(os.path.join(maskdir, maskDirArr[imgDirArr.index(image)]),0)
            #cv2.imshow("pic", img)
            #cv2.imshow("mask", mask)

            combo = cv2.bitwise_and(img, img, mask=mask)
            img = combo
            #cv2.imshow("mask on img", masked)
            cv2.imwrite(os.path.join(savedir, image), combo)