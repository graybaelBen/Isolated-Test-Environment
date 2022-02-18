import cv2
import numpy as np
import argparse
import os

#assign active directories
imgdir = 'Batch1'
maskdir = 'Batch1M'
combodir = 'Batch1C'

imgDirArr = os.listdir(imgdir)
maskDirArr = os.listdir(maskdir)
comboDirArr = os.listdir(combodir)

for image in imgDirArr:
    img = cv2.imread(os.path.join(imgdir, image),0)
    mask = cv2.imread(os.path.join(maskdir, maskDirArr[imgDirArr.index(image)]),0)
    #cv2.imshow("pic", img)
    #cv2.imshow("mask", mask)

    combo = cv2.bitwise_and(img, img, mask=mask)
    img = combo
    #cv2.imshow("mask on img", masked)
    cv2.imwrite(os.path.join(combodir, image), combo)