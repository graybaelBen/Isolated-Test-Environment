import cv2
from cv2 import ellipse
import numpy as np
from matplotlib import pyplot as plt
import os

class processor:

    def threshold(image):
       #cv2.ADAPTIVE_THRESH_MEAN_C
       #cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        image = cv2.medianBlur(image,5)
        return cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                        cv2.THRESH_BINARY,7,2)

    def mask(image, mask):
        return cv2.bitwise_and(image, image, mask= mask) 

    def grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def erode(image):
        erosion_size = 1
        erosion_shape = 2     
        element = cv2.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                        (erosion_size, erosion_size))  
        return cv2.erode(image, element, iterations=1)

    def dilate(image):
        dilatation_size = 1
        dilation_shape = 2
        element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                        (dilatation_size, dilatation_size))
        return cv2.dilate(image, element, iterations=1)

    def erosionDialation(imgPath,iterations=1):
        # Python program to demonstrate erosion and
        # dilation of images.

        # Reading the input image
        img = cv2.imread(imgPath, 0)
        
        # Taking a matrix of size 5 as the kernel
        kernel = np.ones((5,5), np.uint8)
        
        # The first parameter is the original image,
        # kernel is the matrix with which image is
        # convolved and third parameter is the number
        # of iterations, which will determine how much
        # you want to erode/dilate a given image.
        img_erosion = cv2.erode(img, kernel, iterations=1)
        img_dilation = cv2.dilate(img, kernel, iterations=1)
        
        cv2.imshow('Input', img)
        cv2.imshow('Dilation', img_erosion)
        cv2.imshow('Erosion', img_dilation)
        img_erosion_dilation = cv2.erode(img_dilation, kernel, iterations=1)
        cv2.imshow("Erosion->Dialation", img_erosion_dilation)
        
        cv2.waitKey(0)

    '''
    
     def threshold(imgdir, maskdir, savedir):
        imgDirArr = os.listdir(imgdir)
        maskDirArr = os.listdir(maskdir)
        imgDirArr.sort()
        maskDirArr.sort()

        for image in imgDirArr:

            img = cv2.imread(os.path.join(imgdir, image), 0)
            mask = cv2.imread(os.path.join(maskdir, maskDirArr[imgDirArr.index(image)]), 0)
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

                '''