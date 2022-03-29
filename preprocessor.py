import cv2
from cv2 import ellipse
import numpy as np
from matplotlib import pyplot as plt
import os
from PIL import Image
from sklearn.cluster import MiniBatchKMeans


class Processor:

    def threshold(self,image):
       #cv2.ADAPTIVE_THRESH_MEAN_C
       #cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        image = cv2.medianBlur(image,5)
        return cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                        cv2.THRESH_BINARY,7,2)

    def mask(self, image, mask):
        return cv2.bitwise_and(image, image, mask= mask) 

    def grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def erode(self, image, iterations = 1):
        erosion_size = 2
        erosion_shape = 2     
        element = cv2.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                        (erosion_size, erosion_size)) 
        img = cv2.erode(image, element, iterations)
        return img

    def dilate(self, image, iterations = 1):
        dilatation_size = 2
        dilation_shape = 2
        element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                        (dilatation_size, dilatation_size))
        return cv2.dilate(image, element, iterations)

    def draw_cross_keypoints(self, img, keypoints, color):
        """ https://stackoverflow.com/questions/67762285/drawing-sift-keypoints
        Draw keypoints as crosses, and return the new image with the crosses. """
        img_kp = img.copy()  # Create a copy of img

        # Iterate over all keypoints and draw a cross on evey point.
        for kp in keypoints:
            x, y = kp.pt  # Each keypoint as an x, y tuple  https://stackoverflow.com/questions/35884409/how-to-extract-x-y-coordinates-from-opencv-cv2-keypoint-object

            x = int(round(x))  # Round an cast to int
            y = int(round(y))

            # Draw a cross with (x, y) center
            cv2.drawMarker(img_kp, (x, y), color, markerType=cv2.MARKER_CROSS, markerSize=5, thickness=1, line_type=cv2.LINE_8)
        #Demo code
        #img_kp = process.draw_cross_keypoints(image, kp, color=(120,157,187))
        #cv2.imshow("KeyPoints",img_kp)
        #cv2.waitKey(0)

        return img_kp  # Return the image with the drawn crosses.
    
    def cluster_quantize(self, orig_image, n_clusters = 3):
        # https://pyimagesearch.com/2014/07/07/color-quantization-opencv-using-k-means-clustering/


        (h, w) = orig_image.shape[:2]

        # convert the image from the RGB color space to the L*a*b*
        # color space -- since we will be clustering using k-means
        # which is based on the euclidean distance, we'll use the
        # L*a*b* color space where the euclidean distance implies
        # perceptual meaning Source: https://www.xrite.com/blog/lab-color-space
        image = cv2.cvtColor(orig_image,cv2.COLOR_GRAY2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # reshape the image into a feature vector so that k-means
        # can be applied
        image = image.reshape((image.shape[0] * image.shape[1], 3))

        # apply k-means using the specified number of clusters and
        # then create the quantized image based on the predictions
        clt = MiniBatchKMeans(n_clusters)
        labels = clt.fit_predict(image)
        quant = clt.cluster_centers_.astype("uint8")[labels]

        # reshape the feature vectors to images
        quant = quant.reshape((h, w, 3))
        image = image.reshape((h, w, 3))

        # convert from L*a*b* to RGB
        quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
        # display the images and wait for a keypress
        #cv2.imshow("Cluster Quantized", np.hstack([image, quant]))
        #cv2.waitKey(0)
        image = cv2.cvtColor(quant, cv2.COLOR_BGR2GRAY)
        return image


    def quantize(self, image):
        #https://pythonmana.com/2021/11/20211124113510502r.html
        #img1 = cv2.resize(image, fx = 0.5, fy = 0.5, dsize = None) # Resize image
        #img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) # Convert the image to a grayscale image

        img2 = img1 = image

        height = img1.shape[0] #shape[0] The first dimension of the image , Height
        width = img1.shape[1] #shape[1] The second dimension of the image , Width

        # Create a matrix of the same size as the original image
        img3 = np.zeros((width, height, 3), np.uint(8))
        img4 = np.zeros((width, height, 3), np.uint(8))
        img5 = np.zeros((width, height, 3), np.uint(8))
        img6 = np.zeros((width, height, 3), np.uint(8))
        img7 = np.zeros((width, height, 3), np.uint(8))

        # Operate on the values of the original image matrix
        img3 = np.uint8(img2/4) * 4
        img4 = np.uint8(img2/16) * 16
        img5 = np.uint8(img2/32) * 32
        img6 = np.uint8(img2/64) * 64
        img7 = np.uint8(img2 >= 128) * 128

        cv2.imshow("8 Levels", img5)
        cv2.imshow("4 Levels", img6)
        cv2.imshow("2 Levels", img7)

        return Image.fromarray(img6)

        '''
        
        print(img1.shape)
        print(width, height)
        cv2.namedWindow("W0")
        cv2.imshow("W0", img2)
        cv2.waitKey(delay = 0)

        #plt.rcParams['font.family'] = 'SimHei' # Change the global Chinese font to bold
        #plt.rcParams['axes.unicode_minus'] = False # Normal indicates a minus sign
        # Display the resulting image
        title = [' original image ', ' Quantified as 64 Share ', ' Quantified as 16 Share ', ' Quantified as 8 Share ', ' Quantified as 4 Share ', ' Quantified as 2 Share '] # Subgraph title
        img = [img2, img3, img4, img5, img6, img7]
        for i in range(6):
            plt.subplot(2, 3, i + 1) #python List from 0 Start counting , So here i+1
            plt.imshow(img[i], 'gray')
            plt.title(title[i])
            plt.xticks([]),plt.yticks([])
            plt.show()
        '''

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