# adapted from opencv flann w/ sift tutorial
# https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html

import numpy as np
import cv2
from matplotlib import pyplot as plt
from SIFT_descriptor_module import SIFT_descriptor
import os

directory = 'imgdir'

for image in os.listdir(directory):
    fn1 = os.path.join(directory,image)
    for compare in os.listdir(directory):
        fn2 = os.path.join(directory, compare)

print(fn1, '\n', fn2, '\n\n')

def matcher(fn1, fn2):
    # find the keypoints and descriptors with SIFT
    descriptor = SIFT_descriptor
    kp1, des1 = descriptor.SIFT_descript(fn1)
    kp2, des2 = descriptor.SIFT_descript(fn2)

    # everything above here is used for testing the flann implementation below
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(des1,des2,k=2)
    print(len(matches))

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    counter =0
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
            counter += 1

    print('after ratio test', counter)

    #draw matches on images
'''
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

img1 = cv2.imread('midi.jpg',) # queryImage
img2 = cv2.imread('desk.jpg') # trainImage

img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

plt.imshow(img3,),plt.show()
'''