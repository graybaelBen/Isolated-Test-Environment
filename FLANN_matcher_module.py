# adapted from opencv flann w/ sift tutorial
# https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html

import numpy as np
import cv2
from matplotlib import pyplot as plt
from SIFT_descriptor_module import SIFT_descriptor
import os
import csv

class FLANN_matcher:
    
    # pass in relative file path of current directory such as Batch1/B1.1
    def match(des1, des2, image, compare, kp1, kp2, current_dir):

        image1 = cv2.imread(current_dir+image)
        image2 = cv2.imread(current_dir+compare)
    
        # FLANN parameters
        #FLANN_INDEX_KDTREE = 0
        FLANN_INDEX_LSH = 6
        #index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 6, key_size = 12, multi_probe_level = 1)
        search_params = dict(checks=50)   # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params,search_params)

        matches = flann.knnMatch(des1,des2,k=2)

        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]

        # ratio test as per Lowe's paper
        matchCount = 0
        for i, pair in enumerate(matches):
            try:
                m, n = pair
                if m.distance < 0.7*n.distance:
                    matchesMask[i]=[1,0]
                    matchCount += 1
            except ValueError:
                pass


        #drawing code
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matchesMask,
                           flags=cv2.DrawMatchesFlags_DEFAULT)

        matchesDrawn = cv2.drawMatchesKnn(image1,kp1,image2,kp2, matches,None,**draw_params)
        results = os.path.join(current_dir,"/results",compare)
        cv2.imwrite(results,matchesDrawn)

        return matchCount

#draw matches on images
'''
BACKUP ARCHIVE

#drawing code
img= 'BatchD/'+ image
cmp = 'BatchD/'+compare
draw_params = dict(matchColor = (0,255,0),
        singlePointColor = (255,0,0),
        matchesMask = matchesMask,
        flags = 0)
pic1 = cv2.imread(img) # queryImage
pic2 = cv2.imread(cmp) # trainImage
pic3 = cv2.drawMatchesKnn(pic1,kp1,pic2,kp2, matches,None,**draw_params)
out = "BatchDR/"+compare
cv2.imwrite(out,pic3)



draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)
pic1 = cv2.imread('midi.jpg') # queryImage
pic2 = cv2.imread('desk.jpg') # trainImage
pic3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
plt.imshow(img3,),plt.show()
'''

# find the keypoints and descriptors with SIFT (manual use)
'''
descriptor = SIFT_descriptor
kp1, des1 = descriptor.SIFT_descript(img1, mask1)
kp2, des2 = descriptor.SIFT_descript(img2, mask2)
'''

# old runner (manual use)
'''
kpCountArray = []
matchCountArray = []
imgDirArr = os.listdir(imgdir)
maskDirArr = os.listdir(maskdir)
loopcount = 0
for image in imgDirArr:
    img1 = os.path.join(imgdir, image)
    mask1 = os.path.join(maskdir, maskDirArr[imgDirArr.index(image)])
    for compare in imgDirArr[imgDirArr.index(image)+1:]:
        img2 = os.path.join(imgdir, compare)
        mask2 = os.path.join(maskdir, maskDirArr[imgDirArr.index(compare)])
        
        #print(img1, '\n', img2, '\n\n')
        
        kp1count, kp2count, matchCount = FLANN_match(img1, mask1, img2, mask2)
        kpCountArray.append(kp1count)
        kpCountArray.append(kp2count)
        matchCountArray.append(matchCount)
        
        print(matchCount)
        loopcount +=1

print(loopcount)
print(len(kpCountArray))
print(len(matchCountArray))
'''