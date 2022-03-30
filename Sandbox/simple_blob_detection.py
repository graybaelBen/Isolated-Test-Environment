import cv2
import numpy as np
import os

#https://learnopencv.com/blob-detection-using-opencv-python-c/


ori1 = cv2.imread('processed/02__Station32__Camera1__2012-7-14__4-48-10(7).JPG')
im1 = cv2.imread("processed/02__Station32__Camera1__2012-7-14__4-48-10(7).JPG", cv2.IMREAD_GRAYSCALE)

ori2 = cv2.imread('processed/02__Station13__Camera1__2012-9-13__2-21-36(2).JPG')
im2 = cv2.imread("processed/02__Station13__Camera1__2012-9-13__2-21-36(2).JPG", cv2.IMREAD_GRAYSCALE)


# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 1
params.maxThreshold = 1000

# Filter by Area.
params.filterByArea = True
params.minArea = 100

# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = False
params.minInertiaRatio = 0.001


# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)
keypoints1 = detector.detect(im1)
keypoints2 = detector.detect(im2)

im_with_keypoints1 = cv2.drawKeypoints(im1, keypoints1, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#cv2.imshow('Original',ori) 
#cv2.imshow('BLOB1',im_with_keypoints1)

im_with_keypoints2 = cv2.drawKeypoints(im2, keypoints2, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#cv2.imshow('Original',ori) 
#cv2.imshow('BLOB2',im_with_keypoints2)



orb = cv2.ORB_create(nfeatures=200)
#kp = orb.detect(img, None)

kp1, des1 = orb.compute(im1, keypoints1)
img1 = cv2.drawKeypoints(im1, kp1, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('ORB1', img1)

kp2, des2 = orb.compute(im2, keypoints2)
img2 = cv2.drawKeypoints(im2, kp2, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('ORB2', img2)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None)
cv2.imwrite(os.path.join("results","matched.jpg"), match_img )
cv2.imshow('Matches', match_img)
cv2.waitKey(0)
