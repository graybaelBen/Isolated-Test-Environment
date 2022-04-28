import pyhesaff
from pyhesaff._pyhesaff import grab_test_imgpath
from pyhesaff._pyhesaff import argparse_hesaff_params
import os
import cv2
import ubelt as ub
import numpy as np
# import rhino3dm
# from cv2 import Point2f
current_dir = os.path.join('Demo','D1.1')
imgdir = os.path.join(current_dir,'images')
img_fpath = os.path.join(imgdir, '02__Station13__Camera1__2012-9-13__2-21-36(2).JPG')
img_fpath2 = os.path.join(imgdir, '02__Station32__Camera1__2012-7-14__4-48-10(7).JPG')
image1 = cv2.imread(img_fpath)
image2 = cv2.imread(img_fpath2)
# print(img_fpath)

kwargs = argparse_hesaff_params()
# print('kwargs = %r' % (kwargs,))

(kpts, vecs) = pyhesaff.detect_feats(img_fpath, **kwargs)
(kpts2, vecs2) = pyhesaff.detect_feats(img_fpath2, **kwargs)
imgBGR = cv2.imread(img_fpath)

# vecs[0], vecs[1]
# print(vecs[0])
# print(vecs[0].size)

vecs = np.asarray(vecs, np.float32)
vecs2 = np.asarray(vecs2, np.float32)

# vecs2 = vecs

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(vecs,vecs2,k=2)
matchesMask = [[0,0] for _ in range(len(matches))]

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

cvkp = []
cvkp2 = []
for i in range(len(kpts)):
    cvkp.append(cv2.KeyPoint(x = kpts[i][0], y = kpts[i][1], size = 2))
    cvkp2.append(cv2.KeyPoint(x = kpts2[i][0], y = kpts2[i][1], size = 2))
# test = cv2.KeyPoint(x = kpts[0][0], y = kpts[0][1], size = 1)
draw_params = dict(matchColor=-1,
                           singlePointColor=(255, 0, 0),
                           matchesMask=matchesMask,
                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
drawnMatches = cv2.drawMatchesKnn(image1,cvkp,image2,cvkp2, matches,None,**draw_params)
# compared_images = image1+image2
results = os.path.join(current_dir,"results", "test.jpg")
cv2.imwrite(results,drawnMatches)







# print(cvkp)

# print("desc", vecs)
# #cv2.imshow("KeyPoints",imgBGR)
# #cv2.waitKey(0)

""" https://stackoverflow.com/questions/67762285/drawing-sift-keypoints
Draw keypoints as crosses, and return the new image with the crosses. """
img_kp = imgBGR.copy()  # Create a copy of img

# Iterate over all keypoints and draw a cross on evey point.
for kp in kpts:
    # x, y = kp.pt
    x = kp[0]
    y = kp[1]  # Each keypoint as an x, y tuple  https://stackoverflow.com/questions/35884409/how-to-extract-x-y-coordinates-from-opencv-cv2-keypoint-object
    x = int(round(x))  # Round an cast to int
    y = int(round(y))

    # Draw a cross with (x, y) center
    cv2.drawMarker(img_kp, (x, y), color = (250,0,0), markerType=1, markerSize=5, thickness=1, line_type=cv2.LINE_8)
#Demo code
#img_kp = process.draw_cross_keypoints(image, kp, color=(120,157,187))
cv2.imwrite("KeyPoints.jpg",img_kp)
cv2.waitKey(0)

#return img_kp  # Return the image with the drawn crosses.

# if ub.argflag('--show'):
#     # Show keypoints
#     imgBGR = cv2.imread(img_fpath)
#     default_showkw = dict(ori=False, ell=True, ell_linewidth=2,
#                             ell_alpha=.4, ell_color='distinct')
#     print('default_showkw = %r' % (default_showkw,))
#     #import utool as ut
#     #showkw = ut.argparse_dict(default_showkw)
#     #import plottool_ibeis as pt
#     #pt.interact_keypoints.ishow_keypoints(imgBGR, kpts, vecs, **showkw)
#     #pt.show_if_requested()

#     cv2.imshow("Processed Image",imgBGR)
#     cv2.waitKey(0)

