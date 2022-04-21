import pyhesaff
from pyhesaff._pyhesaff import grab_test_imgpath
from pyhesaff._pyhesaff import argparse_hesaff_params
import ubelt as ub
import cv2
import numpy as np
import scipy as sp
from scipy import linalg

def kpToEllipse(kp):
    x = round(int(kp[0]))
    y = round(int(kp[1]))
    a = kp[2]
    b = kp[3]
    c = kp[4]
    matA = np.array([[a,b/2],[b/2,c]]) # creates matrix A
    w,v = np.linalg.eig(matA) # gets eigenvalues and vectors
    v= -1*v
    det = np.linalg.det(v)
    
    theta = v[1][0]/v[0][0]
    degs = theta*180/np.pi

    center = (x,y)
    xax = 200*(1/w[0])**.5
    yax = 200*(1/w[1])**.5
    print(xax, yax)
    axes = (int(xax), int(yax))
    
    return center, axes, degs

#img_fpath = grab_test_imgpath(ub.argval('--fname', default='astro.png'))
img_fpath = 'Batches/pristine2send/processed/02__Station13__Camera1__2012-9-13__2-21-40(9).JPG'
kwargs = argparse_hesaff_params()
print('kwargs = %r' % (kwargs,))

(kpts, vecs) = pyhesaff.detect_feats(img_fpath, **kwargs)
imgBGR = cv2.imread(img_fpath)

print(kpts[0:10])
#print("desc", vecs[])
#cv2.imshow("KeyPoints",imgBGR)
#cv2.waitKey(0)



""" https://stackoverflow.com/questions/67762285/drawing-sift-keypoints
Draw keypoints as crosses, and return the new image with the crosses. """
img_kp = imgBGR.copy()  # Create a copy of img

# Iterate over all keypoints and draw a cross on evey point.
for kp in kpts:
    # Each keypoint as an x, y tuple  https://stackoverflow.com/questions/35884409/how-to-extract-x-y-coordinates-from-opencv-cv2-keypoint-object
    # Round and cast to int
    
    #kp coords
    x = round(int(kp[0]))
    y = round(int(kp[1]))
    
    #ellipse where a(x-u)(x-u)+2b(x-u)(y-v)+c(y-v)(y-v)=1
    a = round(int(kp[2]))
    b = round(int(kp[3]))
    c = round(int(kp[4]))

    # Draw a cross with (x, y) center
    cv2.drawMarker(img_kp, (x, y), color = (255,0,0), markerType=cv2.MARKER_CROSS, markerSize=5, thickness=1, line_type=cv2.LINE_8)
    center, axes, degs = kpToEllipse(kp)
    print(axes)
    cv2.ellipse(img_kp, center, axes, degs, 0, 360, (0,0,255), )

#Demo code
#img_kp = process.draw_cross_keypoints(image, kp, color=(120,157,187))
cv2.imwrite("/mnt/c/Users/dudeb/OneDrive/Documents/GitHub/Isolated-Test-Environment/test.png", img_kp)
print("done")

#return img_kp  # Return the image with the drawn crosses.



"""
if ub.argflag('--show'):
    # Show keypoints
    imgBGR = cv2.imread(img_fpath)
    default_showkw = dict(ori=False, ell=True, ell_linewidth=2,
                            ell_alpha=.4, ell_color='distinct')
    print('default_showkw = %r' % (default_showkw,))
    #import utool as ut
    #showkw = ut.argparse_dict(default_showkw)
    #import plottool_ibeis as pt
    #pt.interact_keypoints.ishow_keypoints(imgBGR, kpts, vecs, **showkw)
    #pt.show_if_requested()

    cv2.imshow("Processed Image",imgBGR)
    cv2.waitKey(0)
"""