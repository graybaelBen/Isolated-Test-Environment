import pyhesaff
from pyhesaff._pyhesaff import grab_test_imgpath
from pyhesaff._pyhesaff import argparse_hesaff_params
import cv2
import ubelt as ub

img_fpath = grab_test_imgpath(ub.argval('--fname', default='astro.png'))
kwargs = argparse_hesaff_params()
print('kwargs = %r' % (kwargs,))

(kpts, vecs) = pyhesaff.detect_feats(img_fpath, **kwargs)
imgBGR = cv2.imread(img_fpath)
print(kpts)
print("desc", vecs)
#cv2.imshow("KeyPoints",imgBGR)
#cv2.waitKey(0)

'''

""" https://stackoverflow.com/questions/67762285/drawing-sift-keypoints
Draw keypoints as crosses, and return the new image with the crosses. """
img_kp = imgBGR.copy()  # Create a copy of img

# Iterate over all keypoints and draw a cross on evey point.
for kp in kpts:
    x, y = kp.pt  # Each keypoint as an x, y tuple  https://stackoverflow.com/questions/35884409/how-to-extract-x-y-coordinates-from-opencv-cv2-keypoint-object

    x = int(round(x))  # Round an cast to int
    y = int(round(y))

    # Draw a cross with (x, y) center
    cv2.drawMarker(img_kp, (x, y), color, markerType=cv2.MARKER_CROSS, markerSize=5, thickness=1, line_type=cv2.LINE_8)
#Demo code
#img_kp = process.draw_cross_keypoints(image, kp, color=(120,157,187))
cv2.imshow("KeyPoints",img_kp)
cv2.waitKey(0)

#return img_kp  # Return the image with the drawn crosses.

'''

'''
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
'''