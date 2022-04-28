import pyhesaff
from pyhesaff._pyhesaff import grab_test_imgpath
from pyhesaff._pyhesaff import argparse_hesaff_params
import os
import cv2
import ubelt as ub
import numpy as np
import scipy as sp
from scipy import linalg

class HessianAffine:
    def __init__(self):
        self.kpts = []
        self.vecs = []

    def detect(self,image, mask):
        # img_fpath = grab_test_imgpath(ub.argval('--fname', default='astro.png'))
        current_dir = os.path.join('Demo','D1.1')
        imgdir = os.path.join(current_dir,'images')
        img_fpath = os.path.join(imgdir, '02__Station13__Camera1__2012-9-13__2-21-36(2).JPG')
        img_fpath2 = os.path.join(imgdir, '02__Station32__Camera1__2012-7-14__4-48-10(7).JPG')
        print(img_fpath)
        kwargs = argparse_hesaff_params()
        print('kwargs = %r' % (kwargs,))

        # removed tuple ( )
        self.kpts, self.vecs = pyhesaff.detect_feats(img_fpath, **kwargs)

        return self.kpts
     
    def descript(self, image, kp):
        return self.kpts, self.vecs
    
    def drawKeypoints():
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
            center, axes, degs = self.kpToEllipse(kp)
            print(axes)
            cv2.ellipse(img_kp, center, axes, degs, 0, 360, (0,0,255), )

        #Demo code
        #img_kp = process.draw_cross_keypoints(image, kp, color=(120,157,187))
        cv2.imwrite("/mnt/c/Users/dudeb/OneDrive/Documents/GitHub/Isolated-Test-Environment/test.png", img_kp)
        print("done")

    def match(des1, des2, img_fpath1, img_fpath2, kp1, kp2):
        '''
        Take in relative image path, not actual image
        '''
        # img_fpath = grab_test_imgpath(ub.argval('--fname', default='astro.png'))
        print(img_fpath1)
        kwargs = argparse_hesaff_params()
        print('kwargs = %r' % (kwargs,))

        (kpts, vecs) = pyhesaff.detect_feats(img_fpath1, **kwargs)
        (kpts2, vecs2) = pyhesaff.detect_feats(img_fpath2, **kwargs)
        imgBGR = cv2.imread(img_fpath1)

        # vecs[0], vecs[1]
        # print(vecs[0])
        # print(vecs[0].size)

        vecs1 = np.asarray(vecs1, np.float32)
        vecs2 = np.asarray(vecs2, np.float32)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(vecs1,vecs2,k=2)
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

        drawnMatches = self.drawMatches(image1,image2)
        
        return matchCount, drawnMatches
        
    def drawMatches():



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