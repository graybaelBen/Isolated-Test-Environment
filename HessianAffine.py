import pyhesaff
from pyhesaff._pyhesaff import grab_test_imgpath
from pyhesaff._pyhesaff import argparse_hesaff_params
import os
import cv2
import ubelt as ub
import numpy as np
import scipy as sp
from scipy import linalg

# import modules
from FLANN_matcher_module import FLANN_matcher # ins: kp, des; outs:
from preprocessor  import Processor

#descriptor = RootSIFT_descriptor
matcher = FLANN_matcher
# instatiate object of class
process = Processor()

class HessianAffine:
    def __init__(self):
        # storing data here was not used the right way!!
        # a new HA object would have needed to be created for every image
        # if thats the route we want to take

        self.kpts = []
        self.vecs = []

    def run(batch):
        # assign active directories
        current_dir = os.path.join('Batches',batch)

        imgdir = os.path.join(current_dir,'images')
        maskdir = os.path.join(current_dir,'masks')
        processeddir = os.path.join(current_dir,'processed')

        imgDirArr = os.listdir(imgdir)
        maskDirArr = os.listdir(maskdir)

        imgDirArr.sort()
        maskDirArr.sort()

        if not os.path.exists(os.path.join(current_dir,'processed')):
            os.makedirs(os.path.join(current_dir,'processed'))

        # Image Processing
        for idx, img in enumerate(imgDirArr):
            image = cv2.imread(os.path.join(imgdir, img))
            mask = cv2.imread(os.path.join(maskdir, maskDirArr[idx]),0)
            processed = image
            processed = process.grayscale(processed)
            processed = process.mask(processed, mask)
            
            #processed = process.threshold(processed)
            #processed = process.cluster_quantize(processed,6)   
            # image is inverted so erode and dilate are swapped from our perspective
            #processed = process.erode(processed,3)
            #processed = process.dilate(processed,10)

            cv2.imwrite(os.path.join(processeddir, img), processed)


        # COMMENT OUT FOR NO PROCESSING
        imgDirArr = os.listdir(processeddir)

        # Keypoint Detection Description
        kpArray = []
        kpCountArray = []
        desArray = []
        
        for idx, img in enumerate(imgDirArr):
            image = os.path.join(imgdir, img)
            mask = os.path.join(maskdir, maskDirArr[idx])
            kps, des = self.detect_describe(image,mask)
                    
            kps, des = self.reapplyMask(image, mask, kps, des)

            desArray.append(des)
            kpArray.append(kps)
            kpCountArray.append(len(kps))

        #print('kpcount length :', len(kpCountArray))
        #print('descriptor length :', len(desArray))
        if not os.path.exists(os.path.join(current_dir,'results')):
            os.makedirs(os.path.join(current_dir,'results'))
            print("creating results directory")

        # Image Comparison
        matchCountArray = []
        for idx1, img1 in enumerate(imgDirArr):
            # internal loop starts at image in the next index
            for idx2, img2 in enumerate(imgDirArr[idx1+1:]):
                matchCount = 0
                image1 = cv2.imread(os.path.join(imgdir,img1))
                # print(type(image))
                image2 = cv2.imread(os.path.join(imgdir,img2))
                matchCount, drawnMatches = matcher.match(desArray[idx1], desArray[idx1+idx2+1], image1, image2, kpArray[idx1], kpArray[idx1+idx2+1])
                compared_images = img1+img2
        
                results = os.path.join(current_dir,"results", compared_images)
                print(results)
                cv2.imwrite(results,drawnMatches)  
                matchCountArray.append(matchCount)
                
                # print(idx2)
        print(batch, " COMPLETE")
        return matchCountArray, kpCountArray

    def detect_describe(self,img_fpath, mask):
        kwargs = argparse_hesaff_params()
        print('kwargs = %r' % (kwargs,))
        (kpts, vecs) = pyhesaff.detect_feats(img_fpath, **kwargs)
        casted_vecs = np.asarray(vecs, np.float32)
        cvkp = []
        for i in range(len(kpts)):
            cvkp.append(cv2.KeyPoint(x = kpts[i][0], y = kpts[i][1], size = 6))

        return cvkp, casted_vecs

    def detect(self,image, mask):
        #get image path from image
        img_fpath = image
        print(img_fpath)
        kwargs = argparse_hesaff_params()
        print('kwargs = %r' % (kwargs,))

        (self.kpts, self.vecs) = pyhesaff.detect_feats(img_fpath, **kwargs)

        cvkp = []
        for i in range(len(self.kpts)):
            cvkp.append(cv2.KeyPoint(x = self.kpts[i][0], y = self.kpts[i][1], size = 6))

        return cvkp

    def descript(self, image, kp):
        casted_vecs = np.asarray(self.vecs, np.float32)
        return self.kpts, casted_vecs
    
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

    def match(self, des1, des2, img_fpath1, img_fpath2, kp1, kp2):
        '''
        Take in relative image path, not actual image
        '''
        # img_fpath = grab_test_imgpath(ub.argval('--fname', default='astro.png'))
        print(img_fpath1)
        kwargs = argparse_hesaff_params()
        print('kwargs = %r' % (kwargs,))

        (kpts, vecs1) = pyhesaff.detect_feats(img_fpath1, **kwargs)
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

        images1 = cv2.imread(img_fpath1, 0)
        images2 = cv2.imread(img_fpath2, 0)

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


        drawnMatches = self.drawMatches(matchesMask, images1,kp1,images2, kp2, matches)
        
        return matchCount, drawnMatches
        
    def drawMatches(self, matchesMask, image1, kp1, image2, kp2, matches):
        draw_params = dict(matchColor=-1,
                           singlePointColor=(255, 0, 0),
                           matchesMask=matchesMask,
                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        drawnMatches = cv2.drawMatchesKnn(image1,kp1,image2,kp2, matches,None,**draw_params)
        return drawnMatches

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

        # adapted from https://stackoverflow.com/questions/67622830/how-do-i-mask-opencvs-keypoints-post-detection
    def reapplyMask(self, img_fpath, mask, kpts, vecs):
        image = cv2.imread(img_fpath)
        mask = cv2.imread(mask,0)
        good_kp = [] # list of good keypoints
        good_desc = [] # list of good descriptors

        print(image.shape)    
        print(mask.shape)

        for kp, desc in zip(kpts,vecs):
            x, y = kp.pt
            if mask[int(y),int(x)] !=0:
                print(mask[int(y),int(x)])
                good_kp.append(kp)
                good_desc.append(desc)
        good_desc = np.asarray(good_desc, np.float32)
        return good_kp, good_desc