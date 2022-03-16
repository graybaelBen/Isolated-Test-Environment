#!/usr/bin/env python

'''
Affine invariant feature-based image matching sample.

This sample is similar to find_obj.py, but uses the affine transformation
space sampling technique, called ASIFT [1]. While the original implementation
is based on SIFT, you can try to use SURF or ORB detectors instead. Homography RANSAC
is used to reject outliers. Threading is used for faster affine sampling.

[1] http://www.ipol.im/pub/algo/my_affine_sift/

USAGE
  

  --feature  - Feature to use. Can be sift, surf, orb or brisk. Append '-flann'
               to feature name to use Flann-based matcher instead bruteforce.

  Press left mouse button on a feature point to see its matching point.
'''

# Python 2/3 compatibility
from __future__ import print_function
from cv2 import BORDER_REPLICATE, BORDER_WRAP

import numpy as np
import cv2
import os
import csv
# built-in modules
import itertools as it
from multiprocessing.pool import ThreadPool

# local modules
from common import Timer
from find_obj import init_feature, filter_matches, explore_match


def affine_skew(tilt, phi, img, mask=None):
    '''
    affine_skew(tilt, phi, img, mask=None) -> skew_img, skew_mask, Ai

    Ai - is an affine transform matrix from skew_img to img
    '''
    h, w = img.shape[:2]
    if mask is None:
        mask = np.zeros((h, w), np.uint8)
        mask[:] = 255
    A = np.float32([[1, 0, 0], [0, 1, 0]])
    if phi != 0.0:
        phi = np.deg2rad(phi)
        s, c = np.sin(phi), np.cos(phi)
        A = np.float32([[c,-s], [ s, c]])
        corners = [[0, 0], [w, 0], [w, h], [0, h]]
        tcorners = np.int32( np.dot(corners, A.T) )
        x, y, w, h = cv2.boundingRect(tcorners.reshape(1,-1,2))
        A = np.hstack([A, [[-x], [-y]]])
        img = cv2.warpAffine(img, A, (w, h), flags=cv2.INTER_LINEAR, borderMode=0)
    if tilt != 1.0:
        s = 0.8*np.sqrt(tilt*tilt-1)
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=s, sigmaY=0.01)
        img = cv2.resize(img, (0, 0), fx=1.0/tilt, fy=1.0, interpolation=cv2.INTER_NEAREST)
        A[0] /= tilt
    if phi != 0.0 or tilt != 1.0:
        h, w = img.shape[:2]
        mask = cv2.warpAffine(mask, A, (w, h), flags=cv2.INTER_NEAREST)
    Ai = cv2.invertAffineTransform(A)
    return img, mask, Ai


def affine_detect(detector, img, mask=None, pool=None):
    '''
    affine_detect(detector, img, mask=None, pool=None) -> keypoints, descrs

    Apply a set of affine transormations to the image, detect keypoints and
    reproject them into initial image coordinates.
    See http://www.ipol.im/pub/algo/my_affine_sift/ for the details.

    ThreadPool object may be passed to speedup the computation.
    '''
    params = [(1.0, 0.0)]
    for t in 2**(0.5*np.arange(1,6)):
        for phi in np.arange(0, 180, 72.0 / t):
            params.append((t, phi))

    def f(p):
        t, phi = p
        timg, tmask, Ai = affine_skew(t, phi, img)
        keypoints, descrs = detector.detectAndCompute(timg, tmask)
        for kp in keypoints:
            x, y = kp.pt
            kp.pt = tuple( np.dot(Ai, (x, y, 1)) )
        if descrs is None:
            descrs = []
        return keypoints, descrs

    keypoints, descrs = [], []
    if pool is None:
        ires = it.imap(f, params)
    else:
        ires = pool.imap(f, params)

    for i, (k, d) in enumerate(ires):
        print('affine sampling: %d / %d\r' % (i+1, len(params)), end='')
        keypoints.extend(k)
        descrs.extend(d)

    print()
    return keypoints, np.array(descrs)

if __name__ == '__main__':
    print(__doc__)

    import sys, getopt
    opts, args = getopt.getopt(sys.argv[1:], '', ['feature='])
    opts = dict(opts)
    feature_name = opts.get('--feature', 'sift-flann')
    print(feature_name)

    imgdir = 'Batch1/Batch1.3/img'
    maskdir = 'Batch1/Batch1.3/mask'
    patchedir = 'Batch1/Batch1.3/patched'
    # graydir = 'BatchDG'

    imgDirArr = os.listdir(imgdir)
    maskDirArr = os.listdir(maskdir)
    patchDirArr = os.listdir(patchedir)
    # grayDirArr = os.listdir(graydir)
    
    """
    for image in imgDirArr:
        img = cv2.imread(os.path.join(imgdir, image), 0)
        mask2 = cv2.imread(os.path.join(maskdir, maskDirArr[imgDirArr.index(image)]), 0)
        masked = cv2.bitwise_and(img, img, mask=mask2)
        img2 = masked
        print(type(img))
        img = cv2.resize(img2, (960, 480))
        out = patchedir + image
        # ^^ out might be broken like this, try Batch1/Batch1.1/patched or whatever batch it is
        cv2.imwrite(out,img)
    """    
    detector, matcher = init_feature(feature_name)

    # if img1 is None:
    #     print('Failed to load fn1:', fn1)
    #     sys.exit(1)

    # if img2 is None:
    #     print('Failed to load fn2:', fn2)
    #     sys.exit(1)

    # if detector is None:
    #     print('unknown feature:', feature_name)
    #     sys.exit(1)

    print('using', feature_name)

    pool=ThreadPool(processes = cv2.getNumberOfCPUs())
    matchCountArr = []
    def match_and_draw(win, img1, img2, kp1, kp2, desc1, desc2):
        with Timer('matching'):
            raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2) #2
        p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)
        if len(p1) >= 4:
            H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
            print('%d / %d  inliers/matched' % (np.sum(status), len(status)))
            matchCount = np.sum(status)
            matchCountArr.append(matchCount)
            # do not draw outliers (there will be a lot of them)
            kp_pairs = [kpp for kpp, flag in zip(kp_pairs, status) if flag]
        else:
            H, status = None, None
            print('%d matches found, not enough for homography estimation' % len(p1))

        vis = explore_match(win, img1, img2, kp_pairs, None, H)

    for patched in patchDirArr:
        img1 = cv2.imread(os.path.join(patchedir, patched), 0)
        kp1, desc1 = affine_detect(detector, img1, pool = pool)
        for compare in patchDirArr[patchDirArr.index(patched)+1:]:
            img2 = cv2.imread(os.path.join(patchedir, compare), 0)
            kp2, desc2 = affine_detect(detector, img2, pool = pool)
            print('img1 - %d features, img2 - %d features' % (len(kp1), len(kp2)))
            match_and_draw(patched+"_"+compare, img1, img2, kp1, kp2, desc1, desc2)
            #match_and_draw('affine find_obj', img1, img2, kp1, kp2, desc1, desc2)
            cv2.waitKey()
            cv2.destroyAllWindows()

    np.transpose(matchCountArr)

    with open('results.csv', 'w', newline = '') as f:
        writer = csv.writer(f)
        
        for matchCount in matchCountArr:
            writer.writerow([matchCount])
    print('done!')