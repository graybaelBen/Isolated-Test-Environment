import os
import sys
import cv2
import numpy as np

from PIL import Image

"""
    This is a simple example on how to use OpenCV for MSER
    https://fossies.org/linux/opencv/samples/python/mser.py
    https://github.com/Belval/opencv-mser/blob/master/mser.py
"""

def mser(cv_image):

    delta = 5
    min_area = 60
    max_area = 14400

    max_variation = float(0.25),
    min_diversity = 2,
    max_evolution = 200,
    area_threshold = 1.01,
    min_margin = 0.003,
    edge_blur_size = 5 


    vis = cv_image.copy()
    mser = cv2.MSER_create(delta,min_area,max_area,0.25,.2,200,1.01,0.003,5)
    regions, _ = mser.detectRegions(cv_image)
    

    '''
    
        for p in regions:
        xmax, ymax = np.amax(p, axis=0)
        xmin, ymin = np.amin(p, axis=0)
        cv2.rectangle(vis, (xmin,ymax), (xmax,ymin), (0, 255, 0), 1)
    '''

    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    cv2.polylines(vis, hulls, 1, (0, 255, 0))

    return vis

def main():
    file_path = "processed/02__Station32__Camera1__2012-7-14__4-48-10(7).JPG"
    save_path = "results/02__Station32__Camera1__2012-7-14__4-48-10(7).JPG"

    cv2.imwrite(save_path, mser(cv2.imread(file_path, 0)))

if __name__=='__main__':
    main()