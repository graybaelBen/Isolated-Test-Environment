import numpy as np
import cv2
from matplotlib import pyplot as plt
from SIFT_descriptor_module import SIFT_descriptor
import os
import csv
class LNBNN_matcher:
    #need nearest neighbor index for all descriptors that belongs to an image Q
    #need 'class' (snow leopard) lookup? I'm thinking instead of defined classes it would be each individual image and their keypoints. This may affect runtime
    
    def getNNIndex():
        return
    
    def classify(descArray, keypArray):
        #for each descriptor (di) that belongs to an image Q
            #for each image in set
            #compare to other descriptors in image pool
        return #'class' (image)
        
    def lnbnn_match(descArray):
        #for every descriptor (di) that belongs to an image Q
            # find the probability distribution of the descriptor given 'class' up to k+1 nearest neighbor (p1,p2,....,pk+1)
            # find the distance ||di-pk+1||^2 (distB)
            # for all 'classes' (images?) found in the k nearest neighbors
                # find the minimum distance (distC) between the descriptor (di) and the most probability distribution(pj) ||di-pj||^2
                # find totals += distC - distB     
        return #argmin totals -> 'class' -> 'snow leopard'
    
    