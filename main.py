# Isolated Test Environment Main
import cv2

# Core Structure
import detectionModule # inputs: picture, mask; outputs: keypoint x,y
import descriptionModule # inputs: keypoint x,y; outputs: descriptors[]
import matchingModule # inputs: descriptors[]imgA, descriptors[]imgB; outputs: match count, images with drawn matches

def main(pictures, masks, detector, descriptor, matcher): # pictures and masks w/ correlated indeces
    kpCoords[] # array for arrays of keypoint x,y
    kpDescVecs[] # array for arrays of keypoint descriptors

    for pic in (pictures):
        kpCoords[pic] = detector(pic,masks[pic])
        kpDescVecs[pic] = descriptor(kpCoords[pic])

    
    
    return

