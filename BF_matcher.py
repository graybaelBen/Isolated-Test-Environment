import cv2
import numpy as np

class BF_matcher:
    
    def match(des1, des2, image1, image2, kp1, kp2):

        des1 = np.asarray(des1, np.float32)
        des2 = np.asarray(des2, np.float32)



        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2,k=2)

        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for _ in range(len(matches))]

        # Apply ratio test
        matchCount = 0
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
               # matchesMask[i]=[1,0]
                matchCount += 1

        #drawing keypoint matches
        if len(matches)!=0:
            print(">0", len(matches))
        elif len(matches) ==0:
            print("0", matches)

        draw_params = dict(matchColor=-1,
                           singlePointColor=(255, 0, 0),
                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        drawnMatches = cv2.drawMatchesKnn(image1, kp1, image2, kp2, good, None, **draw_params)

        return matchCount, drawnMatches