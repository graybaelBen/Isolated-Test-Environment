import cv2

class FLANN_matcher:
    
    def match(des1, des2, image1, image2, kp1, kp2):
   
       # ORB
       # FLANN_INDEX_LSH = 6
       # index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 6, key_size = 12, multi_probe_level = 1)

        # SIFT
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)

        # Need to draw only good matches, so create a mask
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

        #drawing keypoint matches
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matchesMask,
                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        drawnMatches = cv2.drawMatchesKnn(image1,kp1,image2,kp2, matches,None,**draw_params)

        return matchCount, drawnMatches