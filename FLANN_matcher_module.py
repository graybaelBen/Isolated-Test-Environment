import cv2
import numpy as np

class FLANN_matcher:
    
    def match(des1, des2, image1, image2, kp1, kp2):
   
       # ORB
        FLANN_INDEX_LSH = 6
        #index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 6, key_size = 12, multi_probe_level = 1)

        # SIFT
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)


        des1 = np.asarray(des1, np.float32)
        des2 = np.asarray(des2, np.float32)

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
        draw_params = dict(matchColor=-1,
                           singlePointColor=(255, 0, 0),
                           matchesMask=matchesMask,
                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        drawnMatches = cv2.drawMatches(image1,kp1,image2,kp2, matches,None,matchColor=-1)

        return matchCount, drawnMatches

    def ransac(self, kp1, kp2, strong_matches):
        MIN_MATCH_COUNT = 10
        if len(strong_matches) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in strong_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in strong_matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            # h,w,d = img1.shape
            # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            # dst = cv2.perspectiveTransform(pts,M)
            # img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

            best_matches = []
            for index, maskI in enumerate(matchesMask):
                if maskI == 1:
                    best_matches.append(strong_matches[index])
            return best_matches

        else:
            print("Not enough matches are found - {}/{}".format(len(strong_matches), MIN_MATCH_COUNT))
            matchesMask = None
            return strong_matches
