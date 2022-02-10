import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

img = cv2.imread('test_img\\02__Station13__Camera1__2012-9-13__2-21-36(5).JPG',0)
mask = cv2.imread('test_mask\\02__Station13__Camera1__2012-9-13__2-21-36(5).BMP',0)
# cv2.imshow("mask", mask)
masked = cv2.bitwise_and(img, img, mask=mask)
img = masked

img = cv2.medianBlur(img,5)

ret,th1 = cv2.threshold(img,110,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

titles = ['Original Image', 'Global Thresholding (v = 100)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]

saveDir = "processed_images"
#cv2.imwrite(os.path.join(graydir, image), gray_img)

for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

    title = titles[i]
    image = images[i]
    cv2.imwrite(os.path.join(saveDir, title+".jpg"), image )

plt.show()



'''
import numpy as np
import cv2

# Apply GaussianBlur to reduce image noise if it is required
# if is_reduce_noise:
#   image = cv2.GaussianBlur(image, (5, 5), 0)


image = 'test_img\\02__Station13__Camera1__2012-9-13__2-21-36(5).JPG'

# Applying Otsu's method setting the flag value into cv.THRESH_OTSU.
# Use a bimodal image as an input.`
# Optimal threshold value is determined automatically.`
otsu_threshold, image_result = cv2.threshold(
    image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
)
print("Obtained threshold: ", otsu_threshold)

'''
