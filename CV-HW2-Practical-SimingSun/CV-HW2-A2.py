import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np
import cv2
from PIL import Image

img = cv2.imread('/Users/simingsun/Desktop/multiple-keys.png', cv2.IMREAD_GRAYSCALE)
print(img.shape)

ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

match_img = cv2.imread('/Users/simingsun/Desktop/match-key-binary.png', cv2.IMREAD_GRAYSCALE)

filter = np.ones((match_img.shape[0],match_img.shape[1]))

for i in range(0, match_img.shape[0]):
    for j in range(0, match_img.shape[1]):
        if (match_img[i, j] == 0):
            filter[i, j] = -1

img = cv2.imread('/Users/simingsun/Desktop/multiple-keys-binary.png', cv2.IMREAD_GRAYSCALE)
result = np.zeros([img.shape[0], img.shape[1]])
filter_width = filter.shape[0]
filter_length = filter.shape[1]
for i in range(filter.shape[0]/2, img.shape[0]-filter.shape[0]/2):
    for j in range(filter.shape[1]/2, img.shape[1] - filter.shape[1]/2):
        tempMat = np.ones([filter.shape[0], filter.shape[1]])
        tempMat = img[i-(filter_width/2):i+(filter_width/2):1, j-(filter_length/2):j+(filter_length/2):1]
        tempMat = tempMat * filter
        result[i, j] = sum(map(sum, tempMat))/(filter.shape[0] * filter.shape[1])
        print('Result[%d, %d] is :%d' % (i, j, result[i, j]))

n_result = cv2.normalize(result, result, 0, 255, cv2.NORM_MINMAX)
r_img = Image.fromarray(n_result)

cv2.imshow('After processing', n_result.astype(np.uint8))
cv2.imwrite('/Users/simingsun/Desktop/MatchResult.png', n_result.astype(np.uint8))
cv2.waitKey(0)

