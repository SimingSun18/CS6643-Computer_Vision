import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np
import cv2

smooth_filter = np.ones((3,3), np.float32)/9
print(smooth_filter)

img = cv2.imread('/Users/simingsun/Desktop/test.png', cv2.IMREAD_GRAYSCALE)
print(img.shape)

width = img.shape[0]
length = img.shape[1]
result_img = np.ones((width, length), np.uint8)
for i in range(1, width-1):
    for j in range(1, length-1):
        tempMat = np.ones((3,3), np.float32)
        tempMat = img[i-1:i+2:1, j-1:j+2:1]
        tempMat = tempMat*smooth_filter
        result_img[i, j] = sum(map(sum, tempMat))

cv2.imshow('After 3*3 smooth', result_img)
cv2.imwrite('/Users/simingsun/Desktop/smooth_3*3_img.png', result_img)
cv2.waitKey(0)

smooth_filter2 = np.ones((5,5), np.float32)/25
result_img2 = np.ones((width, length), np.uint8)
for i in range(2, width-2):
    for j in range(2, length-2):
        tempMat = np.ones((5,5), np.float32)
        tempMat = img[i-2:i+3:1, j-2:j+3:1]
        tempMat = tempMat*smooth_filter2
        result_img2[i, j] = sum(map(sum, tempMat))

cv2.imshow('After 5*5 smooth', result_img2)
cv2.imwrite('/Users/simingsun/Desktop/smooth_5*5_img.png', result_img2)
cv2.waitKey(0)

gaussian_horizontal = np.array([0.03, 0.07, 0.12, 0.18, 0.2, 0.18, 0.12, 0.07, 0.03])

filter_length = gaussian_horizontal.size
gaussian_img_temp = np.ones((width, length), np.uint8)
gaussian_img = np.ones((width, length), np.uint8)

for i in range(4, width-4):
    for j in range(4, length-4):
        tempMat = np.ones((1, filter_length), np.float32)
        tempMat = img[i:i+1:1, j-4:j+5:1]
        tempMat = np.squeeze(np.asarray(tempMat))*np.squeeze(np.asarray(gaussian_horizontal))
        gaussian_img_temp[i, j] = sum(tempMat)

cv2.imshow('After horizontal Gaussian',gaussian_img_temp)
cv2.imwrite('/Users/simingsun/Desktop/HorizontalGaussian_img.png', gaussian_img_temp)
cv2.waitKey(0)

for i in range(4, width-4):
    for j in range(4, length-4):
        tempMat = np.ones((filter_length, 1), np.float32)
        tempMat = gaussian_img_temp[i-4:i+5:1, j:j+1:1]
        tempMat = np.squeeze(np.asarray(tempMat)) * np.squeeze(np.asarray(gaussian_horizontal))
        gaussian_img[i, j] = sum(tempMat)

cv2.imshow('After Gaussian', gaussian_img)
cv2.imwrite('/Users/simingsun/Desktop/Gaussian_img.png', gaussian_img)
cv2.waitKey(0)


laplacian_filter = ([0, -1, 0],[-1, 4, -1],[0, -1, 0])
lapla_img = np.ones((width, length), np.uint8)
for i in range(1, width-1):
    for j in range(1, length-1):
        tempMat = np.ones((3,3), np.float32)
        tempMat = result_img2[i-1:i+2:1, j-1:j+2:1]
        tempMat = tempMat*laplacian_filter
        if (sum(map(sum, tempMat)) > 0):
            lapla_img[i, j] = 0
        else:
            lapla_img[i, j] = 255

cv2.imshow('Laplacian', lapla_img)
cv2.imwrite('/Users/simingsun/Desktop/Laplacian_img.png', lapla_img)
cv2.waitKey(0)