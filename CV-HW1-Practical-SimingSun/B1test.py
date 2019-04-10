import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np
import cv2
from PIL import Image

# B1

# Question1
img = cv2.imread("/Users/simingsun/Desktop/b1.png")
B, G, R = cv2.split(img)
t = img.shape
# Calculate the histogram of images manually
blueHist = np.arange(256)
greenHist = np.arange(256)
redHist = np.arange(256)
for i in range(0, t[0]):
    for j in range(0, t[1]):
        blueHist[B[i, j]] += 1
        greenHist[G[i, j]] += 1
        redHist[R[i, j]] += 1

# The commented out code below is the output of the images
# plt.figure(1, figsize=(15, 8))
# plt.title("b1_a.png RGB histogram")
# x = range(0, 256, 1)
# plt.subplot(221), plt.title('RGB histograms')
# plt.plot(x, blueHist, color='b')
# plt.plot(x, greenHist, color='g')
# plt.plot(x, redHist, color='r')
# plt.xlim(0, 256)
# plt.ylim(0, 850)
# plt.subplot(222), plt.title('B histogram'), plt.xlim(0, 256), plt.bar(x, blueHist, color='b')
# plt.subplot(223), plt.title('G histogram'), plt.xlim(0, 256), plt.bar(x, greenHist, color='g')
# plt.subplot(224), plt.title('R histogram'), plt.xlim(0, 256), plt.bar(x, redHist, color='r')
# plt.show()

# Question 3
graymat = np.zeros([119, 256])
for i in range(0, t[0]):
    for j in range(0, t[1]):
        graymat[i, j] = R[i,j]*0.3 + G[i,j]*0.59 + B[i,j]*0.11
grayimg = Image.fromarray(graymat)
grayimg = grayimg.convert("L")
# cv2.imwrite("/Users/simingsun/Desktop/b2_a_gray.jpg", graymat)

# The commented out code below is the output of the images
# plt.figure(2)
# plt.imshow(grayimg)
# plt.title('Gray Image')
# plt.show()

# Question 4
grayhist = np.zeros(256)
graymat_int = graymat.astype(int)
for i in range(0, t[0]):
    for j in range(0, t[1]):
        grayhist[graymat_int[i, j]] += 1
x = np.arange(0, 256, 1)

# The commented out code below is the output of the images
# plt.figure()
# plt.xlim(0, 256)
# plt.title('Gray Image Histogram')
# plt.bar(x, grayhist)
# plt.show()

# Question 6
Sum = t[0]*t[1]
pdf = np.arange(len(grayhist), dtype='float64')
for i in range(0, 256):
    pdf[i] = float(grayhist[i]/Sum)

# The commented out code below is the output of the images
# plt.figure()
# plt.title('pdf')
# plt.xlim(0, 256)
# plt.plot(x, pdf)
# plt.show()

# Question 7
cdf = np.zeros(len(pdf), dtype='float64')
cdf[0] = pdf[0]
for i in range(1, 256):
    cdf[i] = cdf[i-1] + pdf[i]

# The commented out code below is the output of the images
# plt.figure()
# plt.title('cdf')
# plt.xlim(0, 256)
# plt.plot(x, cdf, color='r')
# plt.show()

# Question 8
CDF = cdf*255 + 0.5
equImg_mat = np.zeros([119, 256])
equImg_mat.astype('int')
for i in range(0, t[0]):
    for j in range(0, t[1]):
        equImg_mat[i, j] = CDF[graymat_int[i, j]]
equImg = Image.fromarray(equImg_mat)
equImg = equImg.convert('L')

# The commented out code below is the output of the images
# plt.subplot(121), plt.imshow(grayimg)
# plt.subplot(122), plt.imshow(equImg)
# plt.show()
