import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np
import cv2
from PIL import Image


# The function 'getEqualizedImg' is implemented to equalize the gray image
def getEqualizedImg(Img):
    img_size = Img.shape
    img_width = img_size[0]
    img_height = img_size[1]
    PixNum = np.zeros(256)
    for i in range(0, img_width):
        for j in range(0, img_height):
            PixNum[Img[i, j]] += 1
    pdf = np.zeros(256, dtype='float64')
    for i in range(0, 256):
        pdf[i] = float(PixNum[i] / (img_height*img_width))
    cdf = np.zeros(256, dtype='float64')
    cdf[0] = pdf[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + pdf[i]
    CDF = cdf * 255 + 0.5
    equImg_mat = np.zeros([img_width, img_height], dtype='int8')
    for i in range(0, img_width):
        for j in range(0, img_height):
            equImg_mat[i, j] = CDF[Img[i, j]]
    equImg = Image.fromarray(equImg_mat)
    equImg = equImg.convert('L')
    return equImg, equImg_mat


testImg = cv2.imread("/Users/simingsun/Desktop/b2_a_gray.jpg", cv2.IMREAD_GRAYSCALE)
# The color patterns of opencv and plt are different so the test image need to be converted
# unless their color will be changed when printed.
plt_testImg = cv2.cvtColor(testImg, cv2.COLOR_BGR2RGB)
equalized_Img, testMat = getEqualizedImg(testImg)

# Calculate the histogram of images manually
originalHist = np.zeros(256)
equalizedHist = np.zeros(256)
x = np.arange(0, 256, 1)
for i in range(0, testImg.shape[0]):
    for j in range(0, testImg.shape[1]):
        originalHist[testImg[i, j]] += 1
        equalizedHist[testMat[i, j]] += 1

# The commented out code below is the output of the images
# plt.figure()
# plt.title('def test')
# plt.subplot(221), plt.title('Original Gray Image'), plt.imshow(plt_testImg)
# plt.subplot(222), plt.title('Equalized Gray Image'), plt.imshow(equalized_Img)
# plt.subplot(223), plt.title('Original Histogram'), plt.bar(x, originalHist)
# plt.subplot(224), plt.title('Equalized Histogram'), plt.bar(x, equalizedHist)
# plt.show()