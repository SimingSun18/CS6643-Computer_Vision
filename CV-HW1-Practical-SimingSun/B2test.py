import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np
import cv2
from PIL import Image

def getBinaryImgWithThreshold(img, threshold):
    S = img.shape
    img_width = S[0]
    img_length = S[1]
    result_img = np.zeros([img_width, img_length])
    for i in range(0, img_width):
        for j in range(0, img_length):
            if (img[i, j] < threshold).all():
                result_img[i, j] = 0
            else:
                result_img[i, j] = 255
    result_img = Image.fromarray(result_img)
    result_img = result_img.convert('L')
    return result_img

img_a = cv2.imread("/Users/simingsun/Desktop/b2_a.png", cv2.IMREAD_GRAYSCALE)
img_b = cv2.imread("/Users/simingsun/Desktop/b2_b.png", cv2.IMREAD_GRAYSCALE)
img_c = cv2.imread("/Users/simingsun/Desktop/b2_c.png", cv2.IMREAD_GRAYSCALE)
img_a = cv2.cvtColor(img_a, cv2.COLOR_RGB2BGR)
img_b = cv2.cvtColor(img_b, cv2.COLOR_RGB2BGR)
img_c = cv2.cvtColor(img_c, cv2.COLOR_RGB2BGR)

img_a_hist = np.zeros(256)
img_b_hist = np.zeros(256)
img_c_hist = np.zeros(256)
for i in range(0, img_a.shape[0]):
    for j in range(0, img_a.shape[1]):
        img_a_hist[img_a[i, j]] += 1
for i in range(0, img_b.shape[0]):
    for j in range(0, img_b.shape[1]):
        img_b_hist[img_b[i, j]] += 1
        img_c_hist[img_c[i, j]] += 1
x = np.arange(0, 256, 1)

bimg_a = getBinaryImgWithThreshold(img_a, 127)
bimg_b = getBinaryImgWithThreshold(img_b, 127)
bimg_c = getBinaryImgWithThreshold(img_c, 127)

# f1 = plt.figure()
# plt.title('Original Images And Binary Images (threshold=127)')
# plt.subplot(331), plt.imshow(img_a), plt.title('b2_a')
# plt.subplot(332), plt.imshow(img_b), plt.title('b2_b')
# plt.subplot(333), plt.imshow(img_c), plt.title('b2_c')
#
# plt.subplot(334), plt.bar(x, img_a_hist), plt.title('b2_a histogram')
# plt.subplot(335), plt.bar(x, img_b_hist), plt.title('b2_b histogram')
# plt.subplot(336), plt.bar(x, img_c_hist), plt.title('b2_c histogram')
#
# plt.subplot(337), plt.imshow(bimg_a), plt.title('binary b2_a')
# plt.subplot(338), plt.imshow(bimg_b), plt.title('binary b2_b')
# plt.subplot(339), plt.imshow(bimg_c), plt.title('binary b2_c')
# plt.show()

#2 Ostu
#
# f2 = plt.figure()
# plt.title('Original Images And Binary Images (threshold=127)')
# fig, ax = plt.subplots(3, 3, sharey='row')
# ax[0, 0].imshow(img_a)
# ax[0, 1].imshow(img_b)
# ax[0, 2].imshow(img_c)
# ax[1, 0].hist(img_a.ravel(), 256, [0, 256])
# ax[1, 1].hist(img_b.ravel(), 256, [0, 256])
# ax[1, 2].hist(img_c.ravel(), 256, [0, 256])
# ax[2, 0].imshow(bimg_a)
# ax[2, 1].imshow(bimg_b)
# ax[2, 2].imshow(bimg_c)
# plt.show()

def getOstuThreshold(img):
    max_variance = 0
    best_threshold = 0
    variance_list = np.zeros(256)
    for temp_thredhold in range(0, 256, 1):
        fore_img = img > temp_thredhold
        fore_pix = np.sum(fore_img)

        back_img = img <= temp_thredhold
        back_pix = np.sum(back_img)

        fore_frac = float(fore_pix)/img.size
        fore_mean = float(np.sum(img*fore_img))/fore_pix

        back_frac = float(back_pix)/img.size
        back_mean = float(np.sum(img*back_img))/back_pix

        inner_class_invariance = fore_frac * back_frac * pow((fore_mean - back_mean),2)
        variance_list[temp_thredhold] = inner_class_invariance

        if inner_class_invariance > max_variance:
            max_variance = inner_class_invariance
            best_threshold = temp_thredhold
    return best_threshold, variance_list


thr, v_list = getOstuThreshold(img_c)
print(thr)
f3 = plt.figure(figsize=(15,7))
plt.subplot(221), plt.imshow(img_c), plt.title('Original')
plt.subplot(222), plt.imshow(bimg_c), plt.title('Manually Choose Threshold (threshold=127)')
ostu_img_a = getBinaryImgWithThreshold(img_c, int(thr))
plt.subplot(223)
plt.hist(img_c.ravel(), 256, [0, 256])
plt.plot(np.arange(0, 256, 1), v_list), plt.title('Inner Class Variance')
plt.vlines(thr, 0, v_list[thr], color='r')
plt.text(thr-20, v_list[thr]+10,'Max Invariance = %d'%thr)
plt.subplot(224), plt.imshow(ostu_img_a), plt.title('Ostu Thredhold (threshold=%d)'%thr)
plt.show()
