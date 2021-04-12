import cpbd
from scipy import ndimage
import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
def Centralization(image):
    mean = np.mean(image)
    # var = np.mean(np.square(image-mean))

    # image = (image - mean)/np.sqrt(var)
    image=image - mean
    return image

def image_cropping(old_im):
    for i in range(old_im.shape[0]):
        old_im = np.require(old_im, requirements=['W'])
        for j in range(old_im.shape[1]):
            center = np.array([int(old_im.shape[0]) / 2, int(old_im.shape[1]) / 2])
            t = np.array([i, j])
            if (sum((t - center) ** 2)) ** (1 / 2) > int(old_im.shape[0] / 2):
                old_im[i, j] = 0.0
    return old_im

def image_cropping(old_im):
    for i in range(old_im.shape[0]):
        old_im = np.require(old_im, requirements=['W'])
        for j in range(old_im.shape[1]):
            center = np.array([int(old_im.shape[0]) / 2, int(old_im.shape[1]) / 2])
            t = np.array([i, j])
            if (sum((t - center) ** 2)) ** (1 / 2) > int(old_im.shape[0] / 2)-9:
                old_im[i, j] = 0.0
    return old_im

# img=ndimage.imread('relion.png')
# img = PIL.Image.fromarray(img)
# img = np.asarray(img.resize((90, 90), PIL.Image.BICUBIC))
# img=image_cropping(img)
# img=Centralization(img)
#
# img2=ndimage.imread('hac.png')
# img2= PIL.Image.fromarray(img2)
# img2 = np.asarray(img2.resize((90, 90), PIL.Image.BICUBIC))
# img2=image_cropping(img2)
# img2=Centralization(img2)
#
# img3=ndimage.imread('kmeans.png')
# img3= PIL.Image.fromarray(img3)
# img3 = np.asarray(img3.resize((90, 90), PIL.Image.BICUBIC))
# img3=image_cropping(img3)
# img3=Centralization(img3)
#
# img4=ndimage.imread('our.png')
# img4 = PIL.Image.fromarray(img4)
# img4 = np.asarray(img4.resize((90, 90), PIL.Image.BICUBIC))
# img4=image_cropping(img4)
# img4=Centralization(img4)
#
# # img5=ndimage.imread('our_80_0007.png')
# # img5 = PIL.Image.fromarray(img5)
# # img5 = np.asarray(img5.resize((90, 90), PIL.Image.BICUBIC))
# #
# # img6=ndimage.imread('relion_100_0055.png')
# # img6 = PIL.Image.fromarray(img6)
# # img6 = np.asarray(img6.resize((90, 90), PIL.Image.BICUBIC))
#
# a=cpbd.compute(img)
# b=cpbd.compute(img2)
# c=cpbd.compute(img3)
# d=cpbd.compute(img4)
# # e=cpbd.compute(img5)
# # f=cpbd.compute(img6)
# print(a)
# print(b)
# print(c)
# print(d)
#
# plt.subplot(1,4,1)
# plt.imshow(img,cmap='gray')
# plt.subplot(1,4,2)
# plt.imshow(img2,cmap='gray')
# plt.subplot(1,4,3)
# plt.imshow(img3,cmap='gray')
# plt.subplot(1,4,4)
# plt.imshow(img4,cmap='gray')
# plt.show()

data_path='./data/raw/'
def read_data(data_path):
    for root, dirs, files in os.walk(data_path):
        for index, file in enumerate(files):
            img=cv2.imread(data_path+file,cv2.IMREAD_GRAYSCALE)
            img = PIL.Image.fromarray(img)
            img = np.asarray(img.resize((90, 90), PIL.Image.BICUBIC))
            img=np.expand_dims(img,axis=0)
            if index == 0:
                imgs = img
            else:
                imgs = np.append(imgs, img, axis=0)
    return imgs

def preprocess(imgs):
    len=imgs.shape[0]
    for i in range(len):

        # imgs[i] = image_cropping(imgs[i])
        imgs[i]=cv2.equalizeHist(imgs[i])
        imgs[i] = image_cropping(imgs[i])
    return imgs
if __name__ == '__main__':
    imgs=read_data(data_path)
    processed_imgs=preprocess(imgs)
    num_imgs=processed_imgs.shape[0]
    for j in range(num_imgs):
        plt.subplot(1,num_imgs,j+1)
        plt.imshow(processed_imgs[j],cmap='gray')
    # plt.subplot(1,4,2)
    # plt.imshow(img2,cmap='gray')
    # plt.subplot(1,4,3)
    # plt.imshow(img3,cmap='gray')
    # plt.subplot(1,4,4)
    # plt.imshow(img4,cmap='gray')
    plt.show()
# print(e)
# print(f)
# print((a+b+c+d+e+f)/6)
# print((a+b))