import cv2
import numpy as np
import math
import time
import os
import PIL.Image
import cpbd
def brenner(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0]-2):
        for y in range(0, shape[1]):
            out+=(int(img[x+2,y])-int(img[x,y]))**2
    return out

def Laplacian(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    return cv2.Laplacian(img,cv2.CV_64F).var()

def SMD(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    shape = np.shape(img)
    out = 0
    for x in range(1, shape[0]-1):
        for y in range(0, shape[1]):
            out+=math.fabs(int(img[x,y])-int(img[x,y-1]))
            out+=math.fabs(int(img[x,y]-int(img[x+1,y])))
    return out

def SMD2(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0]-1):
        for y in range(0, shape[1]-1):
            out+=math.fabs(int(img[x,y])-int(img[x+1,y]))*math.fabs(int(img[x,y]-int(img[x,y+1])))
    return out

def variance(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    out = 0
    u = np.mean(img)
    shape = np.shape(img)
    for x in range(0,shape[0]):
        for y in range(0,shape[1]):
            out+=(img[x,y]-u)**2
    return out

def energy(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0]-1):
        for y in range(0, shape[1]-1):
            out+=((int(img[x+1,y])-int(img[x,y]))**2)*((int(img[x,y+1]-int(img[x,y])))**2)
    return out

def Vollath(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    shape = np.shape(img)
    u = np.mean(img)
    out = -shape[0]*shape[1]*(u**2)
    for x in range(0, shape[0]-1):
        for y in range(0, shape[1]):
            out+=int(img[x,y])*int(img[x+1,y])
    return out

def entropy(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    out = 0
    count = np.shape(img)[0]*np.shape(img)[1]
    p = np.bincount(np.array(img).flatten())
    for i in range(0, len(p)):
        if p[i]!=0:
            out-=p[i]*math.log(p[i]/count)/count
    return out

def main(img1, img2):
    print('Brenner',brenner(img1),brenner(img2))
    print('Laplacian',Laplacian(img1),Laplacian(img2))
    print('SMD',SMD(img1), SMD(img2))
    print('SMD2',SMD2(img1), SMD2(img2))
    print('Variance',variance(img1),variance(img2))
    print('Energy',energy(img1),energy(img2))
    print('Vollath',Vollath(img1),Vollath(img2))
    print('Entropy',entropy(img1),entropy(img2))

def calculate_clarity(img):
    bre=brenner(img)
    lap=Laplacian(img)
    smd=SMD(img)
    smd2=SMD2(img)
    var=variance(img)
    en=energy(img)
    vol=Vollath(img)
    entro=entropy(img)
    cpbd_result=cpbd.compute(img)
    return bre,lap,smd,smd2,var,en,vol,entro,cpbd_result
def image_cropping(old_im):
    for i in range(old_im.shape[0]):
        old_im = np.require(old_im, requirements=['W'])
        for j in range(old_im.shape[1]):
            center = np.array([int(old_im.shape[0]) / 2, int(old_im.shape[1]) / 2])
            t = np.array([i, j])
            if (sum((t - center) ** 2)) ** (1 / 2) > int(old_im.shape[0] / 2):
                old_im[i, j] = 0.0
    return old_im
# def loading_generated_projections(path):
#     labels = []
#     for root, dirs, files in os.walk(path):
#         for index, file in enumerate(files):
#
#     return projections

if __name__ == '__main__':

    data_path='./data/raw/'
    result=''


    # #读入原始图像
    # img1 = cv2.imread('hac.png')
    # img2 = cv2.imread('our.png')
    # #灰度化处理
    # img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # main(img1,img2)

    for root, dirs, files in os.walk(data_path):
        for index, file in enumerate(files):
            img=cv2.imread(data_path+file,cv2.IMREAD_GRAYSCALE)
            img = PIL.Image.fromarray(img)
            img = np.asarray(img.resize((90, 90), PIL.Image.BICUBIC))
            img=cv2.equalizeHist(img)
            img=image_cropping(img)

            bre, lap, smd, smd2, var, en, vol, entro,cpbd_result=calculate_clarity(img)
            result=result+file+'\n'+'Brenner: '+str(bre)+'\n'+'Laplacian: '+str(lap)+'\n'+'SMD: '+str(smd)+'\n'+'SMD2: '+str(smd2)+'\n'+'Variance: '+str(var)+'\n'+'Energy: '+str(en)+'\n'+'Vollath: '+str(vol)+'\n'+'Entropy: '+str(entro)+'\n'+'Cpbd: '+str(cpbd_result)+'\n\n'

    time_of_run = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    run_root_dir = './result/' + time_of_run + '/'
    if not os.path.exists(run_root_dir):
        os.makedirs(run_root_dir)
    # if not os.path.exists(run_root_dir + 'best/'):
    #     os.makedirs(run_root_dir + 'best/')
    # np.savetxt(run_root_dir + 'best/' + 'best_labels.txt', clustering_labels)
    with open(run_root_dir + 'record.txt', 'w') as record:
        record.write(result)
    print(result)
