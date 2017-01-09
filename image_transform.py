# reference: https://nbviewer.jupyter.org/github/vxy10/SCND_notebooks/blob/master/preprocessing_stuff/img_transform_NB.ipynb
# The same transformation can be done via keras.preprocessing.image.ImageDataGenerator

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
#from keras.preprocessing.image import ImageDataGenerator as imgen
import os
import numpy as np

def transform_image(img,ang_range,shear_range,trans_range):
    '''
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation in degrees
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over.

    A Random uniform distribution is used to generate different parameters for transformation

    '''
    # Rotation

    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows,cols,ch = img.shape
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])

    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2

    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

    shear_M = cv2.getAffineTransform(pts1,pts2)

    img = cv2.warpAffine(img,Rot_M,(cols,rows))
    img = cv2.warpAffine(img,Trans_M,(cols,rows))
    img = cv2.warpAffine(img,shear_M,(cols,rows))

    return img

def display2(im, ang,shear,trans):
    im1 = cv2.resize(stop,dsize=(32,32))
    im2 = transform_image(im1,ang,shear,trans)
    fig = plt.figure()
    a = fig.add_subplot(2,1,1)
    plt.imshow(cv2.cvtColor(im1, cv2.COLOR_BGR2RGB))
    a = fig.add_subplot(2,1,2)
    plt.imshow(cv2.cvtColor(im2, cv2.COLOR_BGR2RGB))
    plt.show()

def display(im, ang,shear,trans, idx):
    im2 = transform_image(im1,ang,shear,trans)
    a = fig.add_subplot(idx[0],idx[1],idx[2])
    plt.imshow(cv2.cvtColor(im2, cv2.COLOR_BGR2RGB))
    return a

if __name__ == '__main__':
    wdir = os.getcwd()
    stop = cv2.imread(wdir+'/etc/Stop.jpeg')
    img = cv2.resize(stop,dsize=(32,32))
    fig = plt.figure()
    for i in range(100):
        im = transform_image(img,60,5,10)
        a = fig.add_subplot(10,10,i+1)
        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.show()    
