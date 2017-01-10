import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import os
import numpy as np
import pandas as pd

def transform_image(img,ang_range,shear_range,trans_range):
    #This function was from internet, but I modified it to fill the blank (with background extrapolation) when transformed.
    '''
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation in degrees
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over.

    A Random uniform distribution is used to generate different parameters for transformation

    '''
    padding =8
    img = cv2.copyMakeBorder(img,padding,padding,padding,padding, borderType=cv2.BORDER_REPLICATE)
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
    imgcut = img[padding:img.shape[0]-padding, padding:img.shape[1]-padding]
    return imgcut

# def dataaug(X, n_out): #for one kind of label
#     while X.shape[0] < n_out:
#         n_seed = np.random.choice(X.shape[0],1)
#         im_seed = X[n_seed,:,:,:].reshape(32,32,3)
#         im_aug = transform_image(im_seed, 45,5,10)
#         X = np.append(X,im_aug.reshape(1,32,32,3), axis = 0)
#     return X

def aug(X, n_repeat):
    X_out=[]
    for i in range(n_repeat):
        for x in X:
            im = transform_image(x, 45, 5, 10)
            X_out.append(im)
    return np.array(X_out)

def showsample(X, n, m):
    if X.shape[0] == n*m:
        fig = plt.figure()
        for i in range(X.shape[0]):
            a = fig.add_subplot(n,m,i+1)
            plt.imshow(X[i])
            #plt.imshow(cv2.cvtColor(X[i], cv2.COLOR_BGR2RGB))
        plt.show()

def sample(X, y, n_sample):
    Xy = pd.DataFrame(np.array(list(zip(X, y))), columns=['X','y'])
    Xnew=[]
    ynew=[]
    for i in range(43):
        Xs = np.array(Xy[Xy['y']==i]['X'])
        js = np.random.choice(Xs.shape[0],n_sample)
        for j in js:
            Xnew.append(Xs[j])
            ynew.append(i)
    return np.array(Xnew), np.array(ynew)

def datagen(X, y, n_sample, rsd, mode=0):
    """
    X: image data such as X_train shape: (n,w,h,c)
    y: label such as y_train
    n_sample: number of samples per label
    rsd = [angle, shear, diaplacement]
    """
    Xy = np.array(list(zip(X, y)))

    if mode < 2:
        Xnew=[]
        ynew=[]
        for i in range(43):
            Xyi = list(filter(lambda x: x[1] == i, Xy))
            Xs, _ = zip(*Xyi)
            Xs = np.array(Xs)
            js = np.random.choice(Xs.shape[0],n_sample)
            for j in js:
                Xnew.append(transform_image(Xs[j],rsd[0],rsd[1],rsd[2]))
                ynew.append(i)
        if mode == 0: # return only generated data
            return np.array(Xnew), np.array(ynew)
        elif mode == 1: # return concatenated data (original+generated)
            return np.concatenate((X,np.array(Xnew))), np.concatenate((y,np.array(ynew)))
    elif mode == 2: # fill the rest with generated data
        Xtot=[]
        ytot=[]
        for i in range(43):
            Xyi = list(filter(lambda x: x[1] == i, Xy))
            Xs, ys = zip(*Xyi)
            Xs = list(Xs)
            ys = list(ys)
            n = len(Xs)
            if n < n_sample:
                js = np.random.choice(n,n_sample-n)
                Xnew=[]
                ynew=[]
                for j in js:
                    Xnew.append(transform_image(Xs[j],rsd[0],rsd[1],rsd[2]))
                    ynew.append(i)
            else:
                Xnew=[]
                ynew=[]
                js = np.random.choice(n,n_sample)
                Xss=[]
                for j in js:
                    Xss.append(Xs[j])

                Xs = Xss
                ys = list(i*np.ones(len(Xs)))

            Xtot += Xs + Xnew
            ytot += ys + ynew
            #Xtot = np.concatenate((Xtot,np.concatenate((Xs,np.array(Xnew)))))
            #ytot = np.concatenate((ytot,np.concatenate((ys,np.array(ynew)))))
        return np.array(Xtot), np.array(ytot)
