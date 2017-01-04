import numpy as np
import cv2

def basic(X):
    Xc = np.zeros(X.shape)
    Xc[:,:,:,0] = X[:,:,:,0]-X[:,:,:,0].mean()
    Xc[:,:,:,1] = X[:,:,:,1]-X[:,:,:,1].mean()
    Xc[:,:,:,2] = X[:,:,:,2]-X[:,:,:,2].mean()
    return Xc/255

def gray(X):
    Xg = np.array(list(map(lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), X)))
    Xg = Xg.reshape(Xg.shape+(1,))
    return (Xg-Xg.mean())/255

def eqhRGB(X): # equalize histogram color
    Xr = np.array(list(map(lambda x: cv2.equalizeHist(x), X[:,:,:,0])))
    Xg = np.array(list(map(lambda x: cv2.equalizeHist(x), X[:,:,:,1])))
    Xb = np.array(list(map(lambda x: cv2.equalizeHist(x), X[:,:,:,2])))
    Xe = np.array(list(map(lambda i: np.dstack((Xr[i,:,:], Xg[i,:,:], Xb[i,:,:])),range(X.shape[0]))))
    return (Xe-Xe.mean())/255


def eqhGray(X): # equalize histogram gray
    if X.shape[-1] ==3:
        Xg = np.array(list(map(lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), X)))
    elif X.shape[-1] ==1:
        Xg = X.reshape(X.shape[0],32,32)
    else:
        print("Error: wrong image dimension")
    Xe = np.array(list(map(lambda x: cv2.equalizeHist(x), Xg)))
    Xe = Xe.reshape(X.shape[0],32,32,1)
    return (Xe-Xe.mean())/255
