import numpy as np

def basic(X):
    Xc = np.zeros(X.shape)
    Xc[:,:,:,0] = X[:,:,:,0]-X[:,:,:,0].mean()
    Xc[:,:,:,1] = X[:,:,:,1]-X[:,:,:,1].mean()
    Xc[:,:,:,2] = X[:,:,:,2]-X[:,:,:,2].mean()
    return Xc/255
