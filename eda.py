import preprocess
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd

wdir = os.getcwd()
training_file = wdir+'/data/train.p'
testing_file = wdir+'/data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']
print("Import data")


def preprocess_demo(X_tr, y_tr):
    X0 = preprocess.eqhGray(X_tr)
    xy = np.array(list(zip(X0, y_tr)))

    for picnum in range(43):
        pics = np.array(list(filter(lambda x: x[1] == picnum, xy)))
        fig = plt.figure()
        plt.title(str(picnum))
        for i in range(4):
            a = fig.add_subplot(2,2,i+1)
            if X0.shape[-1] == 3:
                plt.imshow(pics[i,0][:,:,:])
            if X0.shape[-1] == 1:
                plt.imshow(pics[i,0][:,:,0],cmap='gray')
        plt.show()

print(Counter(y_train))

# import cv2
# im = cv2.imread(wdir+'/etc/STOP_sign.jpg', 0)
# im = X_train[0,:,:,0]
# th,imth = cv2.threshold(im,0,255,cv2.THRESH_BINARY)
# #plt.imshow(imth)
# edges = cv2.Canny(im, th/2, th)
# plt.imshow(edges)
# plt.show()
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
