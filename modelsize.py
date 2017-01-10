import tensorflow as tf
import preprocess
import numpy as np
from tensorflow.contrib.layers import flatten
import os
import pickle
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from models import *
import moredata

global  dropoutD
if __name__ == '__main__':
# Data import
    wdir = os.getcwd()
    training_file = wdir+'/data/train_aug_2000_2.p'
    testing_file = wdir+'/data/test.p'

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    X_train, y_train = train['features'], train['labels']
    X_test, y_test = test['features'], test['labels']
    print("Import data")

# preprocess
    X0 = preprocess.eqhGray(X_train)
    X1 = preprocess.eqhGray(X_test)
    print("Data preprocess done.")

# train, validation, test sets
    X_Train, X_Val, y_Train, y_Val = train_test_split(X0, y_train, test_size=0.2, random_state=123)
    X_Test, y_Test = shuffle(X1, y_test, random_state=123)
    print("Train, Validation, and Test set ready.")
    print(len(y_Train), len(y_Val), len(y_Test))

# Train pipeline
    # config = tf.ConfigProto()
    # config.gpu_options.allocator_type = 'BFC'
    # with tf.Session(config = config) as s:
    with tf.device('/gpu:0'):
        EPOCHS = 30
        BATCH_SIZE = 128
        rate = 0.0005
        dropoutD=[0.5,0.5] # dropout keep rate for the layers

        # placeholders
        x = tf.placeholder(tf.float32, (None, 32, 32, X_Train.shape[-1]))
        y = tf.placeholder(tf.int32, (None))
        keep_prob = tf.placeholder(tf.float32, (None))
        one_hot_y = tf.one_hot(y, 43)

# Paste a model
    mu = 0
    sigma = 0.1

    # Layer 1: Convolutional.
    nf1 = 32
    f1 = 3
    conv1_W = tf.Variable(tf.truncated_normal(shape=(f1, f1, x._shape[-1].value, nf1), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(nf1))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='SAME') + conv1_b
    conv1 = tf.nn.relu(conv1)

    # Layer 2: Convolutional.
    nf2 = 32
    f2 = 3
    conv2_W = tf.Variable(tf.truncated_normal(shape=(f2, f2, conv1._shape[-1].value, nf2), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(nf2))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='SAME') + conv2_b
    conv2 = tf.nn.relu(conv2)

    # Pooling.
    conv2m = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    # Dropout
    conv2m = tf.nn.dropout(conv2m, 0.5)

    # Layer 3: Convolutional.
    nf3 = 64
    f3 = 3
    conv3_W = tf.Variable(tf.truncated_normal(shape=(f3, f3, conv2._shape[-1].value, nf3), mean = mu, stddev = sigma))
    #conv1_W = tf.get_variable("W_c1", shape=(5, 5, x._shape[-1].value, 6), initializer=tf.contrib.layers.xavier_initializer())
    conv3_b = tf.Variable(tf.zeros(nf3))
    conv3   = tf.nn.conv2d(conv2m, conv3_W, strides=[1, 1, 1, 1], padding='SAME') + conv3_b
    conv3 = tf.nn.relu(conv3)

    # Layer 4: Convolutional.
    nf4 = 64
    f4 = 3
    conv4_W = tf.Variable(tf.truncated_normal(shape=(f4, f4, conv3._shape[-1].value, nf4), mean = mu, stddev = sigma))
    conv4_b = tf.Variable(tf.zeros(nf4))
    conv4   = tf.nn.conv2d(conv3, conv4_W, strides=[1, 1, 1, 1], padding='SAME') + conv4_b
    conv4 = tf.nn.relu(conv4)

    # Pooling.
    conv4m = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    # Dropout
    conv4m = tf.nn.dropout(conv4m, 0.5)

    # Layer 5: Convolutional.
    nf5 = 128
    f5 = 3
    conv5_W = tf.Variable(tf.truncated_normal(shape=(f5, f5, conv4._shape[-1].value, nf5), mean = mu, stddev = sigma))
    conv5_b = tf.Variable(tf.zeros(nf5))
    conv5   = tf.nn.conv2d(conv4m, conv5_W, strides=[1, 1, 1, 1], padding='SAME') + conv5_b
    conv5 = tf.nn.relu(conv5)

    # Layer 6: Convolutional.
    nf6 = 128
    f6 = 3
    conv6_W = tf.Variable(tf.truncated_normal(shape=(f6, f6, conv5._shape[-1].value, nf6), mean = mu, stddev = sigma))
    conv6_b = tf.Variable(tf.zeros(nf5))
    conv6   = tf.nn.conv2d(conv5, conv6_W, strides=[1, 1, 1, 1], padding='SAME') + conv6_b
    conv6 = tf.nn.relu(conv6)
    # Pooling.
    conv6m = tf.nn.max_pool(conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    # Dropout
    conv6m = tf.nn.dropout(conv6m,0.5)

    # Flatten.
    print(flatten(conv2m)._shape,flatten(conv4m)._shape,flatten(conv6m)._shape )
    fc0   = tf.concat(1,[flatten(conv6m),flatten(conv4m),flatten(conv2m)])

    # Layer 3: Fully Connected.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(fc0._shape[-1].value, 1024), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(1024))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    fc1   = tf.nn.dropout(fc1, 0.5)
    # Activation.
    fc1    = tf.nn.relu(fc1)

    #Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(1024, 128), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(128))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    fc2   = tf.nn.dropout(fc2, 0.5)
    # Activation.
    fc2    = tf.nn.relu(fc2)

    # Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(128, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
