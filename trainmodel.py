import preprocess
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import os
import pickle
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from models import *
import moredata

global  dropoutD


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    total_loss = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        end = offset + BATCH_SIZE
        batch_x, batch_y = X_data[offset:end], y_data[offset:end]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})#, keep_prob: np.ones(len(dropoutD))})
        loss = sess.run(loss_operation, feed_dict={x: batch_x, y: batch_y})#, keep_prob: np.ones(len(dropoutD))})
        total_accuracy += (accuracy * len(batch_x))
        total_loss += (loss * len(batch_x))
    return total_accuracy / num_examples, total_loss / num_examples

def training(X_data, y_data):
    num_examples = len(X_data)
    total_loss = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        end = offset + BATCH_SIZE
        batch_x, batch_y = X_data[offset:end], y_data[offset:end]
        sess.run(training_operation, feed_dict={x: batch_x, y: batch_y}) #, keep_prob: dropoutD})
        loss = sess.run(loss_operation, feed_dict={x: batch_x, y: batch_y})#, keep_prob: np.ones(len(dropoutD))})
        total_loss += (loss * len(batch_x))
    return total_loss / num_examples

def saveaug(wdir, filename, X, y):
    data={}
    data['features'] = X
    data['labels'] = y
    with open(wdir+'/data/'+filename+'.p','wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
# Data import
    wdir = os.getcwd()
    training_file = wdir+'/data/train_aug_2000_2.p' # To change back to original train data, use train.p
    testing_file = wdir+'/data/test.p'

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    X_train, y_train = train['features'], train['labels']
    X_test, y_test = test['features'], test['labels']
    print("Import data")
    #print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# Uncommenct below to generate data
# # data generation
#     X_aug, y_aug = moredata.datagen(X_train, y_train, 2000, [30, 5, 5], 2)
#     print("More data generated", str(len(y_aug))+' training data')

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
    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'
    with tf.Session(config = config) as s:

        EPOCHS = 30
        BATCH_SIZE = 128
        rate = 0.0005
        #dropoutD=[0.5,0.5] # dropout keep rate for the layers

        # placeholders
        x = tf.placeholder(tf.float32, (None, 32, 32, X_Train.shape[-1]))
        y = tf.placeholder(tf.int32, (None))
        keep_prob = tf.placeholder(tf.float32, (None))
        one_hot_y = tf.one_hot(y, 43)

        logits =  miniVGG(x) # LeNet(x) #
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
        loss_operation = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate = rate)
        training_operation = optimizer.minimize(loss_operation)

        # evaluation
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        saver = tf.train.Saver()
        print("Graph constructed.")

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            num_examples = len(X_Train)

            print("Training...")
            print()
            for i in range(EPOCHS):
                X_Train, y_Train = shuffle(X_Train, y_Train)
                train_loss = training(X_Train, y_Train)

                validation_accuracy, validation_loss = evaluate(X_Val, y_Val)
                print("EPOCH {} ...".format(i+1))
                print("Train Loss = {:.3f}".format(train_loss), "Validation Loss = {:.3f}".format(validation_loss), "Validation Accuracy = {:.3f}".format(validation_accuracy))


            saver.save(sess, 'model')
            print("Model saved")
