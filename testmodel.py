import tensorflow as tf
from trainmodel import *

saver = tf.train.Saver()
if __name__ == '__main__':
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))
        test_accuracy, _ = evaluate(X_Test, y_Test)
        print("Test Accuracy = {:.3f}".format(test_accuracy))
