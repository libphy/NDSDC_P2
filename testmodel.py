import tensorflow as tf
from trainmodel import *

# copy & paste below after training and run
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    test_accuracy, _ = evaluate(X_Test, y_Test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
