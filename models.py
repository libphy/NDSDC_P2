import tensorflow as tf
from tensorflow.contrib.layers import flatten


def LeNet(x):
    # Hyperparameters
    mu = 0
    sigma = 0.1

    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, x._shape[-1].value, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b

    # Activation.
    fc1    = tf.nn.relu(fc1)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b

    # Activation.
    fc2    = tf.nn.relu(fc2)

    # Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits

def LeNetDrop(x, keep_prob): # To run this model, need to uncomment dropoutD in the trainmodel.py, then change feed_dict to include dropoutD
    # Hyperparameters
    mu = 0
    sigma = 0.1

    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, x._shape[-1].value, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    #conv1   = tf.nn.dropout(conv1, keep_prob[0])

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    #conv2   = tf.nn.dropout(conv2, keep_prob[1])
    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    fc1   = tf.nn.dropout(fc1, keep_prob[0])
    # Activation.
    fc1    = tf.nn.relu(fc1)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    fc2   = tf.nn.dropout(fc2, keep_prob[1])
    # Activation.
    fc2    = tf.nn.relu(fc2)

    # Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits

def miniVGG(x):
    # Hyperparameters
    mu = 0
    sigma = 0.1

    # Layer 1: Convolutional.
    nf1 = 32
    f1 = 3
    conv1_W = tf.Variable(tf.truncated_normal(shape=(f1, f1, x._shape[-1].value, nf1), mean = mu, stddev = sigma))
    #conv1_W = tf.get_variable("W_c1", shape=(5, 5, x._shape[-1].value, 6), initializer=tf.contrib.layers.xavier_initializer())
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
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 3: Convolutional.
    nf3 = 64
    f3 = 3
    conv3_W = tf.Variable(tf.truncated_normal(shape=(f3, f3, conv2._shape[-1].value, nf3), mean = mu, stddev = sigma))
    #conv1_W = tf.get_variable("W_c1", shape=(5, 5, x._shape[-1].value, 6), initializer=tf.contrib.layers.xavier_initializer())
    conv3_b = tf.Variable(tf.zeros(nf3))
    conv3   = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='SAME') + conv3_b
    conv3 = tf.nn.relu(conv3)

    # Layer 4: Convolutional.
    nf4 = 64
    f4 = 3
    conv4_W = tf.Variable(tf.truncated_normal(shape=(f4, f4, conv3._shape[-1].value, nf4), mean = mu, stddev = sigma))
    conv4_b = tf.Variable(tf.zeros(nf4))
    conv4   = tf.nn.conv2d(conv3, conv4_W, strides=[1, 1, 1, 1], padding='SAME') + conv4_b
    conv4 = tf.nn.relu(conv4)
    # dropout
    #conv4 = tf.nn.dropout(conv4, 0.5)
    # Pooling.
    conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 5: Convolutional.
    nf5 = 128
    f5 = 3
    conv5_W = tf.Variable(tf.truncated_normal(shape=(f5, f5, conv4._shape[-1].value, nf5), mean = mu, stddev = sigma))
    conv5_b = tf.Variable(tf.zeros(nf5))
    conv5   = tf.nn.conv2d(conv4, conv5_W, strides=[1, 1, 1, 1], padding='SAME') + conv5_b
    conv5 = tf.nn.relu(conv5)

    # Layer 6: Convolutional.
    nf6 = 128
    f6 = 3
    conv6_W = tf.Variable(tf.truncated_normal(shape=(f6, f6, conv5._shape[-1].value, nf6), mean = mu, stddev = sigma))
    conv6_b = tf.Variable(tf.zeros(nf5))
    conv6   = tf.nn.conv2d(conv5, conv6_W, strides=[1, 1, 1, 1], padding='SAME') + conv6_b
    conv6 = tf.nn.relu(conv6)
    #conv6  = tf.nn.dropout(conv6, 0.5)
    # Pooling.
    conv6 = tf.nn.max_pool(conv6, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')

    # Layer 7: Convolutional.
    nf7 = 256
    f7 = 3
    conv7_W = tf.Variable(tf.truncated_normal(shape=(f7, f7, conv6._shape[-1].value, nf7), mean = mu, stddev = sigma))
    conv7_b = tf.Variable(tf.zeros(nf7))
    conv7   = tf.nn.conv2d(conv6, conv7_W, strides=[1, 1, 1, 1], padding='SAME') + conv7_b
    conv7 = tf.nn.relu(conv7)

    # Layer 8: Convolutional.
    nf8 = 256
    f8 = 3
    conv8_W = tf.Variable(tf.truncated_normal(shape=(f8, f8, conv7._shape[-1].value, nf8), mean = mu, stddev = sigma))
    conv8_b = tf.Variable(tf.zeros(nf8))
    conv8   = tf.nn.conv2d(conv7, conv8_W, strides=[1, 1, 1, 1], padding='SAME') + conv8_b
    conv8 = tf.nn.relu(conv8)

    # Layer 9: Convolutional.
    nf9 = 256
    f9 = 3
    conv9_W = tf.Variable(tf.truncated_normal(shape=(f9, f9, conv8._shape[-1].value, nf9), mean = mu, stddev = sigma))
    conv9_b = tf.Variable(tf.zeros(nf9))
    conv9   = tf.nn.conv2d(conv8, conv9_W, strides=[1, 1, 1, 1], padding='SAME') + conv9_b
    conv9 = tf.nn.relu(conv9)
    #conv9 = tf.nn.dropout(conv9, 0.5)
    # Pooling.
    conv9 = tf.nn.max_pool(conv9, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')

    # Flatten.
    fc0   = flatten(conv9)

    # Layer 3: Fully Connected.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(fc0._shape[-1].value, 512), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(512))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    fc1   = tf.nn.dropout(fc1, 0.5)
    # Activation.
    fc1    = tf.nn.relu(fc1)

    #Layer 4: Fully Connected. 
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(512, 128), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(128))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    fc2   = tf.nn.dropout(fc2, 0.5)
    # Activation.
    fc2    = tf.nn.relu(fc2)

    # Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(128, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits

def cascadeVGG(x):
    # Hyperparameters
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
    #conv2m = tf.nn.dropout(conv2m, 0.5)

    # Branching with 1x1 conv
    conv2r_W = tf.Variable(tf.truncated_normal(shape=(1, 1, conv2m._shape[-1].value, conv2m._shape[-1].value//4), mean = mu, stddev = sigma))
    conv2r_b = tf.Variable(tf.zeros(conv2m._shape[-1].value//4))
    conv2r   = tf.nn.conv2d(conv2m, conv2r_W, strides=[1, 1, 1, 1], padding='SAME') + conv2r_b

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
    #conv4m = tf.nn.dropout(conv4m, 0.5)

    # Branching with 1x1 conv
    conv4r_W = tf.Variable(tf.truncated_normal(shape=(1, 1, conv4m._shape[-1].value,  conv4m._shape[-1].value//2), mean = mu, stddev = sigma))
    conv4r_b = tf.Variable(tf.zeros(conv4m._shape[-1].value//2))
    conv4r   = tf.nn.conv2d(conv4m, conv4r_W, strides=[1, 1, 1, 1], padding='SAME') + conv4r_b

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
    #conv6m = tf.nn.dropout(conv6m,0.5)

    # Flatten.
    print(flatten(conv2r)._shape,flatten(conv4r)._shape,flatten(conv6m)._shape )
    fc0   = tf.concat(1,[flatten(conv2r),flatten(conv4r),flatten(conv6m)])

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

    return logits

def cascadeVGG2(x):
    # Hyperparameters
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

    # Branching with 1x1 conv
    conv2r_W = tf.Variable(tf.truncated_normal(shape=(1, 1, conv2m._shape[-1].value, conv2m._shape[-1].value//4), mean = mu, stddev = sigma))
    conv2r_b = tf.Variable(tf.zeros(conv2m._shape[-1].value//4))
    conv2r   = tf.nn.conv2d(conv2m, conv2r_W, strides=[1, 1, 1, 1], padding='SAME') + conv2r_b

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

    # Branching with 1x1 conv
    conv4r_W = tf.Variable(tf.truncated_normal(shape=(1, 1, conv4m._shape[-1].value,  conv4m._shape[-1].value//2), mean = mu, stddev = sigma))
    conv4r_b = tf.Variable(tf.zeros(conv4m._shape[-1].value//2))
    conv4r   = tf.nn.conv2d(conv4m, conv4r_W, strides=[1, 1, 1, 1], padding='SAME') + conv4r_b

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
    flat2 = flatten(conv2r)
    flat4 = flatten(conv4r)
    flat6 = flatten(conv6m)
    fc0   = tf.concat(2,[tf.reshape(flat2,[-1,flat2._shape[-1].value,1]),tf.reshape(flat4,[-1,flat2._shape[-1].value,1]),tf.reshape(flat2,[-1,flat2._shape[-1].value,1])])

    conv7_W = tf.Variable(tf.truncated_normal(shape=(1, 3, 1), mean = mu, stddev = sigma))
    conv7_b = tf.Variable(tf.zeros(1))
    conv7   = tf.nn.conv1d(fc0, conv7_W, stride=1, padding='SAME') + conv7_b
    conv7 = tf.reshape(conv7,[-1, conv7._shape[1].value])
    # Layer 3: Fully Connected.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(conv7._shape[-1].value, 512), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(512))
    fc1   = tf.matmul(conv7, fc1_W) + fc1_b
    fc1   = tf.nn.dropout(fc1, 0.5)
    # Activation.
    fc1    = tf.nn.relu(fc1)

    #Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(512, 128), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(128))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    fc2   = tf.nn.dropout(fc2, 0.5)
    # Activation.
    fc2    = tf.nn.relu(fc2)

    # Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(128, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits
#def miniInception(x, keep_prob):
