# ND SDC project 2. Traffic signs classification
This project is part of Udacity SDC course.
The data is from the German traffic sign dataset [original](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). A processed, pickled dataset is available through the Udacity project repo, but I won't release here for the protection of their course copyrights.
The goal is to build various models for the traffic signs classification.

## EDA
Number of training examples = 39209
Number of testing examples = 12630
Image data shape = (32, 32, 3) *e.g. X_train.shape = (39209,32,32,3)*
Number of classes = 43

## Experiment results
#### Day 1
- simple pre-process (X-X_mean_channel)/255 for each channel
- set aside validation data from training data (train:val = 0.8:0.2)
- I used the plain LeNet (introduced in the class exercise), but got a result that it overfits. (96% validation accuracy, then 88% test accuracy) It seems it needs regularization/dropouts or a change in structure.

#### Day 2
- Implemented dropout layers after conv2d layers. It rarely improved the test accuracy although it improved the signs of overfitting (I printed train & validation loss for each epoch).
- Color and grayscales were skewed. Implemented equalizeHist to each channel. It improved test accuracy by 4% on plain LeNet, while it doesn't overfit as much.
- Maybe I'll try putting dropouts in the FC layer(s) instead.
