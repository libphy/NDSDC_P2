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
#### Exp 1
- simple pre-process (X-X_mean_channel)/255 for each channel
- set aside validation data from training data (train:val = 0.8:0.2)
- I used the plain LeNet (introduced in the class exercise), but got a result that it overfits. (96% validation accuracy, then 88% test accuracy) It seems it needs regularization/dropouts or a change in structure.

#### Exp 2
- Implemented dropout layers after conv2d layers. It rarely improved the test accuracy although it improved the signs of overfitting (I printed train & validation loss for each epoch).
- Color and grayscales were skewed. Implemented equalizeHist to each channel. It improved test accuracy by 4% on plain LeNet, while it doesn't overfit as much.
- Maybe I'll try putting dropouts in the FC layer(s) instead.

#### Exp 3
- Dropout layers on fc1 and fc2 instead conv layers. It surely less overfit, but it alone doesn't improve test accuracy. Long epoch at slow lr had very little improvement.

#### Exp 4
- Generated more data using random rotation, shear and translation. However, it does not help improving accuracy. Perhaps something funny happens during equalizing histogram since the generated data have blank (black) area.
- Fixed blank area by extrapolating the boundary. It was improved a little, but still it was worse than performance with bare data
- Added more options on generating data such that I can choose to have only new data, original +new data (not balanced, just new data added to original), original + new data (balanced, keep the original as much as possible, then fill the rest with the new data). Intuitively, the last option (mode=2) seems better.
- When transforming with ranges of (rot_ang = 45, shear =5, displacement=10), it was still producing a lot lower accuracy (<83%). Assuming that the rotation and displacement were too large, reducing to (30,5,5) gave a better result (90%), as well as similar to the test accuracy (improved). Model was the default LeNet model, datagen with n_sample=2000, mode=2.  

#### Exp 5
- changing to Xavier initializations in convnet weights in LeNet doesn't help, results in less accuracy. -> changed back to truncated normal
- built miniVGG, which consists of conv-conv-maxpool repeated twice or third times. Starting with the same number of filters as LeNet didn't improve accuracy much, until I increased filters to 16/16, 32/32, 64/64 all 3x3 conv with stride 1. MXPL has 2x2 filter with stride 2. I tried 32/32, 64,64, 128/128, and even adding 3 conv stacks - like it appears in the later modules in VGGNet. However, too many layers (9 conv layers) were not so useful as it starts overfit and sort of sit in the same range of accuracy. This results in 98% validation accuracy and over 96% test accuracy. It may help to have dropout layers in conv layers (so far, I have one dropout in a fc layer)  
