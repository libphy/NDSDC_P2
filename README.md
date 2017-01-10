# ND SDC project 2. Traffic signs classification
This project is part of Udacity SDC course.
The data is from the German traffic sign dataset [original](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). A processed, pickled dataset is available through the Udacity project repo, but I won't release here for the protection of their course copyrights.
The goal is to build various models for the traffic signs classification.

#### Key files
- README.md : explains thought process
- trainmodel.py : main function for training 
- moredata.py : generate more data using transformation
- preprocess.py : preprocessing
- models.py : various models
- testmodel.py : test
- other files : for checking things

## EDA
Number of training examples = 39209
Number of testing examples = 12630
Image data shape = (32, 32, 3) *e.g. X_train.shape = (39209,32,32,3)*
Number of classes = 43

least frequent: 210 counts, most frequent: 2250 counts. at least 10x imbalance.
If I generate data with 2000 per each label, I get 43*2000 = 86000, nearly 100k. original training size = 39209 (~121 MB)

Counter({0: 210,      1: 2220,       2: 2250,     3: 1410,
         4: 1980,     5: 1860,       6: 420,      7: 1440,
         8: 1410,     9: 1470,       10: 2010,    11: 1320,
         12: 2100,    13: 2160,      14: 780,     15: 630,
         16: 420,     17: 1110,      18: 1200,    19: 210,
         20: 360,     21: 330,       22: 390,     23: 510,
         24: 270,     25: 1500,      26: 600,     27: 240,
         28: 540,     29: 270,       30: 450,     31: 780,
         32: 240,     33: 689,       34: 420,     35: 1200,
         36: 390,     37: 210,       38: 2070,    39: 300,
         40: 360,     41: 240,       42: 240})

Visualizing label 0 (speed limit 20) shows that the data is quite similar to others of the same label. It seems some are even augmented data.   

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
- Changing to Xavier initializations in convnet weights in LeNet doesn't help, results in less accuracy. -> changed back to truncated normal
- Built [miniVGG], which consists of conv-conv-maxpool repeated twice or third times. Starting with the same number of filters as LeNet didn't improve accuracy much, until I increased filters to 16/16, 32/32, 64/64 all 3x3 conv with stride 1. MXPL has 2x2 filter with stride 2. I tried 32/32, 64/64, 128/128, and even adding 3 conv stacks (256/256/256) - like it appears in the later modules in VGGNet. However, too many layers (9 conv layers) were not so useful as it starts overfit and sort of sit in the same range of accuracy. This results in 98% validation accuracy and over 96% test accuracy. It may help to have dropout layers in conv layers (so far, I have one dropout in a fc layer)  

#### Exp 6
- [Analyzing the layer dimensions](https://docs.google.com/spreadsheets/d/1lcifpdc5MRsckGcHUz2mxFbEMopHZIkCHcs5R6w0l8U/edit?usp=sharing) showed that MXPL with stride 2 shrinks layer width and height rapidly, which might be why I did not get improvement when I added 3-layer stack (conv7-9) previously. So the maxpool strides for MXPL layer after conv 6 and conv9 have been changed to 1, such that the w x h becomes 6 x 6 after conv 9 maxpool. This gave 98-99% validation accuracy (very little, but still an improvement).  
- Applied dropout(0.5) with various combination (after conv4, 6, 9, fc 1, 2), (after conv 6, 9, fc 1, 2), (after conv 9, fc 1, 2), (after fc 1, 2), which yield 98+% validation accuracy. However, it seemed the drop out did not help with improving accuracy, although the difference in train & validation losses was lessened a little. Nothing spectacular though.
- Tried a different architecture: cascade structure [cascadeVGG]- the goal was to see if concatenating features from earlier layers and later layers help classification.  Using 1x1 convolutions branches are generated after conv2, conv4, and conv6 layers then concatenated at fc0 layer. This made a very long flattened layer (fc0), then it made weights for fc1 layer very large, training this model took long time and did not have a good performance. With or without dropouts in conv layers were not so different.
- [casecadeVGG2] To reduce the dimension of fc layer, the three branches were made to have the same dimension, then concatenated in depth, then convolve with 1d conv filter with size 1 to shrink the depth dimension. It is a little better than the original cascadeVGG, but still a lot slower than miniVGG. Also Epoch 30 at lr = 0.0005 or 0.001 are too slow. I haven't tried tweaking hyperparameters much. My general feeling is that this cascadeVGG model isn't very advantageous over miniVGG model. Probably because the layers are not that deep anyway, cascade (or skip-connection-like) effect may not be noticeable.
- I haven' had time to explore inception modules.
- Speed of training: 9-conv-layer miniVGG can be trained within a few minutes using a gtx 970 or a titan X gpu on 68000 training data.

## Conclusion
98+% validation accuracy with reasonable train-validation loss difference with miniVGG model.  
Final miniVGG architecture selection is
```
(c32-c32-m/2s)-
(c64-c64-m/2s)-
(c128-c128-m/1s)-
(c256-c256-c256-m/1s)-
fc0(flatten)-fc1(512, dr:0.5)-fc2(128, dr:0.5)-logit(43)
@ ep-30, lr-0.0005, bat-128, opt-Adam
```
