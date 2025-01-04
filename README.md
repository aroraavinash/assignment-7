# MNIST Classification with PyTorch

This project implements various CNN architectures for MNIST digit classification using PyTorch.

## Project Overview

The project contains multiple model implementations with different architectural choices and regularization techniques:
- Basic CNN (Model1)
- CNN with Batch Normalization (Model2)
- CNN with Batch Normalization and Dropout (Model2b)
- CNN with GAP and optimized architecture (Model2c)
- Advanced CNN with regularization techniques (Model3)

## Model Architecture

The models follow a general structure of:
- Input convolution block
- Multiple convolution blocks with increasing channels
- Transition blocks with 1x1 convolutions
- Global Average Pooling (GAP)
- Softmax output layer

Key features:
- Uses BatchNormalization for better training stability
- Implements Dropout for regularization
- Uses Global Average Pooling to reduce parameters
- ReLU activation functions

## Requirements
python
torch
torchvision
numpy

## Usage
Import the model
from model import Model3 # or any other model variant
Create model instance
model = Model3()
Train the model


## Model Variants

### Model1
- Basic CNN implementation
- No regularization

### Model2
- Added Batch Normalization
- Improved training stability

### Model2b
- Added Dropout (0.25)
- Better generalization

### Model2c
- Optimized architecture
- Global Average Pooling

### Model3
- Advanced implementation
- Dropout (0.1)
- Optimized for MNIST classification

## Training

The models are trained on the MNIST dataset with:
- Data augmentation (RandomRotation)
- Normalization (mean=0.1307, std=0.3081)
- Cross-entropy loss
- Various optimization techniques

## Results

 
# Model1 Parameters: 238,926
Accuracy: 99.14% in 15 epochs.

Target:
    1. Got the basic skeleton. 
    2. No fancy stuff
    3. Results:
        ◦ Parameters: 194k
        ◦ Best Train Accuracy: 99.92
        ◦ Best Test Accuracy: 99.15
    4. Analysis:
        ◦ Heavy Model , but working. 
        ◦ We see some over-fitting

# Model2 Parameters: 5784
Accuracy: 88.67% in 15 epochs.

Target:
       1. Results:
        ◦ Parameters: 5784
        ◦ Best Train Accuracy: 88.17
        ◦ Best Test Accuracy: 88.67%
       2. Analysis:
        ◦  Model has less than 8000 parameters , but low accuracy. 
        ◦  We see slight under-fitting


# Model 2 Of Parameters: 5936

Target:
1. Results:
    • Parameters: 5936
    • Best Train Accuracy: 93.82 %
    • Best Test Accuracy: 91.82 %
Analysis:
	Added Batch Normalization after every layer. 
	Number of parameters increased slightly but still way under 8000.
	Accuracy improved, but model is over fitting

# Model 2(b):
Target:
    1. Added Dropout layer as part of regularization in addition to Batch Normalization.
    2. Results:
        1. Parameters: 5936
        2. Best Train Accuracy: 89.19% (in 14th epoch)
        3. Best Test Accuracy: 91.84% (in 13th epoch)
    3. Analysis:
        1. Same number of parameters as previous model.
        2. Accuracy didn’t improved (Infact training accuracy dropped) but model became underfitting. 
        3. Overall accuracy not improvement by adding dropout. 


#Model2(c)
Target: 
    1. Added GAP layer at the end. 
    2. Results:
        1. Parameter: 5936
        2. Best Train Accuracy: 98.50%
        3. Best Test Accuracy: 97.29%
Analysis
Added Average pooling at the end of model. 
Gap between train and test accuracy reduced;
Model is slightly overfitting


#Model 3:
Target:
    1. Added LRPlateau Scheduler for improving accuracy.
    2. Results:
        1. #Of parameters: 5920 
        2. Best Train Accuracy: 99.35% (15th EPOCH)
        3. Best Train Accuracy: 97.63% (15th EPOCH)

Analysis:
    1.  Model is underfitting.
    2. Adding Scheduler on test accuracy to adjust improved the accuracy.
    3. No change in the optimizer. Tried with Adam Optimizer didn’t helped much. So kept the optimizer as SGD.
    4. Kept plateau at 3 and factor of 0.5. Which means learning rate is adjusted after 3 epochs. In current output logs, Accuracy at Epoch 4,   reached 99.11% but after which it kept falling to 99%, 99.04, 99.01%. in subsequent iterations. Hence Scheduler would have come into picture at EPOCH 8. Which helped improving accuracy going forward to 99.17%, 99.19%, 99.20%, 99.27% (EPOCH 12).  

Receptive Field Calculations (Model 3):



Block Name	Input Size	Kernel	Stride	Output	Receptive Field
Convblock1	28*28*1	3*3*1*8	1	26*26*8	3*3
Convblock2	26*26*8	3*3*8*12	1	24*24*12	5*5
Convblock3	24*24*12	1*1*12*8	1	24*24*8	5*5
Pool1	24*24*8	2*2	2	12*12*8	6*6
Convblock4	12*12*8	3*3*8*12	1	10*10*12	10*10
Convblock5	10*10*12	3*3*12*12	1	8*8*12	14*14
Convblock6	8*8*12	3*3*12*12	1	6*6*12	    18*18
Convblock7	6*6*12	3*3*12*12	1	4*4*12	     22*22
GAP	4*4*12	6*6		4*4*12	22*22
					
					
					
