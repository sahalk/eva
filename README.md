# EVA Assignment 

## Session 4

The main objective of this assignment is to design a network to classify the images (digits) from the MNIST dataset with the following requirements: 

* 99.4% validation accuracy
* Less than 20k Parameters
* Less than 20 Epochs
* No fully connected layer

The following changes were made to the Class: 

1. The total number of input and output channels in each convolution (nn.Conv2d) were reduced as compared to the original file. This change had an impact in reducing the total number of parameters.
2. Batch Normalization was done after each convolution (except for the last convolution).
3. Dropouts were added everytime Max Pooling was done. 
4. Finally, Global Average Pooling was implemented at the end of the network.

Additionally, the batch size was reduced to 64.

The resultant Network was created: 

![Network](https://github.com/sahalk/eva/blob/master/images/Network.png)

The total number of parameters were reduced from approximately 6.3 million to 14,194. The following is the architecture of the Network: 

![Architecture](https://github.com/sahalk/eva/blob/master/images/Arc.png)

The following accuracies were acheived on the test set: 

![Accuracies](https://raw.githubusercontent.com/sahalk/eva/master/images/accuracies.png)


## EVA 5 - Session 5
## Step 1

### Target: 

1. Set-up the file by importing all the required libraries.
2. Create the train and test transforms.
3. Loading the MNIST dataset and setting up the data loader.
4. Calculate some statistics related to the data and vizualize how the data actually looks like.
5. Setup a working model; nothing fancy.
6. Results: 
    1. Parameters: 194,884
    2. Best Training Accuracy: 99.54%
    3. Best Test Accuracy: 99.11%
7. Analysis:
    1. The model is very heavy (too many parameters) for this problem.
    2. From the graph, we can observe that the model is over-fitting.


## Step 2

### Target: 

1. To make the model lighter by reducing the total number of parameters.
2. Reduce the total number of epochs to 14.
3. Results: 
    1. Parameters: 8,488
    2. Best Training Accuracy: 98.90%
    3. Best Test Accuracy: 98.57%
4. Analysis:
    1. This model is good and can be modified further to improve accuracies.
    2. The model is not over-fitting.

## Step 3

### Target: 

1. Addition of Batch-norm to improve the models efficiency.
2. Addition of Dropout to the layers.
3. Results: 
    1. Parameters: 8,664
    2. Best Training Accuracy: 99.57%
    3. Best Test Accuracy: 99.24%
4. Analysis:
    1. The model has been regularized.
    2. The overall accuracy of the model has increased. However, we still have not achieved the desired accuracy. Further modification is required.
    
## Step 4

### Target: 

1. Addition of Image Augmentation (Random Rotation).
2. Addition of GAP (Global Average Pooling) layer to the network.
3. Addition of scheduler.
3. Results: 
    1. Parameters: 9,608
    2. Best Training Accuracy: 99.25%
    3. Best Test Accuracy: 99.48%
4. Analysis:
    1. The model has achieved accuracies greater than 99.4% consistently towards the end.
    2. Less than 10k parameters were used for this model.
    3. The model is not over-fitting (Train and Test accuracies are close to each other).
