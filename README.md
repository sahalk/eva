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
