
1. Introduction
Classification is one a common problem well-suited to machine learning. However, many machine-learning techniques cannot leverage the opportunities presented by large datasets with many features. To solve this problem we would be implementing a neural network to make full use of the width and length of the dataset. Neural networks consist of a large number of highly connected processing units called as neurons. These neurons are organised into layers, and a neural network consists of at least one hidden layer and one output layer. Hidden layers require non-linear activation functions, otherwise the result of multiple layers could be achieved by a single node. The non-linear function to be deployed is Rectified Linear Unit (ReLU), the details of which are discussed below. 

Neural networks can face issues with training. Training on the whole data simultaneously creates a large computational overhead for little performance gain. Training on a single observation can produce unpredictable results that do not generalise. Additionally, neural networks are prone to exploding coefficients and can suffer from the zero-gradient trap with the chosen ReLU activation. In order to address these challenges, the following features will be deployed within the neural network: 

* Dropout
* Batch Normalization
* Weight Normalization
* Minibatch SGD
* SGD with Momentum

Since this is a multiclass problem with 10 unique classes, a Softmax activation has been chosen for the output layer, allowing for the calculation and comparison of class probabilities.

1.1 Data
There are 10 classes in this dataset. The dataset has been split into training set and test set, where the training set has 60,000 examples and the test set has 10,000 examples.
