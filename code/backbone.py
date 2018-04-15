import numpy as np

def relu(column):
    # activation function that gives x if x > 0 else 0
    return np.max(np.array([np.zeros((len(column),)),column]),axis=0)

def leaky_relu(column, alpha=0.05):
    # activation function that draws on ReLU but has a slight gradient for x < 0
    return np.max(np.array([column*alpha,column]),axis=0)

def sigmoid(column):
    # activation function that returns a value between 0 and 1, good for probabilities
    return 1/(1+np.exp(-column))

def tanh(column):
    # Activation function that returns a value between -1 and 1
    # problems occurred with large negatives when applying (1 - np.exp(-column)) / (1 + np.exp(-column)) 
    # using numpy equivalent instead
    return np.tanh(column)

def softmax(inputs):
    # Applies the softmax function to a list of outputs
    j = np.array([np.exp(i) for i in inputs])
    return j/j.sum()

def node_mult(d_in, weights):
    # Multiply each feature (including constant) by its weight then sum the result
    d_in = d_in.copy()
    for d in range(d_in.shape[1]):
        d_in[:,d] = weights[d] * d_in[:,d]
    return d_in.sum(axis=1)

def activate(d_in, kind='relu'):
    # Apply an activation function to a node's output
    actionary = {
        'relu':relu,
        'leaky_relu':leaky_relu,
        'sigmoid':sigmoid,
        'tanh':tanh
    }
    return actionary[kind](d_in)



class Node(object):
    '''
    A Node is a modular element of a neural network. It is defined by:
    
    data_in - the feature inputs, including a constant feature
    activation - the activation function to be applied to the node output
    weights - the coefficients to be applied to the features positionally
    train_rate - the rate at which gradient descent updates the weights of the node
    max_iter - the number of training steps to take before ending a training session
    '''
    
    def __init__(self, 
                 n_features, 
                 activation='sigmoid', 
                 weights='None', 
                 train_rate=0.01, 
                 max_iter=10000):
        self.n_features = n_features
        if weights=='None':
            self.weights=np.random.rand(n_features)
        else:
            self.weights=weights
        self.activation_func = activation
        self.train_rate = train_rate
        self.max_iter = max_iter
        
    def set_input(self,data_in):
        # Set the input data for the node, must have number of features equal to n_features used for initialisation
        self.data_in = data_in
        
    def score_input(self, weights='None', alphas='None'):
        # Apply the weights to the features and return the output for the data set in set_input
        if weights=='None':
            weights = self.weights
        data_in = self.data_in.copy()
        self.output = activate(node_mult(self.data_in, weights),kind=self.activation_func)
        return self.output
    
    def score_gradients(self):
        # Calculate the average gradient for each coefficient via a tiny increment over the input data
        self.grads_ = []
        increment = 0.000001
        for i in range(self.data_in.shape[1]):
            self.new_weights = self.weights.copy()
            self.new_weights[i] = self.new_weights[i] + increment
            j1 = self.score_input()
            j2 = self.score_input(weights=self.new_weights)
            self.grads_.append(np.mean(j1-j2)/increment)
        self.grads_ = np.array(self.grads_)
        return self.grads_
            
    def update_weights(self):
        # update the coefficients in the direction of the gradient
        # TODO - set to update against direction of the error when error calculation is done
        self.score_gradients()
        self.weights = self.weights + self.grads_ * self.train_rate
        return self.weights
        
    def gradient_descend(self):
        # recompute coefficients until gradients flatten or max_iter is reached
        old = self.weights[:]
        for i in range(self.max_iter):
            new = self.update_weights()
            #print(np.max(old - new))
            if np.max(old - new) < 0.0000001:
                break
        return i
