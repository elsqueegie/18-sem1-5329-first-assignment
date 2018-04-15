import numpy as np

class Node(object):
    def __init__(self, data_in, activation, train_rate=0.01, max_iter=10000):
        self.weights = np.random.rand(data_in.shape[1])
        self.alphas = np.random.rand(data_in.shape[1])
        self.actifunc = activation
        self.data_in = data_in
        self.train_rate = train_rate
        self.max_iter = max_iter
        
    def score_input(self, weights='None', alphas='None'):
        if weights=='None':
            weights = self.weights
        if alphas=='None':
            alphas = self.alphas
        data_in = self.data_in.copy()
        self.output = activate(node_mult(self.data_in, weights, alphas),kind=self.actifunc)
        return self.output
    
    def score_gradients(self):
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
        self.score_gradients()
        self.weights = self.weights + self.grads_ * self.train_rate
        return self.weights
        
    def gradient_descend(self):
        old = self.weights[:]
        for i in range(self.max_iter):
            new = self.update_weights()
            print(np.max(old - new))
            if np.max(old - new) < 0.0000001:
                break
        return i


def relu(column):
    return np.max(np.array([np.zeros((len(column),)),column]),axis=0)

def leaky_relu(column, alpha=0.05):
    return np.max(np.array([column*alpha,column]),axis=0)

def sigmoid(column):
    return 1/(1+np.exp(-column))

def tanh(column):
    # problems with large negatives when applying
    # (1 - np.exp(-column)) / (1 + np.exp(-column)) so just using numpy equivalent
    return np.tanh(column)

def softmax(inputs):
    j = np.array([np.exp(i) for i in inputs])
    return j/j.sum()

def node_mult(d_in, weights, alphas):
    #Apply weights to each dimension of X
    d_in = d_in.copy()
    for d in range(d_in.shape[1]):
        d_in[:,d] = weights[d] * d_in[:,d]
        d_in[:,d] = alphas[d] + d_in[:,d]
    return d_in.sum(axis=1)

def activate(d_in, kind='relu'):
    actionary = {
        'relu':relu,
        'leaky_relu':leaky_relu,
        'sigmoid':sigmoid,
        'tanh':tanh
    }
    return actionary[kind](d_in)
