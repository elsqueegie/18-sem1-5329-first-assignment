import numpy as np

class Layer:
    
    def __init__(self, n_nodes, activation, n_inputs, weights=np.ndarray):
        self.n_nodes = n_nodes
        self.nodes = {}
        self.activation = activation
        self.n_inputs = n_inputs
        afunc = self.activation
        if isinstance(weights,np.ndarray):
            self.nodes = {i:Node(n_inputs, activation=afunc, weights=weights) for i in range(self.n_nodes)}
        else:
            self.nodes = {i:Node(n_inputs, activation=afunc) for i in range(self.n_nodes)}
        
    def get_layer_output(self, df_in):
        self.output = np.array([n.score_input(df_in) for n in self.nodes.values()])
        return self.output
        
 
class Network:
    
    def __init__(self):
        self.layers = {}
        self.in_data = None
        self.n_layers = 0
        
    def set_indata(self, in_data, label):
        self.in_data = in_data
        self.label = label
        self.in_features = in_data.shape[1]
        self.to_pass = self.in_data
    
    def assign_layer(self, n_nodes, activation, n_inputs, weights=None):
        self.layers[self.n_layers] = Layer(n_nodes, activation, n_inputs, weights)
        #self.layers[self.n_layers].set_input(self.to_pass)
        self.to_pass = self.layers[self.n_layers].n_nodes + 1
        self.n_layers += 1
        
    def feed_forward(self):
        self.outputs_by_layer = []
        data_in = self.in_data.copy()
        for ilayer in self.layers.values():
            data_in = ilayer.get_layer_output(data_in).T
            self.outputs_by_layer.append(data_in.copy())
        self.output = data_in
        return self.outputs_by_layer
        
    def score_network(self):
        self.feed_forward()
        self.prediction = np.array([
            [j for j,k in enumerate(i) if k == i.max()][0] for i in self.output
        ])
        self.softmax_score = np.array(
                [compute_softmax_score(self.output[i],self.label[i]) for i in range(len(self.output))]
        )
        self.error = compute_cross_entropy_loss(self.softmax_score)
        
    def get_loss(self):
        return np.mean([compute_cross_entropy_loss(
                compute_softmax_score(self.output[i],self.label[i])
        ) for i in range(len(self.label))])
    
    def get_batched_loss(self,batch):
        loss = [compute_cross_entropy_loss(compute_softmax_score(self.output[i], self.label[i])) for i in batch]
        return loss
    
    def get_batch(self, frac=0.05):
        return np.random.choice(range(len(self.in_data)), replace=False, size=int(len(self.in_data)*frac))
    
    def get_node_gradient_at_weight(self, ilayer, inode, iweight, in_data):
        node = self.layers[len(self.layers)-ilayer].nodes[inode]
        grad = get_gradient(node.activation_func)(np.mean(in_data[iweight]))
        return grad
            
    def update_weight(self,ilayer,inode,iweight,delta):
        self.layers[layer].nodes[inode].weights[iweight] += delta
        
    def backpropagate(self):
        count = 0
        error = self.error
        _grad = np.mean([grad_loss_by_output(self.output[i],self.label[i]) for i,j in enumerate(self.output)])
        for ilayer in range(self.n_layers):
            ilayer = max(self.layers.keys()) - ilayer
            gradient=0
            for pos,inode in enumerate(self.layers[ilayer].nodes.values()):
                if count > 0:
                    grad = _grad + np.sum([n.weights[pos] for n in self.layers[ilayer+1].nodes.values()])
                else:
                    grad = _grad
                inode.grads = []
                for inum in range(len(inode.weights)):
                    gfunc = get_gradient(inode.activation_func)
                    inode.grads.append(gfunc(grad*np.mean(inode.in_data.T[inum])))
                if count > 1:
                    _grad += np.sum([n.weights[pos] for n in self.layers[ilayer+2].nodes.values()])
            count += 1

    def train(self, iters, train_rate = 0.005):
        old_error = (self.error.mean())
        for i in range(iters):
            self.backpropagate()
            for ilayer in range(self.n_layers):
                for inode in self.layers[ilayer].nodes.values():
                    self.inode = inode
                    inode.weights += (np.array(inode.grads) * train_rate)
            self.score_network()
            print(old_error - self.error.mean())
            old_error = self.error.mean()
        
    def get_batched_network_output(self,batch):
        data_in = self.in_data.copy()[batch]
        for ilayer in self.layers.values():
            data_in = ilayer.get_layer_output(data_in).T
        return data_in


class Node(object):
    '''
    A Node is a modular element of a neural network. It is defined by:
    
    in_data - the feature inputs, including a constant feature
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
                 max_iter=1000):
        self.n_features = n_features
        if weights=='None':
            self.weights=np.random.rand(n_features+1) / 50
        else:
            self.weights=weights
        self.activation_func = activation
        self.train_rate = train_rate
        self.max_iter = max_iter
        
    #def set_input(self,in_data):
        # Set the input data for the node, must have number of features equal to n_features + constant
        #self.in_data = add_constant(in_data)
        
    def score_input(self, in_data, weights='None', alphas='None'):
        # Apply the weights to the features and return the output for the data set in set_input
        if weights=='None':
            weights = self.weights
        self.in_data = add_constant(in_data)
        self.output = activate(node_mult(self.in_data, weights),kind=self.activation_func)
        return self.output
    
    def score_gradients(self,increment=1e-4):
        # Calculate the average gradient for each coefficient via a tiny increment over the input data
        self.grads_ = []
        for i in range(self.in_data.shape[1]):
            self.new_weights = self.weights.copy()
            self.new_weights[i] = self.new_weights[i] + increment
            j1 = self.score_input()
            j2 = self.score_input(weights=self.new_weights)
            self.grads_.append(np.mean(j1-j2)/increment)
        self.grads_ = np.array(self.grads_)
        return self.grads_
    
#    def get_gradients(self):
#        afunc = self.activation
#        gradfunc = get_gradient(afunc)
#        for i in 
        
            
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
            if i % 200 == 0:
                print(np.max(old - new))
            if np.max(old - new) < 0.0000001:
                break
        return i

def add_constant(data):
    return np.c_[data, np.ones(len(data))]

def relu(column):
    # activation function that gives x if x > 0 else 0
    return np.max(np.array([np.zeros((len(column),)),column]),axis=0)

def leaky_relu(column, alpha=0.005):
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

def pre_softmax(column):
    return np.exp(column)

def node_mult(in_data, weights, softmax=False):
    # Multiply each feature (including constant) by its weight then sum the result
    in_data = in_data.copy()
    if in_data.shape[1] != len(weights):
        raise ValueError("Input matrix doesn't match weight vector")
    for d in range(in_data.shape[1]):
        in_data[:,d] = weights[d] * in_data[:,d]
    if softmax:
        in_data = np.exp(in_data)
        return in_data/in_data.sum(axis=1).reshape((len(in_data),1))
    else:
        return in_data.sum(axis=1)

#def node_softmax(in_data, weights):
#    # Multiply each feature (including constant) by its weight then sum the result
#    in_data = in_data.copy()
#    for d in range(in_data.shape[1]):
#        in_data[:,d] = np.exp(weights[d] * in_data[:,d])
#    in_data = in_data / in_data.sum(axis=1)
#    return in_data

def grad_sigmoid(x):
    #returns the gradient of a sigmoid at point x
    return sigmoid(x) * (1-sigmoid(x))

def grad_tanh(x):
    #returns the gradient of a tanh at point x
    return 1 - np.tanh(x)**2

def grad_relu(x):
    #returns the gradient of a relu at point x
    return (np.array(x) > 0) - 0.0

def grad_leaky_relu(x, alpha=0.005):
    #returns the gradient of a leaky_relu at point x
    to_ret = np.array(grad_relu(x))
    to_ret[to_ret<=0] = alpha
    return to_ret
    
def grad_softmax(prediction_list, iclass):
    j = np.exp(prediction_list)
    j =  j/j.sum()
    yhat = j[iclass]
    return yhat - 1
    
def get_gradient(activation_function):
    gradient_dic = {
        'relu':grad_relu,
        'leaky_relu':grad_leaky_relu,
        'tanh':grad_tanh,
        'sigmoid':grad_sigmoid,
        'softmax':grad_softmax
    }
    return gradient_dic[activation_function]

def activate(in_data, kind='relu'):
    # Apply an activation function to a node's output
    actionary = {
        'relu':relu,
        'leaky_relu':leaky_relu,
        'sigmoid':sigmoid,
        'tanh':tanh,
        'pre_softmax':pre_softmax
    }
    return actionary[kind](in_data)

def compute_softmax_score(prediction_list, iclass):
    j = np.exp(prediction_list)
    j = j[iclass]/j.sum()
    return j
    
def compute_cross_entropy_loss(yhat):
    return 0 - np.log(yhat)

def grad_loss_by_output(scores, iclass):
    return compute_softmax_score(scores, iclass) - 1
