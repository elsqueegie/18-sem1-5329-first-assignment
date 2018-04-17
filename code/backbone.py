import numpy as np

 class Layer:
    
    def __init__(self, n_nodes, activation, n_inputs):
        self.n_nodes = n_nodes
        self.nodes = {}
        self.activation = activation
        self.n_inputs = n_inputs
        afunc = self.activation
        self.nodes = {i:Node(n_inputs, activation=afunc) for i in range(self.n_nodes)}
        
    def get_layer_output(self, df_in):
        print(df_in.shape)
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
    
    def assign_layer(self, n_nodes, activation, n_inputs):
        
        self.layers[self.n_layers] = Layer(n_nodes, activation, n_inputs)
        #self.layers[self.n_layers].set_input(self.to_pass)
        self.to_pass = self.layers[self.n_layers].n_nodes
        self.n_layers += 1
        
    def get_network_output(self):
        data_in = self.in_data.copy()
        for ilayer in self.layers.values():
            data_in = ilayer.get_layer_output(data_in).T
        return data_in
        
    def score_network(self):
        self.output = self.get_network_output()
        self.predict_class = np.array([[i for i,j in enumerate(k) if j==max(k)][0] for k in self.output ])
        self.probs = np.array(
                [compute_softmax_score(self.output[i],self.predict_class[i]) for i in range(len(self.output))]
        )
        
    def get_loss(self):
        m = (self.label == self.predict_class).astype(int)
        self.loss = -(m*np.log(self.probs) + (1-m)*np.log(self.probs))
        return self.loss
    
    def get_batch(self, frac=0.05):
        return np.random.choice(range(len(self.in_data)), replace=False, size=int(len(self.in_data)*frac))
    
    def get_gradient(self, batch):
        try:
            loss = self.loss
        except:
            loss = self.get_loss
        
        data = self.data_in[batch]
        
        layers = list(range(len(self.n_layers)))
        layers.reverse
        
        for layer in layers:
            for node in layer:
                None
    
    def train(self, input_data):
        None


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
            self.weights=np.random.rand(n_features) / 50
        else:
            self.weights=weights
        self.activation_func = activation
        self.train_rate = train_rate
        self.max_iter = max_iter
        
    def set_input(self,in_data):
        # Set the input data for the node, must have number of features equal to n_features used for initialisation
        self.in_data = in_data
        
    def score_input(self, in_data, weights='None', alphas='None'):
        # Apply the weights to the features and return the output for the data set in set_input
        if weights=='None':
            weights = self.weights
        self.in_data = in_data.copy()
        #print(self.in_data.shape)
        self.output = activate(node_mult(self.in_data, weights),kind=self.activation_func)
        #print(self.output.shape)
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
    
    def get_gradients(self):
        afunc = self.activation
        gradfunc = get_gradient(afunc)
        for i in 
        
            
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

def node_mult(in_data, weights):
    # Multiply each feature (including constant) by its weight then sum the result
    in_data = in_data.copy()
    for d in range(in_data.shape[1]):
        in_data[:,d] = weights[d] * in_data[:,d]
    return in_data.sum(axis=1)

def grad_sigmoid(x):
    #returns the gradient of a sigmoid at point x
    return sigmoid(x) * (1-sigmoid(x))

def grad_tanh(x):
    #returns the gradient of a tanh at point x
    return 1 - np.tanh(x)**2

def grad_relu(x):
    #returns the gradient of a relu at point x
    if x > 0:
        return 1
    else:
        return 0

def grad_leaky_relu(x, alpha=0.05):
    #returns the gradient of a leaky_relu at point x
    if x > 0:
        return 1
    else:
        return alpha
    
def grad_softmax(prediction_list, iclass):
    j = np.exp(prediction_list)
    i = j
    
def get_gradient(activation_function):
    gradient_dic = {
        'relu':grad_relu,
        'leaky_relu':grad_leaky_relu,
        'tanh':grad_tanh,
        'sigmoid':grad_sigmoid
    }
    return gradient_dic[activation_function]

def activate(in_data, kind='relu'):
    # Apply an activation function to a node's output
    actionary = {
        'relu':relu,
        'leaky_relu':leaky_relu,
        'sigmoid':sigmoid,
        'tanh':tanh
    }
    return actionary[kind](in_data)

def compute_softmax_score(prediction_list, iclass):
    j = np.exp(prediction_list)
    j = j/j.sum()
    return j[iclass] / j.sum()
    
def compute_cross_entropy_loss(yhat):
    return 0 - np.log(yhat)


