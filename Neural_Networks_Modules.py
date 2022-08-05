import numpy as np

# ACtivations  -------------------------------------------------------------------------

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(x, 0)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    expX = np.exp(x)
    return expX/np.sum(expX, axis = 0)

def derivative_tanh(x):
    return (1 - np.power(np.tanh(x), 2))

def derivative_relu(x):
    return np.array(x > 0, dtype = np.float32)

# Intialize Parameter------------------------------------------------------------------------

def initialize_parameters(n_x, n_h, n_y):
    w1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h, 1))
    
    w2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros((n_y, 1))
    
    parameters = {
        "w1" : w1,
        "b1" : b1,
        "w2" : w2,
        "b2" : b2
    }
    
    return parameters

# Forward Propagation -------------------------------------------------------------------------------

def forward_propagation(x, parameters):
    
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']
    
    z1 = np.dot(w1, x) + b1
    a1 = tanh(z1)
    
    z2 = np.dot(w2, a1) + b2
    a2 = softmax(z2)
    
    forward_cache = {
        "z1" : z1,
        "a1" : a1,
        "z2" : z2,
        "a2" : a2
    }
    
    return forward_cache

# Cost Function ------------------------------------------------------------------------------------------

def cost_function(a2, y):
    m = y.shape[1]
    
    cost = -(1/m)*np.sum(y*np.log(a2))
    
    
    return cost

# BAck Propagtion ---------------------------------------------------------------------------------------------

def backward_prop(x, y, parameters, forward_cache):
    
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']
    
    a1 = forward_cache['a1']
    a2 = forward_cache['a2']
    
    m = x.shape[1]
    
    dz2 = (a2 - y)
    dw2 = (1/m)*np.dot(dz2, a1.T)
    db2 = (1/m)*np.sum(dz2, axis = 1, keepdims = True)
    
    dz1 = (1/m)*np.dot(w2.T, dz2)*derivative_tanh(a1)
    dw1 = (1/m)*np.dot(dz1, x.T)
    db1 = (1/m)*np.sum(dz1, axis = 1, keepdims = True)
    
    gradients = {
        "dw1" : dw1,
        "db1" : db1,
        "dw2" : dw2,
        "db2" : db2
    }
    
    return gradients

# Prameters Updation -------------------------------------------------------------------------------------------

def update_parameters(parameters, gradients, learning_rate):
    
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']
    
    dw1 = gradients['dw1']
    db1 = gradients['db1']
    dw2 = gradients['dw2']
    db2 = gradients['db2']
    
    w1 = w1 - learning_rate*dw1
    b1 = b1 - learning_rate*db1
    w2 = w2 - learning_rate*dw2
    b2 = b2 - learning_rate*db2
    
    parameters = {
        "w1" : w1,
        "b1" : b1,
        "w2" : w2,
        "b2" : b2
    }
    
    return parameters

# Model Function for Learning

def model(train_input, train_output, hidden_unit, learning_rate, iterations ,test_input ,test_output ):
    
    input_neuron = train_input.shape[0]
    output_neuron = train_output.shape[0]
    
    train_cost_list = []
    test_cost_list = []
    parameters = initialize_parameters(input_neuron, hidden_unit, output_neuron)
    
    for i in range(iterations):
        
        forward_cache = forward_propagation(train_input, parameters)
        
        cost = cost_function(forward_cache['a2'], train_output)
        
        gradients = backward_prop(train_input, train_output, parameters, forward_cache)
        
        parameters = update_parameters(parameters, gradients, learning_rate)
        
        train_cost_list.append(cost)
        
        if(i%5 == 0):
            print("Cost after", i, "iterations is :", cost)

        test_forward_cache = forward_propagation(test_input ,Parameters)  
        test_cost_list.append( cost_function(test_forward_cache['a2'], test_output))

    return parameters, train_cost_list ,test_cost_list