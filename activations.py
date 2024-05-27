import numpy as np

# activation function and its derivative
def tanh(x):
    return np.tanh(x);

def tanh_prime(x):
    return 1-np.tanh(x)**2;

def relu(x):
    #x=np.array(x)
    return np.maximum(x , 0)

def relu_prime(x):
    return np.where(x <= 0, 0, 1)

def softmax(x):
    exps = np.exp(x - np.max(x))  # Subtracting max(x) for numerical stability
    return exps / np.sum(exps)

def softmax_prime(x):
    s = softmax(x)
    return np.diagflat(s) - np.outer(s, s)

