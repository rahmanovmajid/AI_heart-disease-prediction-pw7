import numpy as np
#Helper functions
def relu(array):
    return np.maximum(0,array)
def reluDeriv_vectorial(  array):
    grad = np.zeros((array.shape))
    for i in range(len(array)):
        if array[i]>0:
            grad[i]=1
    return grad
def reluDeriv_matricial(  matrix):
    grad = np.zeros((matrix.shape))
    for i in range(matrix.shape[1]):
        grad[:,i] =   reluDeriv_vectorial(matrix[:,i])
    return grad
def tanh(  array):
    return 2.0/(1+np.exp(-2*array)) - 1
def tanhDeriv(  array):
    return 1-np.power(  tanh(array),2)

def sigmoid(array):
    return 1.0/(1.0 + np.exp(-array))
def sigmoidDeriv(array):
    return sigmoid(array)*(1-sigmoid(array))
def softplus(array):
    return np.log(1+np.exp(array))
def softplusDeriv(array):
    return 1.0/(1+np.exp(-array))
def entropy(Y,Yhat):
    m = Y.shape[1]
    return -1.0/m * (np.sum(np.multiply(Y,np.log(Yhat)) + np.multiply(1-Y,np.log(1-Yhat))))