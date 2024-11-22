import numpy as np
from common.functions import *

class FourLayerNet:
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size1)
        self.params['b1'] = np.zeros(hidden_size1)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size1, hidden_size2)
        self.params['b2'] = np.zeros(hidden_size2)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size2, hidden_size3)
        self.params['b3'] = np.zeros(hidden_size3)
        self.params['W4'] = weight_init_std * np.random.randn(hidden_size3, output_size)
        self.params['b4'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2, W3, W4 = self.params['W1'], self.params['W2'], self.params['W3'], self.params['W4']
        b1, b2, b3, b4 = self.params['b1'], self.params['b2'], self.params['b3'], self.params['b4']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2, W3) + b3
        z3 = sigmoid(a3)
        a4 = np.dot(z3, W4) + b4
        y = softmax(a4)

        return y

    # loss와 accuracy 메서드는 ThreeLayerNet과 동일

    def gradient(self, x, t):
        W1, W2, W3, W4 = self.params['W1'], self.params['W2'], self.params['W3'], self.params['W4']
        b1, b2, b3, b4 = self.params['b1'], self.params['b2'], self.params['b3'], self.params['b4']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2, W3) + b3
        z3 = sigmoid(a3)
        a4 = np.dot(z3, W4) + b4
        y = softmax(a4)
        
        # backward
        dy = (y - t) / batch_num
        grads['W4'] = np.dot(z3.T, dy)
        grads['b4'] = np.sum(dy, axis=0)

        da3 = np.dot(dy, W4.T)
        dz3 = sigmoid_grad(a3) * da3
        grads['W3'] = np.dot(z2.T, dz3)
        grads['b3'] = np.sum(dz3, axis=0)

        da2 = np.dot(dz3, W3.T)
        dz2 = sigmoid_grad(a2) * da2
        grads['W2'] = np.dot(z1.T, dz2)
        grads['b2'] = np.sum(dz2, axis=0)

        da1 = np.dot(dz2, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads