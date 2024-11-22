# coding: utf-8
import sys, os
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient

class FiveLayerNet:

    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size1)
        self.params['b1'] = np.zeros(hidden_size1)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size1, hidden_size2)
        self.params['b2'] = np.zeros(hidden_size2)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size2, hidden_size3)
        self.params['b3'] = np.zeros(hidden_size3)
        self.params['W4'] = weight_init_std * np.random.randn(hidden_size3, hidden_size4)
        self.params['b4'] = np.zeros(hidden_size4)
        self.params['W5'] = weight_init_std * np.random.randn(hidden_size4, output_size)
        self.params['b5'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2, W3, W4, W5 = self.params['W1'], self.params['W2'], self.params['W3'], self.params['W4'], self.params['W5']
        b1, b2, b3, b4, b5 = self.params['b1'], self.params['b2'], self.params['b3'], self.params['b4'], self.params['b5']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2, W3) + b3
        z3 = sigmoid(a3)
        a4 = np.dot(z3, W4) + b4
        z4 = sigmoid(a4)
        a5 = np.dot(z4, W5) + b5
        y = softmax(a5)
        
        return y
        
    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        grads['W3'] = numerical_gradient(loss_W, self.params['W3'])
        grads['b3'] = numerical_gradient(loss_W, self.params['b3'])
        grads['W4'] = numerical_gradient(loss_W, self.params['W4'])
        grads['b4'] = numerical_gradient(loss_W, self.params['b4'])
        grads['W5'] = numerical_gradient(loss_W, self.params['W5'])
        grads['b5'] = numerical_gradient(loss_W, self.params['b5'])
        
        return grads
        
    def gradient(self, x, t):
        W1, W2, W3, W4, W5 = self.params['W1'], self.params['W2'], self.params['W3'], self.params['W4'], self.params['W5']
        b1, b2, b3, b4, b5 = self.params['b1'], self.params['b2'], self.params['b3'], self.params['b4'], self.params['b5']
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
        z4 = sigmoid(a4)
        a5 = np.dot(z4, W5) + b5
        y = softmax(a5)
        
        # backward
        dy = (y - t) / batch_num
        grads['W5'] = np.dot(z4.T, dy)
        grads['b5'] = np.sum(dy, axis=0)

        da4 = np.dot(dy, W5.T)
        dz4 = sigmoid_grad(a4) * da4
        grads['W4'] = np.dot(z3.T, dz4)
        grads['b4'] = np.sum(dz4, axis=0)

        da3 = np.dot(dz4, W4.T)
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