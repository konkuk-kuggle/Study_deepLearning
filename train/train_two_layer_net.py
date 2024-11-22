# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from PIL import Image
from networks.two_layer_net import TwoLayerNet
from common.trainer import Trainer
from common.data_loader import load_mnist

def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
    return T

# 데이터 로드
(x_train, t_train), (x_test, t_test) = load_mnist()

# 2층 신경망 생성
network = TwoLayerNet(input_size=784, 
                     hidden_size=50,
                     output_size=10)

# 학습 실행
trainer = Trainer(network, x_train, t_train, x_test, t_test, epochs=100)
trainer.train()
trainer.plot() 