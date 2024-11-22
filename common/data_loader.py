from tensorflow.keras.datasets import mnist
import numpy as np

def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
    return T

def load_mnist(normalize=True, one_hot_label=True, verbose=True):
    if verbose:
        print("데이터를 로드하고 있습니다...")
        
    (x_train, t_train), (x_test, t_test) = mnist.load_data()
    
    if verbose:
        print("데이터 형상:")
        print(f"x_train: {x_train.shape}")
        print(f"t_train: {t_train.shape}")
        print(f"x_test: {x_test.shape}")
        print(f"t_test: {t_test.shape}\n")
    
    # reshape
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)
    
    if normalize:
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
    if one_hot_label:
        t_train = _change_one_hot_label(t_train)
        t_test = _change_one_hot_label(t_test)
    
    return (x_train, t_train), (x_test, t_test)