import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # Add parent directory to path
from common.trainer import Trainer
from common.data_loader import load_mnist
from networks.three_layer_net import ThreeLayerNet

# 데이터 로드
(x_train, t_train), (x_test, t_test) = load_mnist()

# 3층 신경망 생성
network = ThreeLayerNet(input_size=784, 
                       hidden_size1=100,
                       hidden_size2=50,
                       output_size=10)

# 학습 실행
trainer = Trainer(network, x_train, t_train, x_test, t_test, epochs=100)
trainer.train()
trainer.plot() 