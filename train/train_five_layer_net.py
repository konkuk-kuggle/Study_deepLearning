import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # Add parent directory to path
from common.trainer import Trainer
from common.data_loader import load_mnist
from networks.five_layer_net import FiveLayerNet

# 데이터 로드
(x_train, t_train), (x_test, t_test) = load_mnist()

# 5층 신경망 생성
network = FiveLayerNet(input_size=784, 
                      hidden_size1=100,
                      hidden_size2=80,
                      hidden_size3=60,
                      hidden_size4=40,
                      output_size=10)

# 학습 실행
trainer = Trainer(network, x_train, t_train, x_test, t_test, epochs=100)
trainer.train()
trainer.plot()