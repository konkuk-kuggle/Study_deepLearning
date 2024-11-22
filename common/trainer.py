import numpy as np
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, network, x_train, t_train, x_test, t_test,
                 epochs=100, batch_size=100, learning_rate=0.1):
        self.network = network
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / batch_size, 1)
        self.total_iters = int(self.epochs * self.iter_per_epoch)
        
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []
        
    def train(self):
        print("학습을 시작합니다...")
        print(f"전체 에폭 수: {self.epochs}")
        print(f"미니배치 크기: {self.batch_size}")
        print(f"학습률: {self.learning_rate}")
        print(f"은닉층 크기: {', '.join(str(self.network.params[f'W{i}'].shape[1]) for i in range(1, len(self.network.params)//2))}\n")

        start = time.time()
        print(f"시작 시간: {start}") # 1970년 1월 1일 00:00:00 (UTC) 를 기준으로 경과된 시간
        for i in range(self.total_iters):
            batch_mask = np.random.choice(self.train_size, self.batch_size)
            x_batch = self.x_train[batch_mask]
            t_batch = self.t_train[batch_mask]
            
            grad = self.network.gradient(x_batch, t_batch)
            
            for key in self.network.params.keys():
                self.network.params[key] -= self.learning_rate * grad[key]
            
            loss = self.network.loss(x_batch, t_batch)
            self.train_loss_list.append(loss)
            
            if i % self.iter_per_epoch == 0:
                self._print_progress(i, loss)

        end = time.time()
        print(f"종료 시간: {end}")  
        print("\n학습이 완료되었습니다!")
        print(f"학습에 소요된 시간: {end-start:.2f}초\n")
        print(f"최종 훈련 정확도: {self.train_acc_list[-1]:.4f}")
        print(f"최종 테스트 정확도: {self.test_acc_list[-1]:.4f}")
    
    def _print_progress(self, i, loss):
        train_acc = self.network.accuracy(self.x_train, self.t_train)
        test_acc = self.network.accuracy(self.x_test, self.t_test)
        self.train_acc_list.append(train_acc)
        self.test_acc_list.append(test_acc)
        epoch = int(i / self.iter_per_epoch)
        print(f"에폭 {epoch:>4} | 손실: {loss:.4f} | 훈련 정확도: {train_acc:.4f} | 테스트 정확도: {test_acc:.4f}")
    
    def plot(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), height_ratios=[1, 2])
        x = np.arange(len(self.train_acc_list))
        
        # 전체 범위 그래프 (위)
        ax1.plot(x, self.train_acc_list, label='train acc', marker='o', markevery=1)
        ax1.plot(x, self.test_acc_list, label='test acc', linestyle='--', marker='s', markevery=1)
        ax1.set_ylim(0, 1.0)
        ax1.grid(True)
        ax1.legend(loc='lower right')
        ax1.set_title("Learning Progress (Full Range)")
        
        # 확대된 그래프 (아래)
        ax2.plot(x, self.train_acc_list, label='train acc', marker='o', markevery=1)
        ax2.plot(x, self.test_acc_list, label='test acc', linestyle='--', marker='s', markevery=1)
        ax2.set_ylim(0.9, 1.0)  # 0.9 이상 구간 확대
        ax2.grid(True)
        ax2.legend(loc='lower right')
        ax2.set_title("Learning Progress (Zoomed: 0.9~1.0)")
        
        # x축 레이블은 아래 그래프에만
        ax2.set_xlabel("Epochs")
        
        # y축 레이블
        fig.text(0.04, 0.5, "Accuracy", va='center', rotation='vertical')
        
        plt.tight_layout()
        plt.show()