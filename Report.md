# 신경망 층별 실험 보고서

## 1. 실험 정보

- **실험자**: <!-- 여기에 이름을 작성하세요 -->
- **실험 날짜**: <!-- YYYY-MM-DD 형식으로 작성하세요 -->

---

## 2. 실험 목적

신경망의 층이 깊어질수록:

1. 학습 능력이 향상되는가?
2. 학습 시간이 얼마나 증가하는가?
3. 과적합이 발생하는가?
4. 기울기 소실 문제가 발생하는가?

---

## 3. 실험 환경 및 설정

- **환경 설정**
  - Python 패키지: `numpy`, `matplotlib`, `tensorflow`
  - 설치 명령어:
    ```bash
    pip install numpy matplotlib tensorflow
    ```
- **공통 실험 조건**
  - 에폭 수: `100`
  - 배치 크기: `100`
  - 학습률: `0.1`

---

## 4. 실험 절차

### 4.1. 코드 실행 순서

- **2층 신경망**: `python train/train_two_layer_net.py`
- **3층 신경망**: `python train/train_three_layer_net.py`
- **4층 신경망**: `python train/train_four_layer_net.py`
- **5층 신경망**: `python train/train_five_layer_net.py`

### 4.2. 각 신경망의 구조

- **2층 신경망**: 입력(784) → 은닉층(50) → 출력(10)
- **3층 신경망**: 입력(784) → 은닉1(100) → 은닉2(50) → 출력(10)
- **4층 신경망**: 입력(784) → 은닉1(100) → 은닉2(80) → 은닉3(60) → 출력(10)
- **5층 신경망**: 입력(784) → 은닉1(100) → 은닉2(80) → 은닉3(60) → 은닉4(40) → 출력(10)

---

## 5. 실험 결과

### 5.1. 2층 신경망

- **최종 훈련 정확도**: <!-- 여기에 값을 입력하세요 -->
- **최종 테스트 정확도**: <!-- 여기에 값을 입력하세요 -->
- **학습에 걸린 시간**: <!-- 여기에 시간을 입력하세요 -->
- **특이사항**:
  <!-- 예: 불안정한 학습, 과적합 등 -->

### 5.2. 3층 신경망

- **최종 훈련 정확도**: 
- **최종 테스트 정확도**: 
- **학습에 걸린 시간**: 
- **특이사항**:
  

### 5.3. 4층 신경망

- **최종 훈련 정확도**: 
- **최종 테스트 정확도**: 
- **학습에 걸린 시간**: 
- **특이사항**:


### 5.4. 5층 신경망

- **최종 훈련 정확도**: 
- **최종 테스트 정확도**: 
- **학습에 걸린 시간**: 
- **특이사항**:


---

## 6. 결과 분석

### 6.1. 학습 속도 비교

- **관찰 내용**:
  - <!-- 각 모델이 높은 정확도에 도달하는 속도를 비교하여 작성하세요 -->

### 6.2. 최종 정확도 비교

- **관찰 내용**:
  - <!-- 각 모델의 최종 성능을 비교하여 작성하세요 -->

### 6.3. 과적합 여부

- **관찰 내용**:
  - <!-- 훈련 정확도와 테스트 정확도의 차이를 분석하여 작성하세요 -->

### 6.4. 안정성

- **관찰 내용**:
  - <!-- 학습 곡선의 안정성을 비교하여 작성하세요 -->

---

## 7. 결론

- **종합 평가**:
  - <!-- 실험 목적에 대한 결과를 종합하여 작성하세요 -->
- **추가적인 고찰**:
  - <!-- 기울기 소실 문제나 다른 관찰된 현상에 대해 작성하세요 -->

---

## 8. 참고 및 참고자료

- **참고한 코드 및 자료**:
  - <!-- 참고한 자료나 코드를 명시하세요 -->
