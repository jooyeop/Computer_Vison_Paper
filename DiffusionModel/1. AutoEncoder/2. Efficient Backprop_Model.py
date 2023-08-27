import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 데이터 로드 및 전처리
data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

'''
논문 : 입력 특성들의 중앙값이 0주변이어야 하며, 그들의 공분산이 효과적으로 학습될 수 있도록 조정되어야 하다는 것이 강조됩니다.
코드 : StandardScaler()를 사용하여 특성들을 표준화하여 평균이 0이고 표준편차가 1이 되도록 만듭니다.

StandardScaler를 사용하는 이유
1. 스케일통일 : 실제 데이터에서특성들은 각각 다른 단위나 스케일을 가질 수있습니다.
예를 들어, 집의 면적을 나타내는 트성은 제곱미터, 집의 가격은 달러로 표시 됩니다.
이러한 차이나는 스케일은 머신러닝 알고리즘의 성능에 부정적인 영향을 줄 수 있습니다.
표준화를 통해 모든 특성의 스케일을 동일하게 만들어 알고리즘의 성능을 최적화 할 수 있습니다.

2. 수렴 속도 개선 : 특히 경사 하강법과 같은 최적화 알고리즘은 표준화된 데이터에서 더 빠르게 수렴하게 됩니다.
스케일이 다른 특성들은 경사의 방향과 크기에 영향을 주어 최적의 솔루션으로 빠르게 수렴하는 것을 방해할 수 있습니다.

3. 정규화의 중요성 : 몇 몇 머신러닝 알고리즘들은 예를들어, 로지스틱 회귀, SVM, 신경망 등은 입력 특성의 스케일에 민감하게 반응합니다.
이러한 알고리즘들은 특성이 표준화되었을때 훨씬 더 잘 작동합니다.

4. 통계적해석 : 표준화된 데이터는 통계적으로 분석하기 더 쉽습니다. 특성의 계수를 해석할 때, 표준화된 특성들은 각각 동일한 스케일에서 비교되므로 계수의 크기가 해당 특성의 중요도를 나타나게 됩니다.

5. 정규화의필요성 : 규제가 포함된 (L1, L2) 알고리즘들은 특성의 스케일에 따라 규제의 효과가 달라지기 때문에 특성들이 동일한 스케일을 가질 때 규제는 모든 특성들에 공평하게 작용하게 됩니다.
'''

# 모델 구성
model = tf.keras.Sequential([
    Dense(8, activation = 'sigmoid', input_shape = (X_train.shape[1],)),
    Dense(3, activation = 'softmax')
])

'''
활성화함수
논문 : sigmoid와 같은 활성화 함수에 대해 특성과 문제점에 대해 논의 합니다.
코드 : sigmoid 활성화 함수를 사용합니다.
'''

# 가중치 초기화
def orthogonal_initialization(shape) :
    if len(shape) == 2 :
        falst_shape = (shape[0], shape[1],)
    else :
        flat_shape = (np.prod(shape),)
    a = np.random.normal(size = flat_shape)
    u, _, v = np.linalg.svd(a.reshape(shape))
    q = u if u.shape == shape else v
    return q.reshape(shape)

model.layers[0].set_weights([orthogonal_initialization((X_train.shape[1], 8)), np.zeros(8)])
model.layers[1].set_weights([orthogonal_initialization((8, 3)), np.zeros(3)])

'''
가중치 초기화
논문 : 효과적인 가중치 초기화 방법이 심층 신경망의 훈련속도와 최종 성능에 큰 영향을 미친다고 설명함.
특히, 작은 초기 가중치는 시그모이드와 가은 활성화 함수에서 그래디언트 소실 문제를 완화할 수 있다
코드 : 직교 초기화 방법을 사용하여 가중치를 초기화 합니다. 이 방법은 논문에서 직접적으로 언급되지 않지만, 효과적인 방법의 한 예로 볼 수 있습니다.

'''

# 모델 컴파일
optimizer = SGD(learning_rate = 0.01)
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

'''
학습률과 최적화
논문 : 학습률의 중요성이 강조됩니다. 너무 큰 학습률은 발산을 초래할 수 있으며, 너무 작은 학습률은 학습이 너무 느려질 수 있습니다. 그래서 SGD와 같은 최적화 방법도 논의 됩니다.
코드 : SGD 최적화기를 사용하여 학습률은 0.01로 설정합니다. 

'''

# 모델 학습
model.fit(X_train, y_train, epochs = 100, batch_size = 16, verbose = 2)

# 모델 평가
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose = 0)
print('test_loss : {:.4f} - test_accuracy : {:.4f}'.format(test_loss, test_accuracy))


