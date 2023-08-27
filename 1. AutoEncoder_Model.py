from keras.datasets import mnist
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model

# MNIST 데이터 로드
(x_train, _), (x_test, _) = mnist.load_data()

# 데이터 전처리 (정규화 및 형태 변경)
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
# 픽셀 값을 [0, 1] 사이로 변환
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
# 28*28 형태의 이미지를 784차원으로 변경

# 인코딩될 표현(representation)의 크기
encoding_dim = 32

# 입력 레이어 정의
input_img = Input(shape=(784,))

# "encoded"는 입력의 인코딩된 표현
encoded = Dense(encoding_dim, activation='relu')(input_img)

# "decoded"는 입력의 손실있는 재구성 (lossy reconstruction)
decoded = Dense(784, activation='sigmoid')(encoded)

# 입력을 출력으로 매핑하는 Autoencoder 모델을 생성
autoencoder = Model(input_img, decoded)

# 별도의 인코더 모델 생성
encoder = Model(input_img, encoded)

# 컴파일
loss = autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# 훈련
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))
