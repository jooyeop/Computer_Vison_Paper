import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

# 1. 비지도 학습으로 특징 추출

# 오토인코더 정의
input_img = Input(shape=(784,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
#인코더 : 784개의 픽셀 값을 가진 이미지를 64개의 픽셀 값을 가진 이미지로 압축

decoded = Dense(128, activation='relu')(encoded)
decoded = Dense(784, activation='sigmoid')(decoded)
# 디코더 : 64개의 픽셀 값을 가진 이미지를 784개의 픽셀 값을 가진 이미지로 복원

autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)
# 전체 오토인코더와 인코더를 각각 모델로 정의

autoencoder.compile(optimizer='adam', loss='mse')
# 오토인코더를 컴파일

# 데이터 로드 및 전처리
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), 784))
x_test = x_test.reshape((len(x_test), 784))
# MNIST 데이터를 로드하고, 전처리하여 0~1사이의 값으로 정규화

autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True)
# 오토인코더를 학습 : x_train을 입력받고, x_train을 재구성하는 것을 목표로 함

# 2. 추출된 특징을 사용하여 지도 학습

encoded_input = Input(shape=(64,))
classifier_layers = Dense(32, activation='relu')(encoded_input)
classifier_layers = Dense(10, activation='softmax')(classifier_layers)
# 지도 학습을 위한 분류기 정의
# 인코더의 출력을 입력으로 받고, 10개의 클래스를 분류하는 분류기 정의

classifier = Model(encoded_input, classifier_layers)
# 분류기를 모델로 정의


# 특징 추출
x_train_encoded = encoder.predict(x_train)
x_test_encoded = encoder.predict(x_test)
# 앞서 정의한 인코더를 사용하여 MNIST 데이터의 특징을 추출

classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
classifier.fit(x_train_encoded, y_train, epochs=50, batch_size=256, shuffle=True)
# 추출된 특징을 사용하여 분류기를 학습
# x_train_encoded를 입력받고, y_train을 분류하는 것을 목표로 함

'''
과정
1. 데이터 로딩 및 전처리
- MNIST 데이터셋을 로드
- 데이터를 0과 1 사이의 값으로 정규화하여 오토인코더의 입력으로 사용할 수 있도록 함
    - (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    - x_train = x_train.astype('float32') / 255.
    - x_test = x_test.astype('float32') / 255.
    - x_train = x_train.reshape((len(x_train), 784))
    - x_test = x_test.reshape((len(x_test), 784))
2. 오토인코더 구축
- Input레이어를 사용하여 784개의 픽셀 값을 가진 입력 이미지를 정의
- 두 개의 'Dense'레이어를 사용하여 이미지를 점진적으로 아북하여 중간의 잠재표현을 생성
- 두 개의 'Dense'레이어를 사용하여 잠재 표현을 다시 원래의 784개의 픽셀 값을 가진 이미지로 복원
- 이러한 인코딩 및 디코딩 과정으로 이토인코더 모델을 구축
    - input_img = Input(shape=(784,))
    - encoded = Dense(128, activation='relu')(input_img)
    - encoded = Dense(64, activation='relu')(encoded)
    - decoded = Dense(128, activation='relu')(encoded)
    - decoded = Dense(784, activation='sigmoid')(decoded)
    - autoencoder = Model(input_img, decoded)
3. 오토인코더 학습
- 오토인코더를 학습하여 입력 이미지를 잘 재구성할 수 있도록 만듭니다. 학습과정에서는 원본 이미지를 입력과 목표값으로 사용하여 모델이 원본 이미지를 잘 재구성하도록 합니다.
    - autoencoder.compile(optimizer='adam', loss='mse')
    - autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True)
4. 특징 추출
- 학습된 오토인코더의 인코더 부분을 사용하여 원본 MNIST 이미지의 중간 잠재표현(특징)을 추출합니다.
    - encoded_input = Input(shape=(64,))
    - classifier_layers = Dense(32, activation='relu')(encoded_input)
    - classifier_layers = Dense(10, activation='softmax')(classifier_layers)
    - classifier = Model(encoded_input, classifier_layers)
    - x_train_encoded = encoder.predict(x_train)
    - x_test_encoded = encoder.predict(x_test)
5. 지도 학습을 위한 분류기 구축 및 학습
- 추출된 특징을 입력으로 사용하여 지도 학습을 위한 분류기를 구축합니다.
- 분류기는 여러 'Dense'레이러를 사용하여 10개의 클래스로 이미지를 분류하는 모델입니다.
- 분류기를 학습시켜 오토인코더로부터 추출된 특징을 기반으로 숫자를 분류하도록 합니다.
    - classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    - classifier.fit(x_train_encoded, y_train, epochs=50, batch_size=256, shuffle=True)

결론적으로, 이 과정은 원본 이미지 데이터로부터 중요한 특징을 비지도 학습 방식으로 추출하고,
해당 특징을 지도 학습에 활용하여 이미지 분류 작업을 수행하는 것입니다.
이러한 방식은 라벨이 없는 대량의 데이터에서 특징을 추출하고, 제한된 양의 라벨 데이터로 분류 모델을 학습시키는 경우에 유용하게 사용될 수 있습니다.
'''

# 재구성된 이미지를 시각적으로 비교하는 코드
import numpy as np
import matplotlib.pyplot as plt

# 테스트 데이터를 사용하여 이미지 재구성
decoded_imgs = autoencoder.predict(x_test)

n = 10 # 10개의 숫자 이미지를 출력
plt.figure(figsize=(20, 4))
for i in range(n):
    # 원본 이미지
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 재구성된 이미지
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# 원본 이미지와 재구성된 이미지 사이의 MSE를 계산하는 코드
from sklearn.metrics import mean_squared_error

# 테스트 데이터를 사용하여 이미지 재구성
decoded_imgs = autoencoder.predict(x_test)

# MSE 계산
mse_values = [mean_squared_error(x_test[i], decoded_imgs[i]) for i in range(len(x_test))]

# 전체 테스트 데이터셋에 대한 평균 MSE 계산
average_mse = np.mean(mse_values)

print("Reconstruction MSE:", average_mse)