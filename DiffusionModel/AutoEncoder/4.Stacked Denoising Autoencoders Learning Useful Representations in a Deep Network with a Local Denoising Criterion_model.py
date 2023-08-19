import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# 1. Denosing AutoEncoder 정의
def denoising_autoencoder(input_dim, encoding_dim, noise_factor=0.5):
    # 인풋레이어에 노이즈 추가
    input_img = Input(shape=(input_dim,))
    noisy_input = tf.keras.layers.GaussianNoise(stddev=noise_factor)(input_img)

    # Encoding layer
    encoded = Dense(encoding_dim, activation='relu')(noisy_input)

    # Decoding layer
    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    return Model(input_img, decoded)

# 2. Denosing AutoEncoder 쌓기
input_dim = 784  # For example, using MNIST data 초기 입력차원
encoding_dims = [512, 256, 128] # 인코딩 차원 = 리스트의 길이는 층의 개수를 의미합니다. 만약 [512, 256, 128, 64]라면 4개의 층을 가진 Denoising AutoEncoder가 됩니다.


previous_ae_output = Input(shape=(input_dim,))
model_input = previous_ae_output
models = []

# 층별로 Denosing AutoEncoder 쌓기
for enc_dim in encoding_dims:
    dae = denoising_autoencoder(input_dim, enc_dim)
    models.append(dae)
    model_input = dae(model_input)
    input_dim = enc_dim  # for the next layer

stacked_dae = Model(previous_ae_output, model_input)

# 3. Compile and train the model (assuming `x_train` is your training data)
stacked_dae.compile(optimizer='adam', loss='binary_crossentropy')
stacked_dae.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True)


'''
학습과정
1. 원본 이미지(노이즈가 없는 이미지)를 인풋으로 받는다.
- input_img = Input(shape=(input_dim,))
2. 원본 이미지에 노이즈를 추가하여 노이즈가 있는 이미지를 생성한다.
- noisy_input = tf.keras.layers.GaussianNoise(stddev=noise_factor)(input_img)
3. 노이즈가 생긴 이미지를 Autoencoder의 입력으로 사용함
- encoded = Dense(encoding_dim, activation='relu')(noisy_input)
4. Autoencoder는 내부적으로 이 이미지를 압축된 형태로 인코딩합니다. (이 때, 중요한 특징들만 포착하려고 시도합니다.)
- decoded = Dense(input_dim, activation='sigmoid')(encoded)
5. 인코딩된 압축된 형태를 다시 이미지의 원본 크기로 디코딩합니다.
- return Model(input_img, decoded)
6. 디코딩된 이미지(복원된 이미지)와 원본 이미지(노이즈 없는)사이의 차이를 최소화 하도록 모델을 학습시킵니다.
- stacked_dae.compile(optimizer='adam', loss='binary_crossentropy')
- stacked_dae.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True)

이 과정을 반복하면서 노이즈가 있는 이미지를 노이즈가 없는 이미지로 복원하는 방법을 학습합니다.
Denoising Autoencoder의 학습 목표는 노이즈가 있는 입력에서 원본 이미지를 최대한 정확하게 복원하는 것입니다. 따라서, 네트워크의 출력(복원된 이미지)과 실제 원본 이미지(노이즈가 없는) 사이의 차이를 기반으로 손실 함수를 계산합니다.
손실 함수는 주로 Mean Squared Error (MSE)와 같은 재구성 손실을 사용합니다.

즉, 학습하는 동안 Denoising Autoencoder는 다음 두 단계를 반복적으로 수행
1. 노이즈가 있는 이미지를 인풋으로 입력 받아 원본 이미지를 재구성하려고 시도
2. 재구성된 이미지와 원본 이미지 사이의 차이를 계산하여 손실 함수를 계산하고 이를 최소화하도록 모델의 가중치를 업데이트
'''