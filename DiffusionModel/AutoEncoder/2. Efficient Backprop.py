# 신경망 구조그림
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize = (10, 6))

# 신경망 구조 그리기
layers = [3, 4, 4, 2]
layer_names = ['Input Layer', 'Hidden Layer 1', 'Hidden Layer 2', 'Output Layer']
x_data = np.linspace(0, len(layers)-1, len(layers))

for i, (layer_size, x) in enumerate(zip(layers, x_data)) :
    y_data = np.linspace(0, layer_size-1, layer_size)
    for y in y_data : 
        circle = plt.Circle((x, y), 0.2, color='blue' if i == 0 else ('red' if i == len(layers)-1 else 'green'))
        ax.add_artist(circle)
        if i < len(layers) - 1:  # Draw arrows to next layer : i가 마지막 layer가 아닐 때
            y_next_data = np.linspace(0, layers[i+1]-1, layers[i+1])
            for y_next in y_next_data:
                plt.arrow(x+0.2, y, 0.6, y_next-y, head_width=0.1, head_length=0.1, fc='gray', ec='gray')

    # Label layers : x축에 layer 이름을 표시
    plt.text(x, y_data[-1] + 1, layer_names[i], ha='center')

# Add title and remove axis : 그래프 제목과 축을 제거
plt.title("Feedforward Neural Network Architecture")
ax.axis('off')
plt.show()


# 파란색원 : 입력 레이어의 뉴런
# 녹색 원 : 은닉 레이어의 뉴런
# 빨간색 원 : 출력 레이어의 뉴런
# 회색 화살표 : 뉴런의 연결 각 연결은 가중치를 나타냄 


fig, ax = plt.subplots(figsize=(10,6))


for i, (layer_size, x) in enumerate(zip(layers, x_data)):
    y_data = np.linspace(0, layer_size-1, layer_size)
    for y in y_data:
        circle = plt.Circle((x, y), 0.2, color='blue' if i == 0 else ('red' if i == len(layers)-1 else 'green'))
        ax.add_artist(circle)
        if i > 0:
            y_prev_data = np.linspace(0, layers[i-1]-1, layers[i-1])
            for y_prev in y_prev_data:
                plt.arrow(x-0.2, y, -0.6, y_prev-y, head_width=0.1, head_length=0.1, fc='purple', ec='purple')

    
    plt.text(x, y_data[-1] + 1, layer_names[i], ha='center')

plt.title("Backpropagation in Neural Network")
ax.axis('off')
plt.show()

# 보라색 화살표 : 역전파 알고리즘에서 사용되는 연결
'''
Backpropagation의 주요 아이디어는 네트워크의 출력에서 발생하는 오류를 역방향으로 전파하고, 이 오류를 사용하여 각 연결의 가중치를 조정하는 것입니다. 이 과정은 연쇄 법칙을 사용하여 각 뉴런에 대한 오류의 편미분을 계산합니다.

이러한 방식으로 신경망은 예상 출력과 실제 출력 간의 차이를 최소화하려고 학습합니다. 각 연결의 가중치는 오류를 줄이기 위해 반복적으로 조정됩니다. 이 과정은 데이터셋의 모든 샘플에 대해 여러 번 반복됩니다.

이 논문은 이러한 방식으로 신경망이 복잡한 함수와 패턴을 학습할 수 있음을 보여주었으며, 이로 인해 딥러닝 연구의 주요 발전이 이루어졌습니다.
'''