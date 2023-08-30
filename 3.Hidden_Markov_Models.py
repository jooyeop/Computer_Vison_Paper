import numpy as np

# 상태 : 0 = 얼음, 1 = 물

states = [0, 1]

# 관측 : 0 = 얼음 - 관측, 1 = 물 - 관측
observations = [0, 1, 0]

# 상태전이 확률 행렬
# 예시, transition_prob[0, 1]은 상태가 얼음에서 물로 전이할 확률
transition_prob = np.array([
    [0.7, 0.3], # 얼음 -> {얼음, 물}
    [0.4, 0.6]  # 물 -> {얼음, 물}
])

# 관측 확률 행렬
# emission_prob[0, 1]은 상태가 얼음일 때 물 - 관측일 확률
emission_prob = np.array([
    [0.8, 0.2], # 얼음 -> {얼음 - 관측, 물 - 관측}
    [0.2, 0.8]  # 물 -> {얼음 - 관측, 물 - 관측}
])

# Viterbi 알고리즘
def viterbi(obs, states, start_p, trans_p, emit_p) :
    V = np.zeros((len(obs), len(states)))
    path = {}

    # 초기화
    V[0, :] = start_p * emit_p[:, obs[0]]

    for t in range(1, len(obs)) :
        new_path = {}
        for y in states:
            (prob, state) = max((V[t-1, y0] * trans_p[y0, y] * emit_p[y, obs[t]], y0) for y0 in states)
            V[t, y] = prob
            new_path[y] = path[state] + [y] if t != 1 else [state, y]

        path = new_path

    (prob, state) = max((V[len(obs) - 1, y], y) for y in states)
    return (prob, path[state])


# 초기 상태 확률(얼음 : 0.5, 물 : 0.5)
start_probability = np.array([0.5, 0.5])

# Viterbi 알고리즘 실행
(prob, seq) = viterbi(observations, states, start_probability, transition_prob, emission_prob)
state_map = {0 : '얼음', 1 : '물'}
state_seq = [state_map[v] for v in seq]

print("확률 : ", prob)

import os
os.system('python 3.Hidden_Markov_Models.py')