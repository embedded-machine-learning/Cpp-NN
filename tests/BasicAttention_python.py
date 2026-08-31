import numpy as np

import subprocess
subprocess.run('clang++-18 -march=native -Ofast -fstack-usage -Wc++20-compat -shared -std=c++20 -fconstexpr-steps=100000000 -fPIC $(python3 -m pybind11 --includes) -I ./ BasicAttention_python.cpp -o  BasicAttention.so', shell=True, executable="/bin/bash", stderr=subprocess.STDOUT)

import BasicAttention

import torch.nn

# help(BasicAttention)

dimension_k = 32
dimension_v = 48
N = 700
d_model = 64     # C dimenion

input_data = (np.random.randn(N, d_model) * 0.1).astype(np.float32) # NC dimenion
QWeightMatrix = (np.random.randn(d_model,dimension_k) * 0.1).astype(np.float32) # CK dimenion
KWeightMatrix = (np.random.randn(d_model,dimension_k) * 0.1).astype(np.float32) # CK dimenion
VWeightMatrix = (np.random.randn(d_model,dimension_v) * 0.1).astype(np.float32) # CV dimenion

BasicAttention.set_QWeight(QWeightMatrix)
BasicAttention.set_KWeight(KWeightMatrix)
BasicAttention.set_VWeight(VWeightMatrix)

result = BasicAttention.forward(input_data)


print("Input Data:")
print(input_data)
print("Result:")
print(result)

buffer_bytes, minimal_bytes = BasicAttention.get_memory_info()
print(f"Buffer: {buffer_bytes} bytes, Minimal: {minimal_bytes} bytes")

def torch_basic_attention(X, W_Q, W_K, W_V, dimension_k):
    X = torch.from_numpy(X)
    W_Q = torch.from_numpy(W_Q)
    W_K = torch.from_numpy(W_K)
    W_V = torch.from_numpy(W_V)

    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    Score = Q @ K.T / np.sqrt(dimension_k)
    attention_weights = torch.softmax(Score, dim=1)
    output = attention_weights @ V
    return output.numpy()



result_pytorch = torch_basic_attention(input_data, QWeightMatrix, KWeightMatrix, VWeightMatrix, dimension_k)
print("Result with PyTorch weights:")
print(result_pytorch)

print(np.max(np.abs(result - result_pytorch)))
print(np.max(np.abs(result_pytorch)))
print(np.max(np.abs(result - result_pytorch))/np.max(np.abs(result_pytorch)))


print("Match:", np.allclose(result, result_pytorch, rtol=0, atol=1e-6))