import numpy as np

import subprocess
subprocess.run('clang++-18 -march=native -Ofast -fstack-usage -Wc++20-compat -shared -std=c++20 -fconstexpr-steps=100000000 -fPIC $(python3 -m pybind11 --includes) -I ./ LinearAttention_python.cpp -o  LinearAttention.so', shell=True, executable="/bin/bash", stderr=subprocess.STDOUT)

import LinearAttention

import torch.nn

# help(LinearAttention)

dimension_k = 32
dimension_v = 48
N = 700
d_model = 64     # C dimenion

input_data      = (np.random.randn(N,d_model) * 0.1).astype(np.float32) # NC dimenion
QWeightMatrix   = (np.random.randn(d_model,dimension_k) * 0.1).astype(np.float32) # CK dimenion
KWeightMatrix   = (np.random.randn(d_model,dimension_k) * 0.1).astype(np.float32) # CK dimenion
VWeightMatrix   = (np.random.randn(d_model,dimension_v) * 0.1).astype(np.float32) # CV dimenion

LinearAttention.set_QWeight(QWeightMatrix)
LinearAttention.set_KWeight(KWeightMatrix)
LinearAttention.set_VWeight(VWeightMatrix)

result = LinearAttention.forward(input_data)


print("Input Data:")
print(input_data)
print("Result:")
print(result)

buffer_bytes, minimal_bytes = LinearAttention.get_memory_info()
print(f"Buffer: {buffer_bytes} bytes, Minimal: {minimal_bytes} bytes")

def torch_linear_attention(X, W_Q, W_K, W_V):
    X = torch.from_numpy(X)
    W_Q = torch.from_numpy(W_Q)
    W_K = torch.from_numpy(W_K)
    W_V = torch.from_numpy(W_V)

    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    phi_Q = torch.nn.functional.elu(Q) + 1
    phi_K = torch.nn.functional.elu(K) + 1
    phi_KT_V = phi_K.T @ V
    SUM_K = phi_K.sum(dim=0)                         
    denominator = phi_Q @ SUM_K                     
    numerator = phi_Q @ phi_KT_V
    output = numerator / denominator.unsqueeze(1)
    return output.numpy()


     

result_pytorch = torch_linear_attention(input_data, QWeightMatrix, KWeightMatrix, VWeightMatrix)

print("Result with PyTorch weights:")
print(result_pytorch)

print(np.max(np.abs(result - result_pytorch)))
print(np.max(np.abs(result_pytorch)))
print(np.max(np.abs(result - result_pytorch))/np.max(np.abs(result_pytorch)))

print("Match:", np.allclose(result, result_pytorch, rtol=0, atol=1e-6))