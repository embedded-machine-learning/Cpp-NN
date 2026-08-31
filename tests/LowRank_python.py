import numpy as np

import subprocess
subprocess.run('clang++-18 -march=native -Ofast -Wc++20-compat -fstack-usage -shared -std=c++20 -fconstexpr-steps=100000000 -fPIC $(python3 -m pybind11 --includes) -I ./ LowRank_python.cpp -o  LowRank.so', shell=True, executable="/bin/bash", stderr=subprocess.STDOUT)

import LowRank

import torch.nn

# help(LowRank)

dimension_k = 32
dimension_v = 48
N = 700
d_model = 64     # C dimenion
r = 10
input_data  = (np.random.randn(N,d_model) * 0.1).astype(np.float32) # NC dimenion
W_Q         = (np.random.randn(d_model, dimension_k) * 0.1).astype(np.float32) 
W_K         = (np.random.randn(d_model, dimension_k) * 0.1).astype(np.float32) 
W_V         = (np.random.randn(d_model, dimension_v) * 0.1).astype(np.float32) 

U_Q, S_Q, Vt_Q = np.linalg.svd(W_Q)
U_K, S_K, Vt_K = np.linalg.svd(W_K)
U_V, S_V, Vt_V = np.linalg.svd(W_V)

QWeightMatrix = U_Q[:, :r] @ np.diag(S_Q[:r])
KWeightMatrix = U_K[:, :r] @ np.diag(S_K[:r])
VWeightMatrix = U_V[:, :r] @ np.diag(S_V[:r])

VTrQWeightMatrix =Vt_Q[:r, :]
VTrKWeightMatrix =Vt_K[:r, :]
VTrVWeightMatrix =Vt_V[:r, :]

LowRank.set_QWeight(QWeightMatrix)
LowRank.set_VTrQWeight(VTrQWeightMatrix)

LowRank.set_KWeight(KWeightMatrix)
LowRank.set_VTrKWeight(VTrKWeightMatrix)

LowRank.set_VWeight(VWeightMatrix)
LowRank.set_VTrVWeight(VTrVWeightMatrix)

result = LowRank.forward(input_data)


print("Input Data:")
print(input_data)
print("Result:")
print(result)

buffer_bytes, minimal_bytes = LowRank.get_memory_info()
print(f"Buffer: {buffer_bytes} bytes, Minimal: {minimal_bytes} bytes")

def torch_lowRank(X, A_Q, B_Q, A_K, B_K, A_V, B_V, dimension_k):
    X = torch.from_numpy(X)
    A_Q = torch.from_numpy(A_Q)
    B_Q = torch.from_numpy(B_Q)
    A_K = torch.from_numpy(A_K)
    B_K = torch.from_numpy(B_K)
    A_V = torch.from_numpy(A_V)
    B_V = torch.from_numpy(B_V)

    QL = X @ A_Q
    Q = QL @ B_Q
    KL = X @ A_K
    K = KL @ B_K
    VL = X @ A_V
    V = VL @ B_V
    Score = (Q @ K.T) / torch.sqrt(torch.tensor(dimension_k, dtype=torch.float32))
    attention_weights = torch.softmax(Score, dim=1)
    output = attention_weights @ V
    return output.numpy()


result_pytorch = torch_lowRank(input_data, QWeightMatrix, VTrQWeightMatrix, KWeightMatrix, VTrKWeightMatrix, VWeightMatrix, VTrVWeightMatrix, dimension_k)
print("Result with PyTorch weights (SVD-reduced):")
print(result_pytorch)


print(np.max(np.abs(result - result_pytorch)))
print(np.max(np.abs(result_pytorch)))
print(np.max(np.abs(result - result_pytorch))/np.max(np.abs(result_pytorch)))

print("Match:", np.allclose(result, result_pytorch, rtol=0, atol=1e-7))