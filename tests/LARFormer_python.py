import numpy as np

import subprocess
subprocess.run('clang++-18 -march=native -Ofast -fstack-usage -Wc++20-compat -shared -std=c++20 -fconstexpr-steps=100000000 -fPIC $(python3 -m pybind11 --includes) -I ./ LARFormer_python.cpp -o  LARFormer.so', shell=True, executable="/bin/bash", stderr=subprocess.STDOUT)

import LARFormer

import torch.nn

# help(LARFormer)

N = 700
d_model = 64     # C dimenion

def torch_linear_weight(in_features, out_features):
    layer = torch.nn.Linear(in_features, out_features, bias=False)
    return layer.weight.detach().numpy().T.astype(np.float32)


input_data      = (np.random.randn(N,d_model) * 0.1).astype(np.float32)  # NC dimenion
IWeightMatrix   = (np.random.randn(d_model,1) * 0.1).astype(np.float32)  # CK dimenion
KWeightMatrix   = (np.random.randn(d_model,d_model) * 0.1).astype(np.float32)  # CK dimenion
VWeightMatrix   = (np.random.randn(d_model,d_model) * 0.1).astype(np.float32)  # CV dimenion
OWeightMatrix   = (np.random.randn(d_model,d_model) * 0.1).astype(np.float32)  # CV dimenion


LARFormer.set_IWeight(IWeightMatrix)
LARFormer.set_KWeight(KWeightMatrix)
LARFormer.set_VWeight(VWeightMatrix)
LARFormer.set_OWeight(OWeightMatrix)


result = LARFormer.forward(input_data)


print("Input Data:")
print(input_data)
print("Result:")
print(result)

buffer_bytes, minimal_bytes = LARFormer.get_memory_info()
print(f"Buffer: {buffer_bytes} bytes, Minimal: {minimal_bytes} bytes")

def torch_LARFormer(X, W_I, W_K, W_V, W_O):
    X = torch.from_numpy(X)
    W_I = torch.from_numpy(W_I)
    W_K = torch.from_numpy(W_K)
    W_V = torch.from_numpy(W_V)
    W_O = torch.from_numpy(W_O)

    cs = torch.relu(X @ W_I)
    ck = X @ W_K
    cv = (cs * ck).sum(dim=0, keepdim=True)
    xv = torch.relu(X @ W_V)
    cv_xv = cv * xv
    result = cv_xv @ W_O
    rms = torch.sqrt((result**2).mean(dim=1, keepdim=True) + 1e-8)

    output = result / rms
    return output.numpy()




result_pytorch = torch_LARFormer(input_data, IWeightMatrix, KWeightMatrix, VWeightMatrix, OWeightMatrix)
print("Result with PyTorch weights:")
print(result_pytorch)


print(np.max(np.abs(result - result_pytorch)))
print(np.max(np.abs(result_pytorch)))
print(np.max(np.abs(result - result_pytorch))/np.max(np.abs(result_pytorch)))

print("Match:", np.allclose(result, result_pytorch, rtol=0, atol=1e-5))