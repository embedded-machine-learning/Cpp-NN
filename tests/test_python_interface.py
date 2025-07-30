import numpy as np

import subprocess
subprocess.run('clang++-18 -march=native -O0 -Wpedantic -shared -std=c++20 -fconstexpr-steps=100000000 -fPIC $(python3 -m pybind11 --includes) -I ./ test_python_interface.cpp -o  TestPython.so', shell=True, executable="/bin/bash", stderr=subprocess.STDOUT)

import TestPython

# help(TestPython)

a = np.random.randn(10,20).astype(dtype=np.float32)

print("Print from C++")
b = TestPython.print(a)

assert (a==b).all(), "The result from the C++ Python Interface does not match the input matrix."
print("The result from the C++ Python Interface matches the input matrix. (assert passed)")


input = np.random.randn(10,20).astype(dtype=np.float32)
weights = np.random.randn(20,30).astype(dtype=np.float32)
bias = np.random.randn(30).astype(dtype=np.float32)

print("Performing C++ Matrix Multiplication")
result = TestPython.matrix_mult(input, weights, bias)
# print("Result from C++ Matrix Multiplication")

#compute the expected result using numpy
np_res = input @ weights + bias[None, :]

assert np.isclose(result,np_res, 1e-6, 1e-6).all(), "The result from the C++ Matrix Multiplication does not match the expected result."
print("The result from the C++ Matrix Multiplication matches the expected result. (assert passed)")

print("Performing C++ Matrix Multiplication with Split Weights")
result_split = TestPython.matrix_mult_split(input, weights, bias)
# print("Result from C++ Matrix Multiplication with Split Weights")

#compute the expected result using numpy
np_res_split = input @ weights + bias[None, :]

# print(np.round(np_res_split- result_split,2))
assert np.isclose(result_split,np_res_split, 1e-6, 1e-6).all(), "The result from the C++ Matrix Multiplication with Split Weights does not match the expected result."
print("The result from the C++ Matrix Multiplication with Split Weights matches the expected result. (assert passed)")
