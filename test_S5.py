# %%
# clang++-18 -Os -Wall -Wc++20-compat -shared -std=c++17 -fconstexpr-steps=100000000 -fPIC $(python3 -m pybind11 --includes) -I $base_path $1 -o  $filename.so
import numpy as np
import sys
import os 
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from write_weight import write_weight_complex, write_weight
import math
import time


# weights_dict = np.load('./model_dict_16kHz_small.npy', allow_pickle=True).item()
# weights_dict = np.load('./model_dict_16kHz.npy', allow_pickle=True).item()
# weights_dict = np.load('./step_scale_models_run303/model_dict_[1, 1, 1, 1, 1, 1].npy', allow_pickle=True).item()
# weights_dict = np.load('./step_scale_models_run303/model_dict_[1, 1, 1, 1, 9, 63].npy', allow_pickle=True).item()
# weights_dict = np.load('./step_scale_models_run303/model_dict_[1, 1, 2, 6, 18, 54].npy', allow_pickle=True).item()
# weights_dict = np.load('./step_scale_models_run303/model_dict_[1, 2, 2, 6, 18, 54].npy', allow_pickle=True).item()
# weights_dict = np.load('./step_scale_models_run303/model_dict_[1, 4, 4, 12, 24, 48].npy', allow_pickle=True).item()

# 141k parmas 
# weights_dict = np.load('./model_exports/145/model_dict_[1, 1, 1, 1, 1, 1].npy', allow_pickle=True).item()
# weights_dict = np.load('./model_exports/145/model_dict_[1, 1, 1, 2, 4, 12].npy', allow_pickle=True).item()
# weights_dict = np.load('./model_exports/145/model_dict_[1, 1, 4, 8, 16, 32].npy', allow_pickle=True).item()
weights_dict = np.load('./model_exports/145/model_dict_[1, 2, 14, 14, 42, 42].npy', allow_pickle=True).item()

# 56k parmas 
# weights_dict = np.load('./model_exports/299/model_dict_[1, 1, 1, 1, 1, 1].npy', allow_pickle=True).item()
# weights_dict = np.load('./model_exports/299/model_dict_[1, 1, 1, 2, 2, 38].npy', allow_pickle=True).item()
# weights_dict = np.load('./model_exports/299/model_dict_[1, 2, 2, 8, 8, 48].npy', allow_pickle=True).item()
# weights_dict = np.load('./model_exports/299/model_dict_[1, 3, 3, 6, 30, 60].npy', allow_pickle=True).item()

# 20k parmas no reg
# weights_dict = np.load('./model_exports/302/model_dict_[1, 1, 1, 1, 1, 1].npy', allow_pickle=True).item()
# weights_dict = np.load('./model_exports/302/model_dict_[1, 1, 1, 1, 2, 22].npy', allow_pickle=True).item()
# weights_dict = np.load('./model_exports/302/model_dict_[1, 1, 3, 6, 6, 42].npy', allow_pickle=True).item()


# 20k parmas reg
# weights_dict = np.load('./model_exports/303/model_dict_[1, 1, 1, 1, 1, 1].npy', allow_pickle=True).item() 
# weights_dict = np.load('./model_exports/303/model_dict_[1, 1, 1, 1, 9, 63].npy', allow_pickle=True).item() 
# weights_dict = np.load('./model_exports/303/model_dict_[1, 2, 2, 6, 18, 54].npy', allow_pickle=True).item() 
# weights_dict = np.load('./model_exports/303/model_dict_[1, 2, 4, 16, 32, 64].npy', allow_pickle=True).item() 
# weights_dict = np.load('./model_exports/303/model_dict_[1, 4, 4, 16, 32, 64].npy', allow_pickle=True).item() 
# weights_dict = np.load('./model_exports/303/model_dict_[1, 4, 4, 16, 64, 64].npy', allow_pickle=True).item()

# [1, 1, 1, 1, 1, 1]
# [1, 1, 1, 1, 9, 63]
# [1, 2, 2, 6, 18, 54]
# [1, 2, 4, 16, 32, 64]
# [1, 4, 4, 16, 32, 64]
# [1, 4, 4, 16, 64, 64]
# 20k parmas reg and pruned
# weights_dict = np.load('./model_exports/303_pruned/model_dict_[1, 1, 1, 1, 1, 1].npy', allow_pickle=True).item() 
# weights_dict = np.load('./model_exports/303_pruned/model_dict_[1, 4, 4, 12, 24, 48].npy', allow_pickle=True).item() 
# weights_dict = np.load('./model_exports/303_pruned/model_dict_[1, 2, 4, 16, 32, 64].npy', allow_pickle=True).item() 
# weights_dict = np.load('./model_exports/303_pruned/model_dict_[1, 4, 4, 16, 32, 64].npy', allow_pickle=True).item() 
# weights_dict = np.load('./model_exports/303_pruned/model_dict_[1, 2, 2, 6, 18, 54].npy', allow_pickle=True).item() 


# 8k parmas no reg 304
# weights_dict = np.load('./model_exports/304/model_dict_[1, 1, 1].npy', allow_pickle=True).item() 
# weights_dict = np.load('./model_exports/304/model_dict_[1, 1, 5].npy', allow_pickle=True).item() 
# weights_dict = np.load('./model_exports/304/model_dict_[1, 2, 22].npy', allow_pickle=True).item() 
# weights_dict = np.load('./model_exports/304/model_dict_[1, 4, 28].npy', allow_pickle=True).item()

# 8k parmas reg 305
# weights_dict = np.load('./model_exports/305/model_dict_[1, 1, 1].npy', allow_pickle=True).item() 
# weights_dict = np.load('./model_exports/305/model_dict_[1, 2, 6].npy', allow_pickle=True).item() 
# weights_dict = np.load('./model_exports/305/model_dict_[1, 5, 20].npy', allow_pickle=True).item() 
# weights_dict = np.load('./model_exports/305/model_dict_[1, 9, 54].npy', allow_pickle=True).item() 
# weights_dict = np.load('./model_exports/305/model_dict_[1, 11, 55].npy', allow_pickle=True).item() 
# weights_dict = np.load('./model_exports/305/model_dict_[2, 8, 64].npy', allow_pickle=True).item() 
# weights_dict = np.load('./model_exports/305/model_dict_[3, 12, 48].npy', allow_pickle=True).item()

# state = A0*state + B0*input + B0_bias
# output = C0*state + C0_bias 
# out2 = Activation(output) + input

# A0 = np.array([[0.9,0],[0.99*math.cos(0.2*math.pi),0.99*math.sin(0.2*math.pi)]], dtype=np.float32).reshape(2,2)
# B0 = np.array([1,0.1,1,0], dtype=np.float32).reshape(2,1,2)
# B0_bias = np.array([0,0,0,0], dtype=np.float32).reshape(2,2)
# C0 = np.array([1,0,0,1], dtype=np.float32).reshape(1,2,2)
# # C0_bias = np.array([0,0], dtype=np.float32).reshape(1,2)

type_of_weights = 'float'

input_scale = 1
input_offset = 0

with open("weights.hpp", "w") as file:
    file.write('#pragma once\n')
    file.write('#include "./include/Matrix.hpp"\n')
    # file.write('#include "./include/helpers/Complex.hpp"\n')
    step_scale = []
    for index in range(len(weights_dict)):
        if 'step_scale' in weights_dict[index].keys():
            step_scale.append(weights_dict[index]['step_scale'])
    
    file.write(f'int step_scale[] = {{{"".join([f"{x}," for x in step_scale])[:-1]}}};\n')

    for index in range(len(weights_dict)):
        print(weights_dict[index].keys())
        if f'A' in weights_dict[index].keys():
            tmp_A = weights_dict[index][f'A']
            if tmp_A.dtype == np.complex64:
                tmp_A = np.stack((tmp_A.real, tmp_A.imag), axis=-1)
            write_weight_complex(tmp_A, type_of_weights, 'C', f"A{index}", file)
        if f'B' in weights_dict[index].keys():
            tmp_B = weights_dict[index][f'B']
            if tmp_B.dtype == np.complex64:
                tmp_B = np.stack((tmp_B.real, tmp_B.imag), axis=-1)
            write_weight_complex(tmp_B, type_of_weights, 'OI', f"B{index}", file)
        if f'B_bias' in weights_dict[index].keys():
            tmp_B_bias = weights_dict[index][f'B_bias']
            if tmp_B_bias.dtype == np.complex64:
                tmp_B_bias = np.stack((tmp_B_bias.real, tmp_B_bias.imag), axis=-1)
            write_weight_complex(tmp_B_bias, type_of_weights, 'C', f"B{index}_bias", file)
        if f'C' in weights_dict[index].keys():
            tmp_C = weights_dict[index][f'C']
            if tmp_C.dtype == np.complex64:
                tmp_C = np.stack((tmp_C.real, tmp_C.imag), axis=-1)
            write_weight_complex(tmp_C, type_of_weights, 'OI', f"C{index}", file)
        if f'C_bias' in weights_dict[index].keys():
            # write_weight_complex(weights_dict[index][f'C{index}_bias'], type_of_weights, 'C', f"C{index}_bias", file)
            tmp_C_bias = weights_dict[index][f'C_bias']
            if tmp_C_bias.dtype == np.complex64:
                print("C_bias is complex, check if it should be")
                tmp_C_bias = tmp_C_bias.real
            write_weight(tmp_C_bias, type_of_weights, 'C', f"C{index}_bias", file)
        if f'SkipLayer' in weights_dict[index].keys():
            SkipLayer = weights_dict[index][f'SkipLayer']
            if SkipLayer is not None:
                if SkipLayer.dtype == np.complex64:
                    print("SkipLayer is complex, check if it should be")
                    SkipLayer = SkipLayer.real
                write_weight(SkipLayer, type_of_weights, 'OI', f"SkipLayer{index}_weights", file)
        if 'W' in weights_dict[index].keys():
            tmp_W = weights_dict[index]['W']
            # scale W to include the last layer mean
            tmp_W = tmp_W/((16000+step_scale[-1]-1)//step_scale[-1])
            write_weight(tmp_W, type_of_weights, 'OI', "W", file)
        if 'b' in weights_dict[index].keys():
            write_weight(weights_dict[index]['b'], type_of_weights, 'C', "b", file)
        if 'scale' in weights_dict[index].keys():
            input_scale = weights_dict[index]['scale']
        if 'offset' in weights_dict[index].keys():
            input_offset = weights_dict[index]['offset']
    
    # file.write(f"constexpr float input_scale = {input_scale.item()};\n")
    # file.write(f"constexpr float input_offset = {input_offset.item()};\n")

with open("network.hpp", "w") as file:
    file.write('#pragma once\n')
    file.write('#include "weights.hpp"\n')
    # file.write('#include "weights_unrolled_1.hpp"\n')
    file.write('#include "./include/NeuralNetwork.hpp"\n')
    file.write('\n')
    file.write('// const auto __attribute__(( section(".data") )) Seq=layers::Sequential(\n')
    file.write('const auto Seq=layers::Sequential(\n')
    string = ""
    for index in range(len(weights_dict)):
        if f'A' in weights_dict[index].keys() and f'SkipLayer' in weights_dict[index].keys() and weights_dict[index]['SkipLayer'] is not None:
            # layers::S5_class_hidden(A, B, C, BiasRNN, BiasOut, Passthrough),
            string += f'    layers::S5(A{index}, B{index}, C{index}, B{index}_bias, C{index}_bias, SkipLayer{index}_weights, LeakyReLU),\n'
        elif f'A' in weights_dict[index].keys() :
            # layers::S5_class_hidden(A, B, C, BiasRNN, BiasOut, Passthrough),
            string += f'    layers::S5(A{index}, B{index}, C{index}, B{index}_bias, C{index}_bias, LeakyReLU),\n'
    
    file.write(string[:-2] + '\n);\n\n')

    file.write('// const auto __attribute__(( section(".data") )) Decoder=layers::Sequential(\n')
    file.write('const auto Decoder=layers::Sequential(\n')
    string = ""
    for index in range(len(weights_dict)):
        if f'W' in weights_dict[index].keys():
            string += f'    layers::Linear<float>(W, b, Passthrough),\n'
    file.write(string[:-2] + '\n);\n\n')

total_number_of_weights = 0 
for index in range(len(weights_dict)):
    for key in weights_dict[index].keys():
        if key != 'step_scale' and weights_dict[index][key] is not None:
            total_number_of_weights += weights_dict[index][key].size

print(f"Total number of weights: {total_number_of_weights}")

#%% Compile

if os.path.exists("CppS5.so"):
    os.remove("CppS5.so")
subprocess.run("clang++-18 -march=native -Ofast -Wpedantic -Wc++20-compat -shared -std=c++17 -fconstexpr-steps=100000000 -fPIC $(python3 -m pybind11 --includes) -I ./ python_S5.cpp -o  CppS5.so", shell=True,executable="/bin/bash", stderr=subprocess.STDOUT)
# subprocess.run("g++ -Wall -std=c++17 -fstack-usage -fconstexpr-ops-limit=100000000 -march=native -Ofast -Wc++20-compat -shared -fPIC $(python3 -m pybind11 --includes) -I ./ python_S5.cpp -o  CppS5.so", shell=True,executable="/bin/bash", stderr=subprocess.STDOUT)

import CppS5


# %%
inputs = np.zeros((1,16000, 1))
# inputs = np.zeros((1,1, 1))
# inputs[0,:,0] = np.linspace(0,20*np.pi,1000)
inputs[0,0,0] = 0
# inputs = np.sin(inputs)

print(inputs.shape)

def infere_model(inputs):
    step_scale = 1
    assert inputs.shape == (1,16000,1)
    for index in range(len(weights_dict)):
        if 'A' in weights_dict[index].keys():
            step_scale_new = weights_dict[index]['step_scale']
            if step_scale_new != step_scale:
                print(f"Step scale changed from {step_scale} to {step_scale_new}")
                inputs = inputs[:,::step_scale_new//step_scale,:]
                step_scale = step_scale_new

            A = weights_dict[index]['A']
            B = weights_dict[index]['B']
            B_bias = weights_dict[index]['B_bias']
            C = weights_dict[index]['C']
            C_bias = weights_dict[index]['C_bias']

            BU = inputs@B.T + B_bias.reshape(1,1,-1)
            state = np.zeros_like(BU)
            state[:,0,:] = BU[:,0,:]         # Fraglich ob das so stimmt
            for i in range(1,inputs.shape[1]):
                state[:,i,:] = state[:,i-1,:]*A + BU[:,i,:]
            output = state@C.T + C_bias

            output = np.real(output)

            if 'SkipLayer' in weights_dict[index].keys() and weights_dict[index]['SkipLayer'] is not None:
                SkipLayer = weights_dict[index]['SkipLayer']
                output = np.where(output > 0, output, 0.01*output) + inputs@SkipLayer.T
            else:
                output = np.where(output > 0, output, 0.01*output) + inputs

            inputs = np.real(output)
    W = weights_dict[len(weights_dict)-1]['W']
    b = weights_dict[len(weights_dict)-1]['b']
    output = inputs.mean(axis=1)@W.T + b.reshape(1,-1)
    return inputs, output
            




# %%
# inputs = np.load("./class0.npy")
# pred = np.load("./model_prediction.npy")
# numpy_res = np.array([ -64.62145  ,  -59.986797 , -103.47521  ,  -37.39458  , 
#         -76.22889  ,  -92.937195 ,    4.1469936,  -66.35126  , 
#         -60.137173 ,  -73.12128  ,  -40.77534  ,  -64.70467  , 
#        -101.21764  ,  -99.69339  ,  -72.35792  ,  -63.904472 , 
#        -112.23827  ,  -87.2785   ,  -95.36414  ,  -68.5412   , 
#         -62.054855 ,  -35.20459  ,  -55.266315 ,  -57.185253 , 
#         -73.86204  ,  -33.274426 ,  -76.66431  , -114.70926  , 
#         -92.431366 ,  -46.69651  ,  -52.500866 ,  -96.02911  , 
#         -60.04177  ,  -28.922403 , -104.02467  ])
# print(inputs.shape)
inputs = inputs.astype(np.float32)
# S5_out,classes = CppS5.run_S5(inputs)
classes = CppS5.run_S5(inputs)
# S5_out,classes = CppS5.run_S5(inputs)

test_eval, test_classes = infere_model(inputs)

test_eval = test_eval.repeat(step_scale[-1], axis=1)
if test_eval.shape[1] > 16000:
    test_eval = test_eval[:,:16000,:]

# print(S5_out.dtype)
# print(test_eval.dtype)
# print(np.allclose(S5_out[:,:test_eval.shape[1],:], test_eval, atol=1e-1))
# print(S5_out[:,:test_eval.shape[1],:]-test_eval)
# time.sleep(1)

print(test_classes)
print(classes)
print(np.allclose(classes, test_classes, atol=1e-5))
# # print(S5_out)
# fig = plt.figure()
# plt.plot(inputs[0,:,0])
# tmp = S5_out[:,:test_eval.shape[1],:]-test_eval
# for i in range(S5_out.shape[2]):
#     plt.plot(tmp[0,:,i])
# # plt.plot(S5_out[0,:,0])
# # plt.yscale('log')
# plt.show()

# for i in range(3):
#     plt.plot(S5_out[0,:,i])
#     plt.plot(test_eval[0,:,i])
# # plt.plot(S5_out[0,:,0])
# # plt.yscale('log')
# plt.show()

# print(S5_out)


# print(pred)
# for i in range(1,35):
# plt.plot(outputs[:])
# # plt.plot(outputs[:,1], label="S5 1")
# # plt.plot(inputs[:,0], label="Input")
# plt.show()



# %%
