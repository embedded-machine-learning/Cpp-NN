# %%
import numpy as np
import sys
import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import math
import time


include_path = '../../'

sys.path.append(os.path.join(os.path.dirname(__file__),include_path, "include"))
from write_weights import write_weight


compiler = "clang++-18"
current_pos = '/'.join(__file__.split("/")[:-1])



S_KP = 14*8
B_KP = 12*4
C_KP = 12*8
W_KP = 14*8
SUB_BATCH = 1

# S_KP = 16
# B_KP = 8
# C_KP = 16
# SUB_BATCH = 1

# S_KP = 32
# B_KP = 32
# C_KP = 32
# W_KP = 32
# SUB_BATCH = 1

# S_KP = 10
# B_KP = 10
# C_KP = 10
# SUB_BATCH = 1


# weights_dict = np.load('./model_dict_16kHz_small.npy', allow_pickle=True).item()
# weights_dict = np.load('./model_dict_16kHz.npy', allow_pickle=True).item()
# weights_dict = np.load('./model_exports/303/model_dict_[1, 1, 1, 1, 1, 1].npy', allow_pickle=True).item()
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

# 20k parmas reg and pruned
# weights_dict = np.load('./model_exports/303_pruned/model_dict_[1, 1, 1, 1, 1, 1].npy', allow_pickle=True).item()
# weights_dict = np.load('./model_exports/303_pruned/model_dict_[1, 4, 4, 12, 24, 48].npy', allow_pickle=True).item()
# weights_dict = np.load( './model_exports/303_pruned/model_dict_[1, 2, 4, 16, 32, 64].npy', allow_pickle=True).item()
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

type_of_weights = 'float'

input_scale = 1
input_offset = 0

complex_types = [np.complex64, np.complex128]

with open("weights.inc", "w") as file:
    file.write('#pragma once\n')
    file.write(f'#include "include/Matrix.hpp"\n')
    file.write(f'#include "include/types/Complex.hpp"\n')
    file.write('constexpr std::size_t B_KP = ' + str(B_KP) + ';\n')
    file.write('constexpr std::size_t C_KP = ' + str(C_KP) + ';\n')
    file.write('constexpr std::size_t S_KP = ' + str(S_KP) + ';\n')
    file.write('constexpr std::size_t W_KP = ' + str(W_KP) + ';\n')

    number_of_ssm_layers = len([1 for x in weights_dict.values() if 'A' in x.keys() and x['A'] is not None])
    file.write(f'#define NUMBER_OF_LAYERS {number_of_ssm_layers}\n')

    file.write('\n')
    step_scale = []
    for index, layer in enumerate(range(len(weights_dict))):
        if 'step_scale' in weights_dict[index].keys():
            step_scale.append(weights_dict[index]['step_scale'])
    
    step_scale_tmp = step_scale.copy()
    step_scale_tmp.append(step_scale_tmp[-1])  # Last needs to be replicated, for the summation layer
    step_scale_tmp.append(((16000-1)//step_scale_tmp[-1])*step_scale_tmp[-1]+1)  # Compute the Decoder only once, so we use the sequence length as step scale
    index_ofsets = np.zeros_like(step_scale_tmp, dtype=np.int32)
    index_ofsets[-1] = 1

    file.write(f'constexpr unsigned int step_scale[] = {{{"".join([f"{x}," for x in step_scale_tmp])[:-1]}}};\n')
    file.write(f'constexpr unsigned int step_scale_index_offsets[] = {{{"".join([f"{x}," for x in index_ofsets])[:-1]}}};\n')

    for index, layer in enumerate(weights_dict):
        print(weights_dict[index].keys())
        if 'A' in weights_dict[index].keys():
            tmp_A = weights_dict[index]['A']
            write_weight(tmp_A, f"A{index}", 'C', file)
        if 'B' in weights_dict[index].keys():
            tmp_B = weights_dict[index]['B']
            write_weight(tmp_B, f"B{index}", 'OI', file)
        if 'B_bias' in weights_dict[index].keys():
            tmp_B_bias = weights_dict[index]['B_bias']
            write_weight(tmp_B_bias, f"B{index}_bias", 'C', file)
        if 'C' in weights_dict[index].keys():
            tmp_C = weights_dict[index]['C']
            write_weight(tmp_C, f"C{index}", 'OI', file)
        if 'C_bias' in weights_dict[index].keys():
            tmp_C_bias = weights_dict[index]['C_bias']
            if tmp_C_bias.dtype in complex_types:
                print("C_bias is complex, check if it should be")
                # tmp_C_bias = tmp_C_bias.real
                tmp_C_bias = np.real(tmp_C_bias)
            write_weight(tmp_C_bias, f"C{index}_bias", 'C', file)
        if 'SkipLayer' in weights_dict[index].keys():
            SkipLayer = weights_dict[index]['SkipLayer']
            if SkipLayer is not None:
                if SkipLayer.dtype == np.complex64:
                    print("SkipLayer is complex, check if it should be")
                    SkipLayer = SkipLayer.real
                write_weight(SkipLayer, f"SkipLayer{index}", 'OI', file)
            else:
                SkipLayer = np.ones((1), dtype=np.float32)
                print("SkipLayer is None, using dummy matrix of ones")
                write_weight(SkipLayer, f"SkipLayer{index}", 'E', file)
        if 'SkipLayer' not in weights_dict[index].keys() :
            # If no SkipLayer is defined, we use a matrix of ones
            print("SkipLayer not defined, using dummy matrix of ones")
            write_weight(np.ones((1)), f"SkipLayer{index}", 'E', file)

        if 'W' in weights_dict[index].keys():
            tmp_W = weights_dict[index]['W']
            # scale W to include the last layer mean
            tmp_tmp = np.zeros((16000))
            tmp_tmp = tmp_tmp[::step_scale[-1]]
            print(f"Scaling W by {(16000+step_scale[-1]-1)//step_scale[-1]} to match the last layer means {len(tmp_tmp)} values")
            tmp_W = tmp_W/((16000+step_scale[-1]-1)//step_scale[-1])
            write_weight(tmp_W, "W", 'OI',  file)
        if 'b' in weights_dict[index].keys():
            write_weight(weights_dict[index]['b'], "b", 'C', file)
        if 'scale' in weights_dict[index].keys():
            input_scale = weights_dict[index]['scale']
        if 'offset' in weights_dict[index].keys():
            input_offset = weights_dict[index]['offset']

    # file.write(f"constexpr float input_scale = {input_scale.item()};\n")
    # file.write(f"constexpr float input_offset = {input_offset.item()};\n")

with open("network.hpp", "w") as file:
    file.write('#pragma once\n')
    file.write(f'#include "include/NeuralNetwork.hpp"\n')
    file.write('#include "weights.inc"\n')
    file.write('#include "weights_unrolled.inc"\n')
    file.write('\n')
    file.write(
        '// const auto __attribute__(( section(".data") )) network=layers::Sequence(\n')
    # file.write('const auto network=layers::Sequence(\n')
    file.write('const auto network=layers::Sequence(\n')
    string = ""
    for index,_ in enumerate(weights_dict):
        if f'A' in weights_dict[index].keys() and f'SkipLayer' in weights_dict[index].keys() and weights_dict[index]['SkipLayer'] is not None:
            # layers::S5_class_hidden(A, B, C, BiasRNN, BiasOut, Passthrough),
            # string += f'    layers::Sedge<float,Complex<float>>(A{index}, B{index}, B{index}_bias, C{index}, C{index}_bias, SkipLayer{index}, LeakyReLU<float>),\n'
            string += f'    layers::Sedge<float,Complex<float>>(A{index}, B{index}_1_{B_KP}, B{index}_bias, C{index}_1_{C_KP}, C{index}_bias, SkipLayer{index}_1_{S_KP}, LeakyReLU<float>),\n'

        elif f'A' in weights_dict[index].keys():
            # layers::S5_class_hidden(A, B, C, BiasRNN, BiasOut, Passthrough),
            # string += f'    layers::Sedge<float,Complex<float>>(A{index}, B{index}, B{index}_bias, C{index}, C{index}_bias, Matrix<float, "E", 1>{{1}}, LeakyReLU<float>),\n'
            string += f'    layers::Sedge<float,Complex<float>>(A{index}, B{index}_1_{B_KP}, B{index}_bias, C{index}_1_{C_KP}, C{index}_bias, Matrix<float, "E", 1>{{1}}, LeakyReLU<float>),\n'
    
    
    string += f'    layers::SumReduction<"S">(),\n'


    for index,_ in enumerate(weights_dict):
        if f'W' in weights_dict[index].keys():
            string += f'    layers::Linear<float>(W_1_{W_KP}, b, PassThrough<float>),\n'
    file.write(string[:-2] + '\n);\n\n')

total_number_of_weights = 0
for index,_ in enumerate(weights_dict):
    for key in weights_dict[index].keys():
        if key != 'step_scale' and weights_dict[index][key] is not None:
            total_number_of_weights += weights_dict[index][key].size

print(f"Total number of weights: {total_number_of_weights}")

# %% Compile


subprocess.run(f"{compiler} -Wall -std=c++20 -O3 -march=native -ftemplate-depth=10000 -fconstexpr-steps=100000000 -Wall -I {include_path} unroll_weights.cpp -o {current_pos}/weight_unrolling_exec.out", shell=True,executable="/bin/bash", stderr=subprocess.STDOUT)
subprocess.run(f"./weight_unrolling_exec.out", shell=True,executable="/bin/bash", stderr=subprocess.STDOUT)


#%%
if os.path.exists("CppSEdge.so"):
    os.remove("CppSEdge.so")
subprocess.run(f"{compiler} -march=native -O3 -fstack-usage -Wall -Wpedantic -Winline -shared -std=c++20 -fconstexpr-steps=100000000 -fPIC $(python3 -m pybind11 --includes) -I {include_path} {current_pos}/test_SEdge.cpp -o  {current_pos}/CppSEdge.so", shell=True, executable="/bin/bash", stderr=subprocess.STDOUT)
# subprocess.run(f"g++-12 -march=native -Wall -Wpedantic -fstack-usage -Wno-inline -O3 -fdump-tree-optimized -shared -std=c++20 --param max-unrolled-insns=1000000000 --param max-inline-insns-single=1000000000 -finline-limit=1000000000 --param inline-unit-growth=1000000 -fPIC $(python3 -m pybind11 --includes) -I {include_path} {current_pos}/test_SEdge.cpp -o  {current_pos}/CppSEdge.so", shell=True, executable="/bin/bash", stderr=subprocess.STDOUT)


# %%
import CppSEdge

inputs = np.zeros((1, 16000, 1))
# inputs = np.zeros((1,1, 1))
# inputs[0,:,0] = np.linspace(0,20*np.pi,1000)
inputs[0, 0, 0] = 0
# inputs = np.sin(inputs)

print(inputs.shape)


def infere_model(inputs):
    step_scale = 1
    assert inputs.shape == (1, 16000, 1)
    for index,_ in enumerate(weights_dict):
        if 'A' in weights_dict[index].keys():
            step_scale_new = weights_dict[index]['step_scale']
            if step_scale_new != step_scale:
                print(f"Step scale changed from {step_scale} to {step_scale_new}")
                print(f"Length has changed from {inputs.shape[1]} to {inputs.shape[1] // (step_scale_new // step_scale)} compared to {16000 // step_scale_new}")
                assert step_scale_new % step_scale == 0, "Step scale must be a multiple of the previous step scale"
                inputs = inputs[:, ::step_scale_new//step_scale, :]
                step_scale = step_scale_new

            A = weights_dict[index]['A']
            B = weights_dict[index]['B']
            B_bias = weights_dict[index]['B_bias']
            C = weights_dict[index]['C']
            C_bias = weights_dict[index]['C_bias']

            BU = inputs@B.T + B_bias.reshape(1, 1, -1)
            state = np.zeros_like(BU)
            state[:,0,:] = BU[:,0,:]         # Fraglich ob das so stimmt
            for i in range(1, inputs.shape[1]):
                state[:, i, :] = state[:, i-1, :]*A + BU[:, i, :]
            output = state@C.T + C_bias

            output = np.real(output)

            if 'SkipLayer' in weights_dict[index].keys() and weights_dict[index]['SkipLayer'] is not None:
                SkipLayer = weights_dict[index]['SkipLayer']
                output = np.where(output > 0, output, 0.01 *
                                  output) + inputs@SkipLayer.T
            else:
                output = np.where(output > 0, output, 0.01*output) + inputs

            inputs = np.real(output)

    W = weights_dict[len(weights_dict)-1]['W']
    b = weights_dict[len(weights_dict)-1]['b']
    output = inputs.mean(axis=1)@W.T + b.reshape(1, -1)
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
CppSEdge.printModelInfo()

inputs = inputs.astype(np.float32)
# S5_out,classes = CppSEdge.run_S5(inputs)
classes = CppSEdge.run(inputs)
# S5_out,classes = CppSEdge.run_S5(inputs)

test_eval, test_classes = infere_model(inputs)

# print(S5_out.dtype)
# print(test_eval.dtype)
# print(np.allclose(S5_out[:,:test_eval.shape[1],:], test_eval, atol=1e-1))
# print(S5_out[:,:test_eval.shape[1],:]-test_eval)
# time.sleep(1)

print("numpy model:", test_classes.shape, test_classes)
print("cpp model:", classes.shape, classes)
print("all close:", np.allclose(classes, test_classes, atol=1e-5))
print("diff:", (classes- test_classes)[0,:])
# print("diff:", (classes- test_classes)[0,:,0])


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

# plt.plot((classes- test_classes)[0,:,0], label="Difference")
# plt.plot(test_classes[0,:,0], label="Test Classes")
# plt.plot(classes[0,:,0], label="CPP Classes")
# plt.legend()
# plt.show()


# %%
