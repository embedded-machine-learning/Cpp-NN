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
model_path = '../../'

sys.path.append(os.path.join(os.path.dirname(__file__),include_path, "include"))
from write_weights import write_weight


current_pos = '/'.join(__file__.split("/")[:-1])


# AVX2 COnfiguration
S_KP = 14*8
B_KP = 12*4
C_KP = 12*8
W_KP = 14*8
SUB_BATCH = 1

# ARM Cortex M4 Configuration, weights should be unrolled, enable UNROLL_UPTO in unroll_weights.cpp
# S_KP = 32
# B_KP = 32
# C_KP = 32
# W_KP = 32
# SUB_BATCH = 1

compiler = "clang++-18"
clang_flags = [
    '-Os',
    '-march=native',
    '-fstack-usage',
    '-std=c++20',
    '-fconstexpr-steps=100000000',
    '-Wextra',
    '-Winline',
    '-Wpedantic',
]


# compiler = "g++-12"
# clang_flags = [
#     '-O3',
#     '-march=native',
#     '-fstack-usage',
#     '-std=c++20',
#     '-Wextra',
#     '-Winline',
#     '-Wpedantic',
# ]


# 141k parmas
# weights_dict = np.load(f'{model_path}/model_exports/145/model_dict_[1, 1, 1, 1, 1, 1].npy', allow_pickle=True).item()
# weights_dict = np.load(f'{model_path}/model_exports/145/model_dict_[1, 1, 1, 2, 4, 12].npy', allow_pickle=True).item()
# weights_dict = np.load(f'{model_path}/model_exports/145/model_dict_[1, 1, 4, 8, 16, 32].npy', allow_pickle=True).item()
# weights_dict = np.load(f'{model_path}/model_exports/145/model_dict_[1, 2, 14, 14, 42, 42].npy', allow_pickle=True).item()

# 56k parmas
# weights_dict = np.load(f'{model_path}/model_exports/299/model_dict_[1, 1, 1, 1, 1, 1].npy', allow_pickle=True).item()
# weights_dict = np.load(f'{model_path}/model_exports/299/model_dict_[1, 1, 1, 2, 2, 38].npy', allow_pickle=True).item()
# weights_dict = np.load(f'{model_path}/model_exports/299/model_dict_[1, 2, 2, 8, 8, 48].npy', allow_pickle=True).item()
# weights_dict = np.load(f'{model_path}/model_exports/299/model_dict_[1, 3, 3, 6, 30, 60].npy', allow_pickle=True).item()

# 20k parmas no reg
# weights_dict = np.load(f'{model_path}/model_exports/302/model_dict_[1, 1, 1, 1, 1, 1].npy', allow_pickle=True).item()
# weights_dict = np.load(f'{model_path}/model_exports/302/model_dict_[1, 1, 1, 1, 2, 22].npy', allow_pickle=True).item()
# weights_dict = np.load(f'{model_path}/model_exports/302/model_dict_[1, 1, 3, 6, 6, 42].npy', allow_pickle=True).item()


# 20k parmas reg
# weights_dict = np.load(f'{model_path}/model_exports/303/model_dict_[1, 1, 1, 1, 1, 1].npy', allow_pickle=True).item()
# weights_dict = np.load(f'{model_path}/model_exports/303/model_dict_[1, 1, 1, 1, 9, 63].npy', allow_pickle=True).item()
# weights_dict = np.load(f'{model_path}/model_exports/303/model_dict_[1, 2, 2, 6, 18, 54].npy', allow_pickle=True).item()
weights_dict = np.load(f'{model_path}/model_exports/303/model_dict_[1, 2, 4, 16, 32, 64].npy', allow_pickle=True).item()
# weights_dict = np.load(f'{model_path}/model_exports/303/model_dict_[1, 4, 4, 16, 32, 64].npy', allow_pickle=True).item()
# weights_dict = np.load(f'{model_path}/model_exports/303/model_dict_[1, 4, 4, 16, 64, 64].npy', allow_pickle=True).item()

# 8k parmas no reg 304
# weights_dict = np.load(f'{model_path}/model_exports/304/model_dict_[1, 1, 1].npy', allow_pickle=True).item()
# weights_dict = np.load(f'{model_path}/model_exports/304/model_dict_[1, 1, 5].npy', allow_pickle=True).item()
# weights_dict = np.load(f'{model_path}/model_exports/304/model_dict_[1, 2, 22].npy', allow_pickle=True).item()
# weights_dict = np.load(f'{model_path}/model_exports/304/model_dict_[1, 4, 28].npy', allow_pickle=True).item()

# 8k parmas reg 305
# weights_dict = np.load(f'{model_path}/model_exports/305/model_dict_[1, 1, 1].npy', allow_pickle=True).item()
# weights_dict = np.load(f'{model_path}/model_exports/305/model_dict_[1, 2, 6].npy', allow_pickle=True).item()
# weights_dict = np.load(f'{model_path}/model_exports/305/model_dict_[1, 5, 20].npy', allow_pickle=True).item()
# weights_dict = np.load(f'{model_path}/model_exports/305/model_dict_[1, 9, 54].npy', allow_pickle=True).item()
# weights_dict = np.load(f'{model_path}/model_exports/305/model_dict_[1, 11, 55].npy', allow_pickle=True).item()
# weights_dict = np.load(f'{model_path}/model_exports/305/model_dict_[2, 8, 64].npy', allow_pickle=True).item()
# weights_dict = np.load(f'{model_path}/model_exports/305/model_dict_[3, 12, 48].npy', allow_pickle=True).item()

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
            write_weight(np.ones((1), dtype=np.float32), f"SkipLayer{index}", 'E', file)

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
            string += f'    layers::Sedge<float,Complex<float>>(A{index}, B{index}_1_{B_KP}, B{index}_bias, C{index}_1_{C_KP}, C{index}_bias, SkipLayer{index}_1_{S_KP}, LeakyReLU<float>),\n'
            # string += f'    layers::Sedge<float,Complex<float>>(A{index}, B{index}, B{index}_bias, C{index}, C{index}_bias, SkipLayer{index}, LeakyReLU<float>),\n'   # Default slow version

        elif f'A' in weights_dict[index].keys():
            string += f'    layers::Sedge<float,Complex<float>>(A{index}, B{index}_1_{B_KP}, B{index}_bias, C{index}_1_{C_KP}, C{index}_bias, Matrix<float, "E", 1>{{0}}, LeakyReLU<float>),\n'
            # string += f'    layers::Sedge<float,Complex<float>>(A{index}, B{index}, B{index}_bias, C{index}, C{index}_bias, Matrix<float, "E", 1>{{0}}, LeakyReLU<float>),\n' # Default slow version
    
    
    string += f'    layers::SumReduction<"S">(),\n'


    for index,_ in enumerate(weights_dict):
        if f'W' in weights_dict[index].keys():
            string += f'    layers::Linear<float>(W_1_{W_KP}, b, PassThrough<float>),\n'
            # string += f'    layers::Linear<float>(W, b, PassThrough<float>),\n' # Default slow version    
    file.write(string[:-2] + '\n);\n\n')

total_number_of_weights = 0
for index,_ in enumerate(weights_dict):
    for key in weights_dict[index].keys():
        if key != 'step_scale' and weights_dict[index][key] is not None:
            total_number_of_weights += weights_dict[index][key].size

print(f"Total number of weights: {total_number_of_weights}")

# %% Compile the unrolling script
if os.path.exists("weight_unrolling_exec.out"):
    os.remove("weight_unrolling_exec.out")
subprocess.run(f"{compiler} {' '.join(clang_flags)} -I {include_path} unroll_weights.cpp -o {current_pos}/weight_unrolling_exec.out", shell=True,executable="/bin/bash", stderr=subprocess.STDOUT)
subprocess.run(f"./weight_unrolling_exec.out", shell=True,executable="/bin/bash", stderr=subprocess.STDOUT)


#%% Compile the Network
if os.path.exists("CppSEdge.so"):
    os.remove("CppSEdge.so")
subprocess.run(f"{compiler} {' '.join(clang_flags)} -shared -fPIC $(python3 -m pybind11 --includes) -I {include_path} {current_pos}/test_SEdge.cpp -o  {current_pos}/CppSEdge.so", shell=True, executable="/bin/bash", stderr=subprocess.STDOUT)


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
CppSEdge.printModelInfo()

inputs = inputs.astype(np.float32)
classes = CppSEdge.run(inputs)

test_eval, test_classes = infere_model(inputs)

print("numpy model:", test_classes.shape, test_classes)
print("cpp model:", classes.shape, classes)
print("all close:", np.allclose(classes, test_classes, atol=1e-5))
print("diff:", (classes- test_classes)[0,:])


# %%

stepscale_times = CppSEdge.getStepScaleTimes()
if not np.all(stepscale_times==-1):
    print("Step scale times:", stepscale_times)
    plt.plot(stepscale_times, 'o', label="Step scale times")
    plt.xlabel("Step scale index")
    plt.ylabel("Time (ns)")
    plt.title("Step scale times")
    plt.yscale('log')
    plt.show()


    buckets = {}
    for i in range(len(stepscale_times)):
        for j in range(len(step_scale_tmp)-1,-1,-1):
            if (i+index_ofsets[j])%step_scale_tmp[j] == 0:
                if j not in buckets:
                    buckets[j] = []
                buckets[j].append([i,stepscale_times[i]])
                break

    plt.figure()
    for key in buckets.keys():
        print(f"Step scale {key} has {len(buckets[key])} times")
        buckets[key] = np.array(buckets[key])
        print(buckets[key].shape)
        plt.boxplot(buckets[key][:,1],positions=[key], widths=0.5, label=f"Step scale {step_scale_tmp[key]}")
    plt.ylabel("Time (ns)")
    plt.title("Step scale times per bucket")
    plt.yscale('log')
    # plt.legend()
    plt.show()

    plt.figure()
    bottom = 0
    for key in sorted(buckets.keys()):
        sum = np.sum(buckets[key][:,1])
        plt.bar(0, sum, bottom=bottom, label=f"Step scale {step_scale_tmp[key]}")
        bottom += sum
    plt.ylabel("Total time (ns)")
    plt.title("Total time per step scale")
    plt.legend()
    plt.show()