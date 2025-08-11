import os
import sys
import numpy as np


include_path = '../../'

sys.path.append(os.path.join(os.path.dirname(__file__),include_path, "include"))
from write_weights import write_weight

complex_types = [np.complex64, np.complex128]


def write_network_weights(weights_dict, step_scale, file):
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



def write_network(weights_dict, file, B_KP=(12*4), C_KP=(12*8), S_KP=(14*8), W_KP=(14*8)):
    file.write('// const auto __attribute__(( section(".data") )) network=layers::Sequence(\n')
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