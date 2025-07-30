# %%
# clang++-18 -Os -Wall -Wc++20-compat -shared -std=c++17 -fconstexpr-steps=100000000 -fPIC $(python3 -m pybind11 --includes) -I $base_path $1 -o  $filename.so
import torchaudio
from tqdm import tqdm
import pretty_midi
import torch
import numpy as np
import sys
import os
import subprocess
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__) + "/../../")

from write_weights import write_weight



compiler = "clang++-18"

# S_KP = 14*8
# B_KP = 12*4
# C_KP = 12*8
# SUB_BATCH=1


S_KP = 24
B_KP = 12
C_KP = 12*2
SUB_BATCH=4

# S_KP = 1

# sampling_rate = 16000
# sampling_rate = 24000
sampling_rate = 44100

model = f"MidiSSM_S_MH_maestro_all_{sampling_rate}_model"
# model = f"MidiSSM_L_MH_maestro_all_{sampling_rate}_model"
# model = f"MidiSSM_XL_MH_maestro_all_{sampling_rate}_model"

current_pos = '/'.join(__file__.split("/")[:-1])

print(current_pos)

weights_dict = torch.load(
    f'{current_pos}/{model}.pth', map_location=torch.device('cpu'))

type_of_weights = 'float'

input_scale = 1
input_offset = 0


complex_types = [np.complex64, np.complex128]


def discretize_zoh_norm_angle(norm_angle, B, B_bias, Delta, bias):
    """Discretize a diagonalized, continuous-time linear SSM
    using zero-order hold method.
    Args:
        Lambda (complex64): diagonal state matrix              (P,)
        B      (complex64): input matrix + bias                (P, H + 1)
        Delta (float32): discretization step sizes             (P,)
    Returns:
        discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H + 1)
    """
    if bias:
        B_concat = torch.cat((B, B_bias.unsqueeze(1)), dim=-1)
    else:
        B_concat = B
    Lambda_bar = torch.exp(torch.exp(norm_angle) * Delta)
    B_bar = ((Lambda_bar - 1)*torch.exp(-norm_angle))[..., None] * B_concat
    return Lambda_bar, B_bar


def as_complex(t: torch.Tensor, dtype=torch.complex64):
    assert t.shape[-1] == 2, "as_complex can only be done on tensors with shape=(...,2)"
    nt = torch.complex(t[..., 0], t[..., 1])
    if nt.dtype != dtype:
        nt = nt.type(dtype)
    return nt


class SequenceLayer(torch.nn.Module):
    def __init__(self, norm, angle, B2, B2_bias, C, C_bias, SkipLayer):
        super(SequenceLayer, self).__init__()
        self.norm = torch.nn.Parameter(norm)
        self.angle = torch.nn.Parameter(angle)
        self.B2 = torch.nn.Parameter(B2)
        self.B2_bias = torch.nn.Parameter(B2_bias)
        self.C = torch.nn.Parameter(C)
        self.C_bias = torch.nn.Parameter(C_bias)
        self.SkipLayer = torch.nn.Parameter(
            SkipLayer) if SkipLayer is not None else None

    def convert_to_discrete(self, step_scale=1):
        norm_angle = torch.complex(self.norm, self.angle)
        self.C_bars = as_complex(self.C)
        self.C_bias_bars = as_complex(self.C_bias)
        B_c = (self.B2)
        B_bias_c = (self.B2_bias)
        self.step = step_scale

        Lambda_bars, B_bars = discretize_zoh_norm_angle(
            norm_angle, B_c, B_bias_c, step_scale, True)

        self.Lambda_bars = Lambda_bars
        self.B_bar = B_bars[:, 0:-1]
        self.B_bias_bar = B_bars[:, -1]


with open(f'{current_pos}/weights.hpp', "w") as file:
    file.write('#pragma once\n')
    file.write('#include "./Matrix.hpp"\n')
    file.write('#include "./types/Complex.hpp"\n')
    file.write('constexpr std::size_t B_KP = ' + str(B_KP) + ';\n')
    file.write('constexpr std::size_t C_KP = ' + str(C_KP) + ';\n')
    file.write('constexpr std::size_t S_KP = ' + str(S_KP) + ';\n')
    file.write('\n')


    layers = []
    key = f'seq_midi.{1}.seq.0.seq'
    layers.append(SequenceLayer(weights_dict.get(key + '.norm', None), weights_dict.get(key + '.angle', None), weights_dict.get(key + '.B2', None), weights_dict.get(
        key + '.B2_bias', None), weights_dict.get(key + '.C', None), weights_dict.get(key + '.C_bias', None), weights_dict.get(key + '.SkipLayer', None)))
    for i in range(2, 5):
        key = f'seq_midi.{i}.s5.seq'
        layers.append(SequenceLayer(weights_dict.get(key + '.norm', None), weights_dict.get(key + '.angle', None), weights_dict.get(key + '.B2', None), weights_dict.get(
            key + '.B2_bias', None), weights_dict.get(key + '.C', None), weights_dict.get(key + '.C_bias', None), weights_dict.get(f'seq_midi.{i}' + '.skipLayer.weight', None)))

    B_scalar = 1
    skip_scalar = 1

    for index, layer in enumerate(layers):
        print(f"Layer {index}")
        if index > 0:
            B_scalar = 1.0/np.sqrt(2)
            skip_scalar = 1.0/np.sqrt(2)

        layer.convert_to_discrete()
        if layer.Lambda_bars is not None:
            tmp_A = layer.Lambda_bars.detach().cpu().numpy()
            write_weight(tmp_A, f"A{index}", 'C', file)
        if layer.B_bar is not None:
            tmp_B = layer.B_bar.detach().cpu().numpy()
            tmp_B = tmp_B * B_scalar
            print(tmp_B.dtype)
            write_weight(tmp_B, f"B{index}", 'OI', file)
        if layer.B_bias_bar is not None:
            tmp_B_bias = layer.B_bias_bar.detach().cpu().numpy()
            write_weight(tmp_B_bias, f"B{index}_bias", 'C', file)
        if layer.C_bars is not None:
            tmp_C = layer.C_bars.detach().cpu().numpy()
            write_weight(tmp_C, f"C{index}", 'OI', file)

        if layer.C_bias_bars is not None:
            tmp_C_bias = layer.C_bias_bars.detach().cpu().numpy()
            if tmp_C_bias.dtype in complex_types:
                print("C_bias is complex, check if it should be")
                # tmp_C_bias = tmp_C_bias.real
                tmp_C_bias = np.real(tmp_C_bias)
            write_weight(tmp_C_bias, f"C{index}_bias", 'C', file)
        if layer.SkipLayer is not None:
            tmp_SkipLayer = layer.SkipLayer.detach().cpu().numpy()
            tmp_SkipLayer = tmp_SkipLayer * skip_scalar
            write_weight(
                tmp_SkipLayer, f"SkipLayer{index}_weights", 'OI', file)

    if weights_dict.get('year_linear.4.weight', None) is not None:

        tmp_W = weights_dict['year_linear.4.weight'].detach().cpu().numpy()
        tmp_W = tmp_W*skip_scalar
        write_weight(tmp_W, 'Decoder_weights', 'OI',  file)
    if weights_dict.get('year_linear.4.bias', None) is not None:
        tmp_b = weights_dict['year_linear.4.bias'].detach().cpu().numpy()
        write_weight(tmp_b, 'Decoder_bias', 'C', file)


with open(f'{current_pos}/network.hpp', "w") as file:
    file.write('#pragma once\n')
    file.write('#include "./weights.hpp"\n')
    file.write('#include "./weights_unrolled.hpp"\n')
    file.write('#include "./NeuralNetwork.hpp"\n')
    file.write('\n')
    file.write(f'constexpr std::size_t SUB_BATCH = {SUB_BATCH};\n')
    file.write(
        '// const auto __attribute__(( section(".data") )) network=layers::Sequence(\n')
    file.write('constexpr auto network=layers::Sequence(\n')
    string = ""
    # for index in range(len(layers)):
    for index in range(len(layers)):
        if layers[index].SkipLayer is not None:
            # layers::S5_class_hidden(A, B, C, BiasRNN, BiasOut, Passthrough),
            # string += f'    layers::Sedge(A{index}, B{index}, B{index}_bias, C{index}, C{index}_bias, SkipLayer{index}_weights, Tanh<float>),\n'
            string += f'    layers::Sedge<float,Complex<float>,SUB_BATCH,SUB_BATCH>(A{index}, B{index}_1_{B_KP}, B{index}_bias, C{index}_1_{C_KP}, C{index}_bias, SkipLayer{index}_weights_1_{S_KP}, FastTanh<float>),\n'
            pass
        else:
            # layers::S5_class_hidden(A, B, C, BiasRNN, BiasOut, Passthrough),
            # string += f'    layers::SSMPiano<float>(A{index}, B{index}, B{index}_bias, C{index}, C{index}_bias, Tanh<float>),\n'
            string += f'    layers::SSMPiano<float,Complex<float>,SUB_BATCH,SUB_BATCH>(A{index}, B{index}_1_{B_KP}, B{index}_bias, C{index}_1_{C_KP}, C{index}_bias, FastTanh<float>),\n'

    string += f'    layers::Linear<float,8,"BS">(Decoder_weights_1_1, Decoder_bias, PassThrough<float>),\n'
    file.write(string[:-2] + '\n);\n\n')

    # file.write(
    #     '// const auto __attribute__(( section(".data") )) Decoder=layers::Sequence(\n')
    # file.write('const auto Decoder=layers::Sequence(\n')
    # string = ""
    # file.write(string[:-2] + '\n);\n\n')


# # total_number_of_weights = 0
# for index in range(len(weights_dict)):
#     for key in weights_dict[index].keys():
#         if key != 'step_scale' and weights_dict[index][key] is not None:
#             total_number_of_weights += weights_dict[index][key].size

# print(f"Total number of weights: {total_number_of_weights}")

# # %% Compile
# if os.path.exists("weights_unrolled.hpp"):
#     os.remove("weights_unrolled.hpp")

# if os.path.exists("a.out"):
#     os.remove("a.out")

subprocess.run(f"{compiler} -Wall -std=c++20 -Ofast -march=native -ftemplate-depth=10000 -fconstexpr-steps=100000000 -Wall -I {current_pos}/../../ unroll_weights.cpp", shell=True,executable="/bin/bash", stderr=subprocess.STDOUT)
subprocess.run(f"./a.out", shell=True,executable="/bin/bash", stderr=subprocess.STDOUT)

#%% 
if os.path.exists("CppPianoSSM.so"):
    os.remove("CppPianoSSM.so")
# subprocess.run(f"{compiler} -march=native --target=x86_64-pc-linux-gnu -Os -Wpedantic -Wc++20-compat -shared -std=c++17 -fconstexpr-steps=100000000 -fPIC $(python3 -m pybind11 --includes) -I {current_pos}/../ {current_pos}/python_S5.cpp -o  {current_pos}/CppS5.so", shell=True,executable="/bin/bash", stderr=subprocess.STDOUT)
subprocess.run(f"{compiler} -march=native -march=native -Os -Wpedantic -shared -std=c++20 -fconstexpr-steps=100000000 -fPIC $(python3 -m pybind11 --includes) -I {current_pos}/../../ {current_pos}/test_paino_ssm.cpp -o  {current_pos}/CppPianoSSM.so",
               shell=True, executable="/bin/bash", stderr=subprocess.STDOUT)
# subprocess.run("g++ -Wall -std=c++17 -fstack-usage -fconstexpr-ops-limit=100000000 -march=native -Ofast -ftemplate-depth=10000 -Wc++20-compat -shared -fPIC $(python3 -m pybind11 --includes) -I ../ python_S5.cpp -o  CppS5.so", shell=True,executable="/bin/bash", stderr=subprocess.STDOUT)

# %%
import CppPianoSSM


midi_sampling_rate = 100
# inputs = (np.load("midi.npy"))#[np.newaxis,...]
# # inputs = np.load("midi (2).npy")
# inputs = inputs.repeat(sampling_rate//midi_sampling_rate,axis=1)
midi_data = pretty_midi.PrettyMIDI('mond_1.mid')
# midi_data = pretty_midi.PrettyMIDI('Yiruma - Rivers Flow In You.mid')

inputs = midi_data.get_piano_roll(fs=midi_sampling_rate)
inputs = inputs[21:109, :]
inputs = inputs.T
inputs = inputs.reshape(1, -1, 88)
inputs = inputs/127
print(inputs.shape)

# inputs = inputs[:,:4*24000,...]
# inputs = inputs[:,:10,...]
#
# inputs = np.zeros_like(inputs)

print(inputs.shape)
inputs = inputs.astype(np.float32)
# inputs = inputs.reshape(1,-1,88)

# plt.imshow(inputs[0,:,:].T,aspect='auto')
# plt.colorbar()
# plt.show()
# %%
# plt.imshow(inputs[0, :, :].T, aspect='auto')
# plt.colorbar()
# plt.show()

# tmp = inputs.copy()
# tmp = tmp[0, ...].flatten()
# tmp = tmp[tmp > 0]
# plt.hist(tmp, bins=100)
# plt.show()


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
# S5_out,classes = CppS5.run_S5(inputs)


upsampling = sampling_rate//midi_sampling_rate

# lenght = 4*24000//upsampling
# lenght = 44100//upsampling
lenght = CppPianoSSM.getInputSize()//upsampling
# lenght = (441*40*4*1)//upsampling


outputs = []
for i in tqdm(range(inputs.shape[1]//lenght)):
    # print(lenght)
    # print(upsampling)

    # for i in tqdm(range(10)):
    input_slice = inputs[:, i*lenght:(i+1)*lenght, ...]
    input_slice = input_slice.repeat(upsampling, axis=1)


#     #benchmarking
#     input_slice = input_slice[:,:CppPianoSSM.getInputSize(),:]  # Limit to 200 time steps
#     input_slice_tmp = np.zeros((1, CppPianoSSM.getInputSize(), 88), dtype=np.float32)
#     input_slice_tmp[0, :input_slice.shape[1], :] = input_slice[0, ...]
#     input_slice = input_slice_tmp
    
    
    # print("input shape: ", input_slice.shape)
    tmp = CppPianoSSM.run(input_slice)
    outputs.append(tmp)
    # break
# S5_out,classes = CppS5.run_S5(inputs)
outputs = np.concatenate(outputs, axis=1)

CppPianoSSM.printModelInfo()

# %%

# outputs = outputs.reshape(1,-1)
print(outputs.shape)

# %%
audio_mean = -2.1367119188653305e-05
audio_std = 0.060498714447021484


outputs = outputs * audio_std + audio_mean


# # display audio
# import IPython.display as ipd
# ipd.Audio(outputs, rate=16000)

plt.plot(outputs[0, ...].mean(axis=1))
plt.show()

torchaudio.save("./output.wav", torch.tensor(outputs[..., 0]), sampling_rate)

# %%
