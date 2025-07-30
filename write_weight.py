from typing import Union

# The C++ enum that is encoded into OrderConversion:
# enum DimensionOrder {
#     ERROR, // Error
#     // 1D
#     D1_Channel, // Default 1D
#     // 2D
#     D2_Batch_Channel,        // Default 2D
#     D2_Channel_Batch,        // Maybe Bacthnorm?
#     D2_OutChannel_InChannel, // Conv1d Weights
#     D2_InChannel_OutChannel, // Conv1d Weights Transposed
#     // 3D
#     D3_Batch_Channel_Width,         // Default 3D, Conv2d Depthwise Preffered
#     D3_Batch_Width_Channel,         // Conv1d Preffered
#     D3_Width_Batch_Channel,         // Conv1d  Strange
#     D3_OutChannel_InChannel_Kernel, // Conv1d Weights
#     D3_OutChannel_Kernel_InChannel, // Conv1d Weights Transposed
#     // 4D
#     D4_Batch_Channel_Width_Height,                    // Default 4D, Conv2d Depthwise Preffered
#     D4_Batch_Width_Height_Channel,                    // Conv2d Preffered
#     D4_OutChannel_InChannel_KernelWidth_KernelHeight, // Conv2d Weights
#     D4_OutChannel_InChannel_KernelParallel_Unrolled,  // Linear Parallel Unrolled Weights
#     // 5D
#     D5_OutChannel_InChannel_Kernel_KernelParallel_Unrolled, // Conv1d Parallel Unrolled Weights suboptimal Order
#     D5_OutChannel_Kernel_InChannel_KernelParallel_Unrolled, // Conv1d Parallel Unrolled Weights
# };
# How to Interprete the DimensionOrder
# \0 is ignored
# B is for Batch
# C is for Channels
# W is for Width either kernel or data
# H is for Height either kernel or data
# O is for OutChannel
# I is for InChannel
# P is for Parallel
# U is for Unrolled
OrderConversion = {
    'C': 'DimensionOrder::D1_Channel',

    'BC': 'DimensionOrder::D2_Batch_Channel',
    'SC': 'DimensionOrder::D2_Sequence_Channel',
    'CB': 'DimensionOrder::D2_Channel_Batch',
    'IO': 'DimensionOrder::D2_InChannel_OutChannel',
    'OI': 'DimensionOrder::D2_OutChannel_InChannel',

    'BCW': 'DimensionOrder::D3_Batch_Channel_Width',
    'BWC': 'DimensionOrder::D3_Batch_Width_Channel',
    'WBC': 'DimensionOrder::D3_Width_Batch_Channel',
    'OIW': 'DimensionOrder::D3_OutChannel_InChannel_Kernel',
    'OWI': 'DimensionOrder::D3_OutChannel_Kernel_InChannel',

    'BCWH': 'DimensionOrder::D4_Batch_Channel_Width_Height',
    'BWHC': 'DimensionOrder::D4_Batch_Width_Height_Channel',
    'WHBC': 'DimensionOrder::D4_Width_Height_Batch_Channel',
    'OIWH': 'DimensionOrder::D4_OutChannel_InChannel_KernelWidth_KernelHeight',
    'OWHI': 'DimensionOrder::D4_OutChannel_KernelWidth_KernelHeight_InChannel',
    'OIPU': 'DimensionOrder::D4_OutChannel_InChannel_KernelParallel_Unrolled',

    'OIWPU': 'DimensionOrder::D5_OutChannel_InChannel_Kernel_KernelParallel_Unrolled',
    'OWIPU': 'DimensionOrder::D5_OutChannel_Kernel_InChannel_KernelParallel_Unrolled',
}


def write_weight(weight, type, short_order : Union['C','BC', 'CB', 'IO', 'OI', 'BCW', 'BWC', 'WBC', 'OIW', 'OWI', 'BCWH', 'BWHC', 'OIWH', 'OIPU', 'OIWPU', 'OWIPU'] , weight_name, fileobject):
    if len(weight.shape) == 1:
        write_weight_1(weight, type, short_order, weight_name, fileobject)
    if len(weight.shape) == 2:
        write_weight_2(weight, type, short_order, weight_name, fileobject)
    if len(weight.shape) == 3:
        write_weight_3(weight, type, short_order, weight_name, fileobject)
    if len(weight.shape) == 4:
        write_weight_4(weight, type, short_order, weight_name, fileobject)

def write_weight_complex(weight, type, short_order : Union['C','BC','SC', 'CB', 'IO', 'OI', 'BCW', 'BWC', 'WBC', 'OIW', 'OWI', 'BCWH', 'BWHC', 'OIWH', 'OIPU', 'OIWPU', 'OWIPU'] , weight_name, fileobject):
    if len(weight.shape) == 2:
        write_weight_1_Complex(weight, type, short_order, weight_name, fileobject)
    elif len(weight.shape) == 3:
        write_weight_2_Complex(weight, type, short_order, weight_name, fileobject)
    # if len(weight.shape) == 4:
    #     write_weight_3(weight, type, short_order, weight_name, fileobject)
    # if len(weight.shape) == 5:
    #     write_weight_4(weight, type, short_order, weight_name, fileobject)
    else:
        raise NotImplementedError(f"Complex weights with shape {weight.shape} , len {len(weight.shape)} are not supported")



def write_weight_1(weight, type, short_order, weight_name, fileobject):
    fileobject.write(
        f"constexpr Matrix<{type},{OrderConversion[short_order]},{weight.shape[0]}> {weight_name} = ")
    fileobject.write("{{")
    for i in range(weight.shape[0]):
        fileobject.write(f"{weight[i]}")
        if i != weight.shape[0]-1:
            fileobject.write(",")
    fileobject.write("}};\n")

def write_weight_1_Complex(weight, type, short_order, weight_name, fileobject):
    fileobject.write(
        f"constexpr Matrix<Complex<{type}>,{OrderConversion[short_order]},{weight.shape[0]}> {weight_name} = ")
    fileobject.write("{{")
    for i in range(weight.shape[0]):
        fileobject.write(f"Complex<{type}>({weight[i,0]},{weight[i,1]})")
        if i != weight.shape[0]-1:
            fileobject.write(",")
    fileobject.write("}};\n")


def write_weight_2(weight, type, short_order, weight_name, fileobject):
    fileobject.write(
        f"constexpr Matrix<{type},{OrderConversion[short_order]},{weight.shape[0]},{weight.shape[1]}> {weight_name} = ")
    fileobject.write("{{")
    for i in range(weight.shape[0]):
        fileobject.write("{")
        for j in range(weight.shape[1]):
            fileobject.write(f"{weight[i,j]}")
            if j != weight.shape[1]-1:
                fileobject.write(",")
        fileobject.write("}")
        if i != weight.shape[0]-1:
            fileobject.write(",")
    fileobject.write("}};\n")

def write_weight_2_Complex(weight, type, short_order, weight_name, fileobject):
    fileobject.write(
        f"constexpr Matrix<Complex<{type}>,{OrderConversion[short_order]},{weight.shape[0]},{weight.shape[1]}> {weight_name} = ")
    fileobject.write("{{")
    for i in range(weight.shape[0]):
        fileobject.write("{")
        for j in range(weight.shape[1]):
            fileobject.write(f"Complex<{type}>({weight[i,j,0]},{weight[i,j,1]})")
            if j != weight.shape[1]-1:
                fileobject.write(",")
        fileobject.write("}")
        if i != weight.shape[0]-1:
            fileobject.write(",")
    fileobject.write("}};\n")


def write_weight_3(weight, type, short_order, weight_name, fileobject):
    fileobject.write(
        f"constexpr Matrix<{type},{OrderConversion[short_order]},{weight.shape[0]},{weight.shape[1]},{weight.shape[2]}> {weight_name} = ")
    fileobject.write("{{")
    for i in range(weight.shape[0]):
        fileobject.write("{")
        for j in range(weight.shape[1]):
            fileobject.write("{")
            for k in range(weight.shape[2]):
                fileobject.write(f"{weight[i,j,k]}")
                if k != weight.shape[2]-1:
                    fileobject.write(",")
            fileobject.write("}")
            if j != weight.shape[1]-1:
                fileobject.write(",")
        fileobject.write("}")
        if i != weight.shape[0]-1:
            fileobject.write(",")
    fileobject.write("}};\n")


def write_weight_4(weight, type, short_order, weight_name, fileobject):
    fileobject.write(
        f"constexpr Matrix<{type},{OrderConversion[short_order]},{weight.shape[0]},{weight.shape[1]},{weight.shape[2]},{weight.shape[3]}> {weight_name} = ")
    fileobject.write("{{")
    for i in range(weight.shape[0]):
        fileobject.write("{")
        for j in range(weight.shape[1]):
            fileobject.write("{")
            for k in range(weight.shape[2]):
                fileobject.write("{")
                for l in range(weight.shape[3]):
                    fileobject.write(f"{weight[i,j,k,l]}")
                    if l != weight.shape[3]-1:
                        fileobject.write(",")
                fileobject.write("}")
                if k != weight.shape[2]-1:
                    fileobject.write(",")
            fileobject.write("}")
            if j != weight.shape[1]-1:
                fileobject.write(",")
        fileobject.write("}")
        if i != weight.shape[0]-1:
            fileobject.write(",")
    fileobject.write("}};\n")
