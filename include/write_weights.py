import numpy as np
def format_float_smart(n):
    if isinstance(n, float):
        if n.is_integer():
            return f"{n:.1f}"  # force .0 for whole floats
        else:
            return repr(n)     # full precision for non-whole floats
    return str(n)  # leave ints or other types as-is

type_map = {
    np.dtype(np.float64):       {"type": "double",                "write_fnction": lambda value: f"{format_float_smart(value)}"},                                                             
    np.dtype(np.float32):       {"type": "float",                 "write_fnction": lambda value: f"(float){format_float_smart(value)}"},                                         
    np.dtype(np.complex128):    {"type": "Complex<double>",       "write_fnction": lambda value: f"Complex<double>({format_float_smart(np.real(value))}, {format_float_smart(np.imag(value))})"},                                                         
    np.dtype(np.complex64):     {"type": "Complex<float>",        "write_fnction": lambda value: f"Complex<float>({format_float_smart(np.real(value))}f, {format_float_smart(np.imag(value))}f)"},                                                     
    np.dtype(np.int64):         {"type": "int64_t",               "write_fnction": lambda value: f"{value}"},                                         
    np.dtype(np.uint64):        {"type": "uint64_t",              "write_fnction": lambda value: f"{value}"},                                             
    np.dtype(np.int32):         {"type": "int32_t",               "write_fnction": lambda value: f"{value}"},                                         
    np.dtype(np.uint32):        {"type": "uint32_t",              "write_fnction": lambda value: f"{value}"},                                             
    np.dtype(np.int16):         {"type": "int16_t",               "write_fnction": lambda value: f"{value}"},                                         
    np.dtype(np.uint16):        {"type": "uint16_t",              "write_fnction": lambda value: f"{value}"},                                             
    np.dtype(np.int8):          {"type": "int8_t",                "write_fnction": lambda value: f"{value}"},                                         
    np.dtype(np.uint8):         {"type": "uint8_t",               "write_fnction": lambda value: f"{value}"},                                         
    np.dtype(np.bool_):         {"type": "bool",                  "write_fnction": lambda value: f"{value}"},                                         
}


def write_weight(weight:np.ndarray, name:str, order:str, fileobject):
    if not isinstance(weight, np.ndarray):
        raise TypeError("weight must be a numpy ndarray")
    if weight.ndim != len(order):
        raise ValueError(f"weight must have {len(order)} dimensions, got {weight.ndim}")
    if weight.dtype not in type_map:
        raise ValueError(f"Unsupported dtype {weight.dtype} for weight")
    type = type_map[weight.dtype]

    fileobject.write(f'constexpr Matrix<{type['type']},"{order}",{','.join(map(str, weight.shape))}> {name} = ')
    weights = weight.flatten()
    fileobject.write("{{")
    for i, value in enumerate(weights):
        fileobject.write(type['write_fnction'](value))
        if i != len(weights) - 1:
            fileobject.write(", ")
    fileobject.write("}};\n")
