#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "../include/Matrix.hpp"
#include "../include/pybind_interface.hpp"
#include "../include/layers/LowRank.hpp"
#include "../include/helpers/print.hpp"
#include "../include/helpers/extended_matrix_ops.hpp"


#include "../include/types/Benchmark.hpp"

constexpr Dim_size_t dimension_k = 32;
constexpr Dim_size_t dimension_v = 48;
constexpr Dim_size_t N           = 700;
constexpr Dim_size_t d_model     = 64;
constexpr Dim_size_t r           = 10;

#define __OP_COUNTING__ true

using Type = float;

#if !__OP_COUNTING__
using UsedType = Type;
#else
using UsedType = helpers::Benchmark::TypeInstance<Type>;
#endif

using InputMatrixType = Matrix<UsedType, "NC", N, d_model>;

auto LowRank = layers::LowRank<UsedType,dimension_k,dimension_v,d_model, r>();

constexpr auto memory_buffer    = decltype(LowRank)::template memory_buffer<InputMatrixType>;

auto buffer    = Matrix<char, "E", memory_buffer>();
auto permanent = Matrix<char, "E", 0>();



using OutputMatrixType = decltype(LowRank)::OutputMatrix<InputMatrixType>;


using QWeightMatrixType = decltype(LowRank)::QWeightMatrixType_;
using VTrQWeightMatrixType = decltype(LowRank)::VTrQWeightMatrixType_;   
using KWeightMatrixType = decltype(LowRank)::KWeightMatrixType_;
using VTrKWeightMatrixType = decltype(LowRank)::VTrKWeightMatrixType_;   
using VWeightMatrixType = decltype(LowRank)::VWeightMatrixType_;
using VTrVWeightMatrixType = decltype(LowRank)::VTrVWeightMatrixType_;   



OutputMatrixType output;
InputMatrixType inputMatrix;

auto forward(pybind11::array_t<Type, pybind11::array::c_style> input) {
    randomize(buffer);  // If this changes things we got a problem with uninitialized memory

    helpers::Benchmark::TypeInstance<Type>::resetAll(); // Reset the benchmark counters
    printBenchmark<Type>();
    
    convertToBaseMatrix(input, inputMatrix);

    LowRank(inputMatrix,output,buffer, permanent);

    printBenchmark<Type>();

    return convertToNumpyArray<decltype(output), Type>(output);
}

QWeightMatrixType QWeightMatrix;
void set_QWeight(pybind11::array_t<Type, pybind11::array::c_style> weight){
    convertToBaseMatrix(weight, QWeightMatrix);
    matrixAssign(LowRank.QWeightMatrix, QWeightMatrix);
}

VTrQWeightMatrixType VTrQWeightMatrix;
void set_VTrQWeight(pybind11::array_t<Type, pybind11::array::c_style> weight){
    convertToBaseMatrix(weight, VTrQWeightMatrix);
    matrixAssign(LowRank.VTrQWeightMatrix, VTrQWeightMatrix);
}

KWeightMatrixType KWeightMatrix;
void set_KWeight(pybind11::array_t<Type, pybind11::array::c_style> weight){
    convertToBaseMatrix(weight, KWeightMatrix);
    matrixAssign(LowRank.KWeightMatrix, KWeightMatrix);
}

VTrKWeightMatrixType VTrKWeightMatrix;
void set_VTrKWeight(pybind11::array_t<Type, pybind11::array::c_style> weight){
    convertToBaseMatrix(weight, VTrKWeightMatrix);
    matrixAssign(LowRank.VTrKWeightMatrix, VTrKWeightMatrix);
}

VWeightMatrixType VWeightMatrix;
void set_VWeight(pybind11::array_t<Type, pybind11::array::c_style> weight){
    convertToBaseMatrix(weight, VWeightMatrix);
    matrixAssign(LowRank.VWeightMatrix, VWeightMatrix);
}

VTrVWeightMatrixType VTrVWeightMatrix;
void set_VTrVWeight(pybind11::array_t<Type, pybind11::array::c_style> weight){
    convertToBaseMatrix(weight, VTrVWeightMatrix);
    matrixAssign(LowRank.VTrVWeightMatrix, VTrVWeightMatrix);
}
auto get_memory_info() {
    return std::make_tuple(
        sizeof(buffer),
        decltype(LowRank)::template memory_minimal<InputMatrixType>
    );
}


PYBIND11_MODULE(LowRank, m) {
    m.doc() = "Testst The python interface";

    m.def("forward", &forward, "Runs the forward function, requires a matrix of floats", pybind11::arg("input"));
   
    m.def("set_QWeight", &set_QWeight, "Set the Q Weight Matrix of the Layer", pybind11::arg("weight"));
    m.def("set_VTrQWeight", &set_VTrQWeight, "Set the VTr_Q Weight Matrix of the Layer", pybind11::arg("weight"));

    m.def("set_KWeight", &set_KWeight, "Set the K Weight Matrix of the Layer", pybind11::arg("weight"));
    m.def("set_VTrKWeight", &set_VTrKWeight, "Set the VTr_K Weight Matrix of the Layer", pybind11::arg("weight"));

    m.def("set_VWeight", &set_VWeight, "Set the V Weight Matrix of the Layer", pybind11::arg("weight"));
    m.def("set_VTrVWeight", &set_VTrVWeight, "Set the VTr_V Weight Matrix of the Layer", pybind11::arg("weight"));

    m.def("get_memory_info", &get_memory_info, "Returns (buffer_bytes, minimal_bytes)");
}