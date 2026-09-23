#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "../include/Matrix.hpp"
#include "../include/pybind_interface.hpp"
#include "../include/layers/BasicAttention.hpp"
#include "../include/helpers/print.hpp"
#include "../include/helpers/extended_matrix_ops.hpp"

#include "../include/types/Benchmark.hpp"

constexpr Dim_size_t dimension_k = 32;
constexpr Dim_size_t dimension_v = 48;
constexpr Dim_size_t N           = 700;
constexpr Dim_size_t d_model     = 64;

#define __OP_COUNTING__ true

using Type = float;

#if !__OP_COUNTING__
using UsedType = Type;
#else
using UsedType = helpers::Benchmark::TypeInstance<Type>;
#endif

using InputMatrixType = Matrix<UsedType, "NC", N, d_model>;

auto BasicAttention = layers::BasicAttention<UsedType,dimension_k,dimension_v,d_model>();

constexpr auto memory_buffer    = decltype(BasicAttention)::template memory_buffer<InputMatrixType>;

auto buffer    = Matrix<char, "E", memory_buffer>();
auto permanent = Matrix<char, "E", 0>();



using OutputMatrixType = decltype(BasicAttention)::OutputMatrix<InputMatrixType>;


using QWeightMatrixType = decltype(BasicAttention)::QWeightMatrixType_;
using KWeightMatrixType = decltype(BasicAttention)::KWeightMatrixType_;
using VWeightMatrixType = decltype(BasicAttention)::VWeightMatrixType_;




InputMatrixType inputMatrix;
OutputMatrixType output;
auto forward(pybind11::array_t<Type, pybind11::array::c_style> input) {
    randomize(buffer);  // If this changes things we got a problem with uninitialized memory
    helpers::Benchmark::TypeInstance<Type>::resetAll(); // Reset the benchmark counters
    printBenchmark<Type>();
    
    convertToBaseMatrix(input, inputMatrix);


    BasicAttention(inputMatrix,output,buffer, permanent);

    printBenchmark<Type>();

    return convertToNumpyArray<decltype(output), Type>(output);
}

QWeightMatrixType QWeightMatrix;
void set_QWeight(pybind11::array_t<Type, pybind11::array::c_style> weight){
    convertToBaseMatrix(weight, QWeightMatrix);
    matrixAssign(BasicAttention.QWeightMatrix, QWeightMatrix);
}

KWeightMatrixType KWeightMatrix;
void set_KWeight(pybind11::array_t<Type, pybind11::array::c_style> weight){
    convertToBaseMatrix(weight, KWeightMatrix);
    matrixAssign(BasicAttention.KWeightMatrix, KWeightMatrix);
}

VWeightMatrixType VWeightMatrix;
void set_VWeight(pybind11::array_t<Type, pybind11::array::c_style> weight){
    convertToBaseMatrix(weight, VWeightMatrix);
    matrixAssign(BasicAttention.VWeightMatrix, VWeightMatrix);
}


auto get_memory_info() {
    return std::make_tuple(
        decltype(BasicAttention)::template memory_buffer<InputMatrixType>,
        decltype(BasicAttention)::template memory_minimal<InputMatrixType>
    );
}


PYBIND11_MODULE(BasicAttention, m) {
    m.doc() = "Testst The python interface";

    m.def("forward", &forward, "Runs the forward function, requires a matrix of floats", pybind11::arg("input"));
   
    m.def("set_QWeight", &set_QWeight, "Set the Q Weight Matrix of the Layer", pybind11::arg("weight"));
    m.def("set_KWeight", &set_KWeight, "Set the K Weight Matrix of the Layer", pybind11::arg("weight"));
    m.def("set_VWeight", &set_VWeight, "Set the V Weight Matrix of the Layer", pybind11::arg("weight"));
    m.def("get_memory_info", &get_memory_info, "Returns (buffer_bytes, minimal_bytes)");
}