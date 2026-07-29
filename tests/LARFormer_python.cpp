#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "../include/Matrix.hpp"
#include "../include/pybind_interface.hpp"
#include "../include/layers/LARFormer.hpp"
#include "../include/helpers/print.hpp"
#include "../include/helpers/extended_matrix_ops.hpp"

#include "../include/types/Benchmark.hpp"


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

auto LARFormer = layers::LARFormer<UsedType,d_model>();

constexpr auto memory_buffer    = decltype(LARFormer)::template memory_buffer<InputMatrixType>;

auto buffer    = Matrix<char, "E", memory_buffer>();
auto permanent = Matrix<char, "E", 0>();



using OutputMatrixType = decltype(LARFormer)::OutputMatrix<InputMatrixType>;


using IWeightMatrixType = decltype(LARFormer)::IWeightMatrixType_;
using KWeightMatrixType = decltype(LARFormer)::KWeightMatrixType_;
using VWeightMatrixType = decltype(LARFormer)::VWeightMatrixType_;
using OWeightMatrixType = decltype(LARFormer)::OWeightMatrixType_;





InputMatrixType inputMatrix;
OutputMatrixType output;
auto forward(pybind11::array_t<Type> input) {
    randomize(buffer); // If this changes things we got a problem with uninitialized memory

    helpers::Benchmark::TypeInstance<Type>::resetAll(); // Reset the benchmark counters
    printBenchmark<Type>();
    
    convertToBaseMatrix(input, inputMatrix);


    LARFormer(inputMatrix,output,buffer, permanent);

    printBenchmark<Type>();

    return convertToNumpyArray<decltype(output), Type>(output);
}

IWeightMatrixType IWeightMatrix;
void set_IWeight(pybind11::array_t<Type> weight){
    convertToBaseMatrix(weight, IWeightMatrix);
    matrixAssign(LARFormer.IWeightMatrix, IWeightMatrix);
}

KWeightMatrixType KWeightMatrix;
void set_KWeight(pybind11::array_t<Type> weight){
    convertToBaseMatrix(weight, KWeightMatrix);
    matrixAssign(LARFormer.KWeightMatrix, KWeightMatrix);
}

VWeightMatrixType VWeightMatrix;
void set_VWeight(pybind11::array_t<Type> weight){
    convertToBaseMatrix(weight, VWeightMatrix);
    matrixAssign(LARFormer.VWeightMatrix, VWeightMatrix);
}

OWeightMatrixType OWeightMatrix;
void set_OWeight(pybind11::array_t<Type> weight){
    convertToBaseMatrix(weight, OWeightMatrix);
    matrixAssign(LARFormer.OWeightMatrix, OWeightMatrix);
}



auto get_memory_info() {
    return std::make_tuple(
        decltype(LARFormer)::template memory_buffer<InputMatrixType>,
        decltype(LARFormer)::template memory_minimal<InputMatrixType>
    );
}
PYBIND11_MODULE(LARFormer, m) {
    m.doc() = "Testst The python interface";

    m.def("forward", &forward, "Runs the forward function, requires a matrix of floats", pybind11::arg("input"));
   
    m.def("set_IWeight", &set_IWeight, "Set the I Weight Matrix of the Layer", pybind11::arg("weight"));
    m.def("set_KWeight", &set_KWeight, "Set the K Weight Matrix of the Layer", pybind11::arg("weight"));
    m.def("set_VWeight", &set_VWeight, "Set the V Weight Matrix of the Layer", pybind11::arg("weight"));
    m.def("set_OWeight", &set_OWeight, "Set the O Weight Matrix of the Layer", pybind11::arg("weight"));

    m.def("get_memory_info", &get_memory_info, "Returns (buffer_bytes, minimal_bytes)");

}
