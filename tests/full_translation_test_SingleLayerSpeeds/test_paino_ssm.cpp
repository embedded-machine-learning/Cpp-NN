#include <chrono>
#include <iostream>
#include <memory>
#include <tuple>

#define __LINEAR_FORCE_INLINE__ true // clang will be faster with forced inlining, might break gcc

#include "./include/NeuralNetwork.hpp"
#include "./include/helpers/human_readable_types.hpp"
#include "./include/helpers/print.hpp"
#include "./include/pybind_interface.hpp"

#include "./include/hardware/AVX2.hpp" // For AVX2 specific optimizations

#include <chrono>

#define __OP_COUNTING__ false // Enable Operation counting
// #define __OP_COUNTING__ true // Enable Operation counting

#if !__OP_COUNTING__
using Type = float;
#else
#include "./include/types/Benchmark.hpp"
using Type = helpers::Benchmark::TypeInstance<float>;
#define float Type
#endif

#include "./network.hpp"


constexpr Dim_size_t DesiredLength = 44100*4 ; // Desired length of the sequence, can be adjusted
constexpr Dim_size_t batch_size          = 1;
constexpr Dim_size_t sequence_length_sub = (SUB_BATCH > 1) ? SUB_BATCH * 4 : 1;
constexpr Dim_size_t sequence_length     = DesiredLength - (DesiredLength) % sequence_length_sub; // better hope the modulo is zero

// constexpr Dim_size_t sequence_length     = DesiredLength;
// constexpr Dim_size_t sequence_length_sub = 1;

constexpr Dim_size_t input_channels = 8;

static_assert(sequence_length % sequence_length_sub == 0, "Sequence length must be a multiple of sequence length sub");

using NetworkType = decltype(network);

using InputMatrixType  = Matrix<Type, "BSC", batch_size, sequence_length, input_channels>;
using OutputMatrixType = NetworkType::OutputMatrix<InputMatrixType>;

using SubInputMatrixType  = OverrideDimensionMatrix<InputMatrixType, "S", sequence_length_sub>;
using SubOutputMatrixType = OverrideDimensionMatrix<OutputMatrixType, "S", sequence_length_sub>;

// Define Memory and Buffer sizes
constexpr auto memory_permanent = NetworkType::memory_permanent<SubInputMatrixType>;
constexpr auto memory_buffer    = NetworkType::memory_buffer<SubInputMatrixType>;

auto buffer    = Matrix<char, "E", memory_buffer>();
auto permanent = Matrix<char, "E", memory_permanent>();

auto input_matrix  = InputMatrixType();
auto output_matrix = OutputMatrixType();

// auto input_slice  = SubInputMatrixType();
// auto output_slice = SubOutputMatrixType();

#if __OP_COUNTING__
#undef float
#endif

auto run_PianoSSM(pybind11::array_t<float> input) {
    // auto start_time = std::chrono::high_resolution_clock::now();
    convertToBaseMatrix<InputMatrixType, float>(input, input_matrix);
#if __OP_COUNTING__
    // helpers::Benchmark::TypeInstance<float>::resetAll(); // Reset the benchmark counters
    #endif
    auto start_time = std::chrono::high_resolution_clock::now();
#pragma GCC unroll(1)
    for (Dim_size_t i = 0; i < sequence_length; i += sequence_length_sub) {
        // std::cout << "Running sub-sequence " << i << std::endl;
        const auto input_slice  = permute<"BSC">(slice<"S", sequence_length_sub>(input_matrix, {i}));
        auto       output_slice = permute<"BSC">(slice<"S", sequence_length_sub>(output_matrix, {i}));
        // const auto& input_slice  = *reinterpret_cast<SubInputMatrixType *>(&input_matrix.data[i * input_channels]);
        // auto& output_slice =  *reinterpret_cast<SubOutputMatrixType *>(&output_matrix.data[i]);

        // Run the network
        network(input_slice, output_slice, buffer, permanent);
    }
    std::cout << "Time to run the network: " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << " microseconds" << std::endl;
#if __OP_COUNTING__
#endif
    return convertToNumpyArray<OutputMatrixType, float>(output_matrix);
}

void printModelInfo() {
    std::cout << "Network size: " << sizeof(NetworkType) << std::endl;
    std::cout << "Input Matrix Type: " << human_readable_type<InputMatrixType> << std::endl;
    std::cout << "Output Matrix Type: " << human_readable_type<OutputMatrixType> << std::endl;
    std::cout << "Input Slice Type: " << human_readable_type<SubInputMatrixType> << std::endl;
    std::cout << "Output Slice Type: " << human_readable_type<SubOutputMatrixType> << std::endl;
    std::cout << "Memory Permanent Size: " << memory_permanent << std::endl;
    std::cout << "Memory Buffer Size: " << memory_buffer << std::endl;

#if __OP_COUNTING__
    std::cout << "Operation Counting Enabled" << std::endl;
    const auto input_slice  = slice<"S", sequence_length_sub>(input_matrix, {0});
    auto       output_slice = slice<"S", sequence_length_sub>(output_matrix, {0});
    printBenchmark<float>();
    helpers::Benchmark::TypeInstance<float>::resetAll(); // Reset the benchmark counters
    network(input_slice, output_slice, buffer, permanent);
    printBenchmark<float>();
#endif
}

int getInputSize() {
    return input_matrix.dimensions[1];
}

PYBIND11_MODULE(CppPianoSSM, m) {
    m.doc() = "Runs the Network";

    m.def("run", &run_PianoSSM, "Runs the network, takes input as a numpy array");
    m.def("printModelInfo", &printModelInfo, "Prints the model information");
    m.def("getInputSize", &getInputSize, "Returns the input size of the network");
}