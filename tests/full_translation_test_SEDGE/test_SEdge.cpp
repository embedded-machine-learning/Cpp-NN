#include <chrono>
#include <iostream>

#if __clang__
#define __LINEAR_FORCE_INLINE__ true // faster but can break gcc
#else
#define __LINEAR_FORCE_INLINE__ false
#endif

#include "include/NeuralNetwork.hpp"
#include "include/helpers/human_readable_types.hpp"
#include "include/helpers/print.hpp"
#include "include/pybind_interface.hpp"

#include "./include/hardware/AVX2.hpp" // For AVX2 specific optimizations

#define __OP_COUNTING__ false // Enable Operation counting
// #define __OP_COUNTING__ true // Enable Operation counting

#ifndef __BENCHMARK__
#define __BENCHMARK__ 0     // Enable Benchmarking
// #define __BENCHMARK__ 1     // Simple Benchmarking
// #define __BENCHMARK__ 2     // Full Benchmarking
#endif

#if !__OP_COUNTING__
using Type = float;
#else
#include "include/types/Benchmark.hpp"
using Type = helpers::Benchmark::TypeInstance<float>;
#define float Type
#endif

#include "./network.hpp"

constexpr Dim_size_t DesiredLength = 16000; // Desired length of the sequence, can be adjusted

constexpr Dim_size_t batch_size          = 1;
constexpr Dim_size_t sequence_length_sub = 1;
constexpr Dim_size_t sequence_length     = DesiredLength;

constexpr Dim_size_t input_channels = 1;

static_assert(sequence_length % sequence_length_sub == 0, "Sequence length must be a multiple of sequence length sub");

using NetworkType = decltype(network);

using InputMatrixType  = Matrix<Type, "BSC", batch_size, sequence_length, input_channels>;
using OutputMatrixType = NetworkType::OutputMatrix<InputMatrixType>;

using SubInputMatrixType = OverrideDimensionMatrix<InputMatrixType, "S", sequence_length_sub>;

// Define Memory and Buffer sizes
constexpr auto memory_permanent = NetworkType::memory_permanent<SubInputMatrixType>;
constexpr auto memory_buffer    = NetworkType::memory_buffer<SubInputMatrixType>;

auto buffer    = Matrix<char, "E", memory_buffer>();
auto permanent = Matrix<char, "E", memory_permanent>();

auto input_matrix  = InputMatrixType();
auto output_matrix = OutputMatrixType();

#if __BENCHMARK__ == 2
std::array<std::chrono::high_resolution_clock::time_point, sequence_length + 1> layer_times; // Store the time points for each layer execution, if desired
auto                                                                            measure = [](int index) { layer_times[index + 1] = std::chrono::high_resolution_clock::now(); };
#endif

#if __OP_COUNTING__
#undef float
#endif

template <int Length, int TrueUpto>
    requires(Length >= TrueUpto && TrueUpto >= 0)
constexpr std::array<bool, Length> continueCalculationAfter = concat(makeFilledArray<bool, TrueUpto>(true), makeFilledArray<bool, Length - TrueUpto>(false));

template <int Length>
constexpr std::array<bool, Length> continueCalculationAfter<Length, Length - 1> = makeFilledArray<bool, Length>(true); // the last layer always continues calculation

template <int TrueUpto>
    requires(NetworkType::layer_count >= TrueUpto && TrueUpto >= 0)
constexpr std::array<bool, NetworkType::layer_count> continueNetworkCalculationAfter = continueCalculationAfter<NetworkType::layer_count, TrueUpto>;

template <IsMatrixType InputMatrixType, IsMatrixType OutputMatrixType, int IndexChecked = NetworkType::layer_count - 1>
__attribute__((always_inline)) inline void runWithStepScale(int index, const InputMatrixType &input, OutputMatrixType &&output) noexcept {
    if ((index + step_scale_index_offsets[IndexChecked]) % step_scale[IndexChecked] == 0) {
#if __BENCHMARK__ < 2
        network.template operator()<continueNetworkCalculationAfter<IndexChecked>>(input, std::forward<OutputMatrixType>(output), buffer, permanent);
#else
        network.template operator()<continueNetworkCalculationAfter<IndexChecked>>(input, std::forward<OutputMatrixType>(output), buffer, permanent, [&index]() { measure(index); });
#endif
    } else {
        if constexpr (IndexChecked > 0) {
            runWithStepScale<InputMatrixType, OutputMatrixType, IndexChecked - 1>(index, input, std::forward<OutputMatrixType>(output));
        }
    }
}

auto run_SEdge(pybind11::array_t<float> input) {
    std::memset(&permanent.data[0], 0, sizeof(permanent));
    std::memset(&buffer.data[0], 0, sizeof(buffer));

    convertToBaseMatrix<InputMatrixType, float>(input, input_matrix);

#if __OP_COUNTING__
    helpers::Benchmark::TypeInstance<float>::resetAll(); // Reset the benchmark counters
    printBenchmark<float>();
#endif
#if __BENCHMARK__ > 0
    auto start_time = std::chrono::high_resolution_clock::now();
#endif
#if __BENCHMARK__ == 2
    measure(-1); // Start measuring time
#endif

#pragma GCC unroll(1)
    for (Dim_size_t i = 0; i < sequence_length; i += sequence_length_sub) {
        // std::cout << "Running sub-sequence " << i << std::endl;
        const auto input_slice = permute<"BSC">(slice<"S", sequence_length_sub>(input_matrix, {i}));

        // Run the network
        runWithStepScale(i, input_slice, output_matrix);
    }

#if __BENCHMARK__ > 0
    std::cout << "Time to run the network: " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << " microseconds" << std::endl;
#endif
#if __OP_COUNTING__
    std::cout << "Operation Counting Enabled" << std::endl;
    printBenchmark<float>();
#endif

    return convertToNumpyArray<OutputMatrixType, float>(output_matrix);
}

void printModelInfo() {
    std::cout << "Network size: " << sizeof(NetworkType) << std::endl;
    std::cout << "Input Matrix Type: " << human_readable_type<InputMatrixType> << std::endl;
    std::cout << "Output Matrix Type: " << human_readable_type<OutputMatrixType> << std::endl;
    std::cout << "Input Slice Type: " << human_readable_type<SubInputMatrixType> << std::endl;
    std::cout << "Memory Permanent Size: " << memory_permanent << std::endl;
    std::cout << "Memory Buffer Size: " << memory_buffer << std::endl;

    std::cout << "Network Execution Schedule: " << std::endl;
    std::cout << NetworkType::template CurrentMemoryPlaning<SubInputMatrixType>::template memory_index_locations<sizeof(buffer), 0, 0> << std::endl;
    std::cout << "Step Scale: " << std::to_array(step_scale) << std::endl;

#if __OP_COUNTING__
    std::cout << "Operation Counting Enabled" << std::endl;
    const auto input_slice = slice<"S", sequence_length_sub>(input_matrix, {0});
    printBenchmark<float>();
    helpers::Benchmark::TypeInstance<float>::resetAll(); // Reset the benchmark counters
    network(input_slice, output_matrix, buffer, permanent);
    printBenchmark<float>();
#endif
}

int getInputSize() {
    return input_matrix.dimensions[1];
}

auto getStepScaleTImes() {
    std::array<int, sequence_length> step_scale_times=makeFilledArray<int, sequence_length>(-1); // Initialize all elements to -1
#if __BENCHMARK__ == 2
    for (unsigned int i = 1; i < sequence_length + 1; ++i) {
        step_scale_times[i - 1] = (int)std::chrono::duration_cast<std::chrono::nanoseconds>(layer_times[i] - layer_times[i - 1]).count();
    }
#endif
    pybind11::array_t<int> array(sequence_length);
    pybind11::buffer_info  info = array.request();

    int *data_ptr = reinterpret_cast<int *>(info.ptr);

    for (unsigned int i = 0; i < sequence_length; ++i) {
        data_ptr[i] = step_scale_times[i];
    }

    return array;
}

PYBIND11_MODULE(CppSEdge, m) {
    m.doc() = "Runs the Network";

    m.def("run", &run_SEdge, "Runs the network, takes input as a numpy array");
    m.def("printModelInfo", &printModelInfo, "Prints the model information");
    m.def("getInputSize", &getInputSize, "Returns the input size of the network");
    m.def("getStepScaleTimes", &getStepScaleTImes, "Returns the step scale times of the network");
}
