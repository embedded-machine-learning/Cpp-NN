#include <chrono>
#include <fstream>
#include <iostream>
#include <tuple>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "include/numpy_conversion.hpp"

#include "include/Matrix.hpp"
#include "include/MatrixOperations.hpp"
#include "include/functions/inference/ActivationFunctions.hpp"
#include "include/helpers/Benchmark.hpp"
#include "include/helpers/Complex.hpp"
#include "include/helpers/PrintTuple.hpp"
#include "include/helpers/TestHelpers.hpp"
#include "include/layers/S5.hpp"
#include "include/layers/Sequential.hpp"

#define double helpers::Benchmark::TypeInstance<double>
#define float  helpers::Benchmark::TypeInstance<float>
// #define float  float
// #define double double

// #include "weights.hpp"

constexpr Dim_size_t InputChannel = 1;
// constexpr Dim_size_t States      = 1;
constexpr Dim_size_t Sequence = 16000;
// constexpr Dim_size_t Sequence = 1;
constexpr Dim_size_t BatchSize = 1;

using BaseType = float;

#include "network.hpp"

using InputType = Matrix<BaseType, DimensionOrder::D3_Batch_Sequence_Channel, BatchSize, Sequence, InputChannel>;
using OutType   = typename decltype(Seq)::OutputMatrix<InputType>;

constexpr Dim_size_t OutputChannel = OutType::template Permutation<DimensionOrder::D3_Batch_Sequence_Channel>::dim3;

using InputSliceType         = Matrix<typename InputType::type, DimensionOrder::D3_Batch_Sequence_Channel, BatchSize, 1, InputChannel>;
using OutputSliceType        = Matrix<typename OutType::type, DimensionOrder::D3_Batch_Sequence_Channel, BatchSize, 1, OutputChannel>;
using OutputAccumulationType = Matrix<typename OutType::type, DimensionOrder::D2_Batch_Channel, BatchSize, OutputChannel>;
using TrueOutputType         = typename decltype(Decoder)::OutputMatrix<OutputAccumulationType>;

InputType              InputBuffer{};
OutType                OutputBuffer{};
OutputAccumulationType OutputAccumulationBuffer{};
TrueOutputType         TrueOutputBuffer{};

Matrix<char, DimensionOrder::ERROR, Seq.MemoryMinimal<InputSliceType>>   BufferMemory{};
Matrix<char, DimensionOrder::ERROR, Seq.MemoryPermanent<InputSliceType>> PermanentMemory{};

// static_assert(sizeof(InputType) <= 850000, "Input size is too large");
// static_assert(sizeof(OutType) <= 850000, "Output size is too large");
//

#undef double
#undef float

template <size_t StopPosition>
using UpdateToLayer = SequencesConcatenate<SequenceRepeater<StopPosition - 1, bool, true>, SequenceRepeater<Seq.number_of_layers() - StopPosition + 1, bool, false>>;

// #define print_stats true

auto run_S5(pybind11::array_t<float> input) {
#ifdef print_stats
    std::cout << "--------------------------------------------------------------------------------------" << std::endl;
    std::cout << "InputType:              " << TypeName<InputType> << std::endl;
    std::cout << "OutType:                " << TypeName<OutType> << std::endl;
    std::cout << "InputSliceType:         " << TypeName<InputSliceType> << std::endl;
    std::cout << "OutputSliceType:        " << TypeName<OutputSliceType> << std::endl;
    std::cout << "OutputAccumulationType: " << TypeName<OutputAccumulationType> << std::endl;
    std::cout << "TrueOutputType:         " << TypeName<TrueOutputType> << std::endl;
    std::cout << "--------------------------------------------------------------------------------------" << std::endl;
    std::cout << "Permanent Memory Required:   " << Seq.MemoryPermanent<InputType> + sizeof(OutputAccumulationBuffer) << " Byte" << std::endl;
    std::cout << "Permanent Memory Provided:   " << sizeof(PermanentMemory) + sizeof(OutputAccumulationBuffer) << " Byte" << std::endl;

    std::cout << "Buffer Memory Required:      " << Seq.MemoryMinimal<InputSliceType> << " Byte" << std::endl;
    std::cout << "Buffer Memory Provided:      " << sizeof(BufferMemory) << " Byte" << std::endl;

    std::cout << "Size of Network:             " << sizeof(Seq) << " Byte" << std::endl;
    std::cout << "Size of Decoder:             " << sizeof(Decoder) << " Byte" << std::endl;

    helpers::Benchmark::TypeInstance<float>::resetAll();
    std::cout << "--------------------------------------------------------------------------------------" << std::endl;
    std::cout << "Pre network run" << std::endl;
    std::cout << "counted_multiplications :                      " << helpers::Benchmark::TypeInstance<float>::counted_multiplications << std::endl;
    std::cout << "counted_additions :                            " << helpers::Benchmark::TypeInstance<float>::counted_additions << std::endl;
    std::cout << "counted_divisions :                            " << helpers::Benchmark::TypeInstance<float>::counted_divisions << std::endl;
    std::cout << "counted_subtractions :                         " << helpers::Benchmark::TypeInstance<float>::counted_subtractions << std::endl;
    std::cout << "counted_comparisons :                          " << helpers::Benchmark::TypeInstance<float>::counted_comparisons << std::endl;
    std::cout << "counted_extractions :                          " << helpers::Benchmark::TypeInstance<float>::counted_extractions << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
#endif

    memset(PermanentMemory.data, 0, sizeof(PermanentMemory));
    memset(OutputAccumulationBuffer.data, 0, sizeof(OutputAccumulationBuffer));

    try {
        array_to_Matrix<BaseType, float>(input, InputBuffer);
        InputSliceType  InputSlice{};
        OutputSliceType OutputSlice{};

        for (int timeStep = 0; timeStep < Sequence; timeStep++) {
            // Convert Input to a time slice
            for (int batch = 0; batch < BatchSize; batch++)
                for (int channel = 0; channel < InputChannel; channel++)
                    InputSlice.template at<DimensionOrder::D3_Batch_Sequence_Channel>(batch, 0, channel) = InputBuffer.template at<DimensionOrder::D3_Batch_Sequence_Channel>(batch, timeStep, channel);

            // S5 Layers
            // Seq(InputSlice, OutputSlice, BufferMemory, PermanentMemory);

            if (timeStep % step_scale[5] == 0) {
                Seq(InputSlice, OutputSlice, BufferMemory, PermanentMemory);
                for (int batch = 0; batch < BatchSize; batch++)
                    for (int channel = 0; channel < OutputChannel; channel++)
                        OutputAccumulationBuffer.template at<DimensionOrder::D2_Batch_Channel>(batch, channel) += OutputSlice.template at<DimensionOrder::D3_Batch_Sequence_Channel>(batch, 0, channel);
            } else if (timeStep % step_scale[4] == 0)
                Seq.template partialExecution<UpdateToLayer<5>>(InputSlice, OutputSlice, BufferMemory, PermanentMemory);
            else if (timeStep % step_scale[3] == 0)
                Seq.template partialExecution<UpdateToLayer<4>>(InputSlice, OutputSlice, BufferMemory, PermanentMemory);
            else if (timeStep % step_scale[2] == 0)
                Seq.template partialExecution<UpdateToLayer<3>>(InputSlice, OutputSlice, BufferMemory, PermanentMemory);
            else if (timeStep % step_scale[1] == 0)
                Seq.template partialExecution<UpdateToLayer<2>>(InputSlice, OutputSlice, BufferMemory, PermanentMemory);
            else if (timeStep % step_scale[0] == 0)
                Seq.template partialExecution<UpdateToLayer<1>>(InputSlice, OutputSlice, BufferMemory, PermanentMemory);

            // if (timeStep % step_scale[2] == 0){
            //     Seq(InputSlice, OutputSlice, BufferMemory, PermanentMemory);
            //     for (int batch = 0; batch < BatchSize; batch++)
            //         for (int channel = 0; channel < OutputChannel; channel++)
            //             OutputAccumulationBuffer.template at<DimensionOrder::D2_Batch_Channel>(batch, channel) +=
            //                     OutputSlice.template at<DimensionOrder::D3_Batch_Sequence_Channel>(batch, 0, channel);
            // }
            // else if (timeStep % step_scale[1] == 0)
            //     Seq.template partialExecution<UpdateToLayer<2>>(InputSlice, OutputSlice, BufferMemory, PermanentMemory);
            // else if (timeStep % step_scale[0] == 0)
            //     Seq.template partialExecution<UpdateToLayer<1>>(InputSlice, OutputSlice, BufferMemory, PermanentMemory);

            // Write out of the full last S5 Layer ( Debugging)
            // for (int batch = 0; batch < BatchSize; batch++)
            //     for (int channel = 0; channel < OutputChannel; channel++)
            //         OutputBuffer.template at<DimensionOrder::D3_Batch_Sequence_Channel>(batch, timeStep, channel) =
            //                 OutputSlice.template at<DimensionOrder::D3_Batch_Sequence_Channel>(batch, 0, channel);
        }

        Decoder(OutputAccumulationBuffer, TrueOutputBuffer, BufferMemory);

#ifdef print_stats
        auto stop = std::chrono::high_resolution_clock::now();

        std::cout << "--------------------------------------------------------------------------------------" << std::endl;
        std::cout << "Post network run" << std::endl;
        std::cout << "Duration:                                  " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << " µs" << std::endl;
        std::cout << "Number of time steps:                      " << Sequence << std::endl;

        std::cout << "counted_multiplications :                  " << helpers::Benchmark::TypeInstance<float>::counted_multiplications << std::endl;
        std::cout << "counted_additions :                        " << helpers::Benchmark::TypeInstance<float>::counted_additions << std::endl;
        std::cout << "counted_divisions :                        " << helpers::Benchmark::TypeInstance<float>::counted_divisions << std::endl;
        std::cout << "counted_subtractions :                     " << helpers::Benchmark::TypeInstance<float>::counted_subtractions << std::endl;
        std::cout << "counted_comparisons :                      " << helpers::Benchmark::TypeInstance<float>::counted_comparisons << std::endl;
        std::cout << "counted_extractions :                      " << helpers::Benchmark::TypeInstance<float>::counted_extractions << std::endl;
        std::cout << "--------------------------------------------------------------------------------------" << std::endl;
        std::cout << "average multiplications per time step :    " << ((double)helpers::Benchmark::TypeInstance<float>::counted_multiplications) / Sequence << std::endl;
        std::cout << "average additions       per time step :    " << ((double)helpers::Benchmark::TypeInstance<float>::counted_additions) / Sequence << std::endl;
        std::cout << "average divisions       per time step :    " << ((double)helpers::Benchmark::TypeInstance<float>::counted_divisions) / Sequence << std::endl;
        std::cout << "average subtractions    per time step :    " << ((double)helpers::Benchmark::TypeInstance<float>::counted_subtractions) / Sequence << std::endl;
        std::cout << "average comparisons     per time step :    " << ((double)helpers::Benchmark::TypeInstance<float>::counted_comparisons) / Sequence << std::endl;
        std::cout << "--------------------------------------------------------------------------------------" << std::endl;
#endif
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
    }
    // const auto OutputBuffer_numpy = Matrix_to_array<float>(OutputBuffer);
    // const auto TrueOutputBuffer_numpy = Matrix_to_array<float>(TrueOutputBuffer);

    // return std::make_tuple(Matrix_to_array<float>(OutputBuffer), Matrix_to_array<float>(TrueOutputBuffer));
    return Matrix_to_array<float>(TrueOutputBuffer);
}

PYBIND11_MODULE(CppS5, m) {
    m.doc() = "Runs the S5 layer";

    m.def("run_S5", &run_S5, "Runs the S5 layer with 10 time steps, 10 input channels and 10 output channels and float as the data type");
}