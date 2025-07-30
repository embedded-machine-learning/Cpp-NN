#pragma once

#include "../../Matrix.hpp"
#include "../../helpers/AccumulationTypes.hpp"

#include "../../helpers/Algorithm.hpp"

#include <type_traits>
#include <utility>

namespace functions {
namespace conv1d {
/*
The best configuration for the Conv1d Channel order is D3_Batch_Width_Channel
*/

/*
Simple Conv1d Layer, uses mostly passed memory
*/
template < // Template parameters starting with manual setting infere
        Dim_size_t stride,
        Dim_size_t padding,
        // template parameters auto infered
        typename InputMatrixType,
        typename OutputMatrixType,
        typename WeightMatrixType,
        typename InputMatrixTypePermuted  = typename InputMatrixType::template Permutation<DimensionOrder::D3_Batch_Width_Channel>,
        typename OutputMatrixTypePermuted = typename OutputMatrixType::template Permutation<DimensionOrder::D3_Batch_Width_Channel>,
        typename WeightMatrixTypePermuted = typename WeightMatrixType::template Permutation<DimensionOrder::D3_OutChannel_Kernel_InChannel>,
        Dim_size_t OutputChannels /*automatically inferred from Bias*/,
        typename AccumulationType /*automatically inferred from Bias*/,
        typename InputType       = typename InputMatrixType::type,
        typename WeightType      = typename WeightMatrixType::type,
        typename OutputType      = typename OutputMatrixType::type,
        Dim_size_t Batch         = InputMatrixTypePermuted::dim1,
        Dim_size_t InputWidth    = InputMatrixTypePermuted::dim2,
        Dim_size_t InputChannels = InputMatrixTypePermuted::dim3,
        Dim_size_t Kernel        = WeightMatrixTypePermuted::dim2,
        class Lambda,
        std::enable_if_t<WeightMatrixType::dims == 3, int> = 0,
        typename... ActivationInformation>
__attribute__((always_inline)) inline void Conv1d( // Function Parameters
        const InputMatrixType                                                      &Input,
        OutputMatrixType                                                           &out,
        const WeightMatrixType                                                     &Weights,
        const Matrix<AccumulationType, DimensionOrder::D1_Channel, OutputChannels> &Bias,
        const Lambda                                                               &Act,
        const Matrix<ActivationInformation, DimensionOrder::D1_Channel, OutputChannels> &...ActivationParameters) {

    static_assert(std::is_same_v<AccumulationType_helper<InputType, WeightType>, void> == false, "AccumulationType is not defined for the given Input and Weight types");
    static_assert(std::is_same_v<AccumulationType_helper<InputType, WeightType>, AccumulationType> == true, "Bias does not use recommended AccumulationType");

    static_assert(InputChannels == WeightMatrixTypePermuted::dim3, "InputChannels has to be the same as Weights dim3");
    static_assert(InputWidth + 2 * padding >= Kernel, "InputWidth+2*padding has to be greater or equal to Kernel");
    static_assert((InputWidth - Kernel + 2 * padding) % stride == 0, "Stride does not fit the Kernel and Padding");
    static_assert(OutputMatrixTypePermuted::dim1 == Batch, "Batch has to be the same as out.dim1");
    static_assert(OutputMatrixTypePermuted::dim2 == ((InputWidth - Kernel + 2 * padding) / stride + 1), "Output Width has to be the same as out.dim3");
    static_assert(OutputMatrixTypePermuted::dim3 == OutputChannels, "OutputChannels has to be the same as out.dim2");

    for (Dim_size_t batch = 0; batch < Batch; batch++) {
        for (Dim_size_t input_width = -padding; input_width < InputWidth - Kernel + 1 + padding; input_width += stride) {
            for (Dim_size_t output_channel = 0; output_channel < OutputChannels; output_channel++) {
                AccumulationType sum{Bias.data[output_channel]};
                for (Dim_size_t kernel = 0; kernel < Kernel; kernel++) {
                    if (input_width + kernel < 0 || input_width + kernel >= InputWidth)
                        continue;
                    else {
                        for (Dim_size_t input_channel = 0; input_channel < InputChannels; input_channel++) {
                            sum += static_cast<AccumulationType>(Input.template at<DimensionOrder::D3_Batch_Width_Channel>(batch, input_width + kernel, input_channel)) *
                                   static_cast<AccumulationType>(Weights.template at<DimensionOrder::D3_OutChannel_Kernel_InChannel>(output_channel, kernel, input_channel));
                        }
                    }
                }
                out.template at<DimensionOrder::D3_Batch_Width_Channel>(batch, (input_width + padding) / stride, output_channel) =
                        Lambda::template Act<AccumulationType, OutputType, ActivationInformation...>(sum, ActivationParameters.at(output_channel)...);
            }
        }
    }
}

// Weight unroll and parallelization constexpr
template <Dim_size_t KernelParallelSuggested,
          Dim_size_t UnrolledSuggested,
          typename InputMatrixType,
          typename InputMatrixTypePermuted = typename InputMatrixType::template Permutation<DimensionOrder::D3_OutChannel_Kernel_InChannel>,
          typename Type                    = typename InputMatrixTypePermuted::type,
          Dim_size_t OutputChannels        = InputMatrixTypePermuted::dim1,
          Dim_size_t Kernel                = InputMatrixTypePermuted::dim2,
          Dim_size_t InputChannels         = InputMatrixTypePermuted::dim3,
          Dim_size_t KernelParallel        = helpers::highest_restless_division_factor_up_to(OutputChannels, KernelParallelSuggested),
          Dim_size_t Unrolled              = helpers::highest_restless_division_factor_up_to(InputChannels, UnrolledSuggested),
          typename OutputMatrixType =
                  Matrix<Type, DimensionOrder::D5_OutChannel_Kernel_InChannel_KernelParallel_Unrolled, OutputChannels / KernelParallel, Kernel, InputChannels / Unrolled, KernelParallel, Unrolled>>
constexpr OutputMatrixType WeightUnrollParallel(const InputMatrixType &Weights) {
    static_assert(OutputChannels % KernelParallel == 0, "OutputChannels must be divisible by KernelParallel");
    static_assert(InputChannels % Unrolled == 0, "InputChannels must be divisible by Unrolled");

    OutputMatrixType WeightsUnrolled;

    for (Dim_size_t output_channel = 0; output_channel < OutputChannels / KernelParallel; output_channel++) {
        for (Dim_size_t kernel = 0; kernel < Kernel; kernel++) {
            for (Dim_size_t input_channel = 0; input_channel < InputChannels / Unrolled; input_channel++) {
                for (Dim_size_t kernel_parallel = 0; kernel_parallel < KernelParallel; kernel_parallel++) {
                    for (Dim_size_t unrolled = 0; unrolled < Unrolled; unrolled++) {
                        WeightsUnrolled.data[output_channel][kernel][input_channel][kernel_parallel][unrolled] =
                                Weights.template at<DimensionOrder::D3_OutChannel_Kernel_InChannel>(output_channel * KernelParallel + kernel_parallel, kernel, input_channel * Unrolled + unrolled);
                    }
                }
            }
        }
    }
    return WeightsUnrolled;
}

/*
Simple Conv1d Layer, uses mostly passed memory, uses unrolled and parallelized weights
*/
template < // Template parameters starting with manual setting infere
        Dim_size_t stride,
        Dim_size_t padding,
        // template parameters auto infered
        typename InputMatrixType,
        typename OutputMatrixType,
        typename InputMatrixTypePermuted  = typename InputMatrixType::template Permutation<DimensionOrder::D3_Batch_Width_Channel>,
        typename OutputMatrixTypePermuted = typename OutputMatrixType::template Permutation<DimensionOrder::D3_Batch_Width_Channel>,
        Dim_size_t OutputChannels, /*automatically inferred from Bias*/
        typename AccumulationType, /*automatically inferred from Bias*/
        typename WeightType,       /*automatically inferred from Weights*/
        Dim_size_t Kernel,         /*automatically inferred from Weights*/
        Dim_size_t KernelParallel, /*automatically inferred from Weights*/
        Dim_size_t Unrolled,       /*automatically inferred from Weights*/
        typename InputType       = typename InputMatrixType::type,
        typename OutputType      = typename OutputMatrixType::type,
        Dim_size_t Batch         = InputMatrixTypePermuted::dim1,
        Dim_size_t InputWidth    = InputMatrixTypePermuted::dim2,
        Dim_size_t InputChannels = InputMatrixTypePermuted::dim3,
        class Lambda,
        size_t... KernelIndexes,
        size_t... UnrollIndexes,
        typename... ActivationInformation>
__attribute__((always_inline)) inline void Conv1d( // Function Parameters
        const InputMatrixType &Input,
        OutputMatrixType      &out,
        const Matrix<WeightType, DimensionOrder::D5_OutChannel_Kernel_InChannel_KernelParallel_Unrolled, OutputChannels / KernelParallel, Kernel, InputChannels / Unrolled, KernelParallel, Unrolled>
                                                                                   &Weights,
        const Matrix<AccumulationType, DimensionOrder::D1_Channel, OutputChannels> &Bias,
        std::index_sequence<KernelIndexes...>,
        std::index_sequence<UnrollIndexes...>,
        const Lambda &Act,
        const Matrix<ActivationInformation, DimensionOrder::D1_Channel, OutputChannels> &...ActivationParameters) {

    static_assert(std::is_same_v<AccumulationType_helper<InputType, WeightType>, void> == false, "AccumulationType is not defined for the given Input and Weight types");
    static_assert(std::is_same_v<AccumulationType_helper<InputType, WeightType>, AccumulationType> == true, "Bias does not use recommended AccumulationType");

    static_assert(InputWidth + 2 * padding >= Kernel, "InputWidth+2*padding has to be greater or equal to Kernel");
    static_assert((InputWidth - Kernel + 2 * padding) % stride == 0, "Stride does not fit the Kernel and Padding");
    static_assert(OutputMatrixTypePermuted::dim1 == Batch, "Batch has to be the same as out.dim1");
    static_assert(OutputMatrixTypePermuted::dim2 == ((InputWidth - Kernel + 2 * padding) / stride + 1), "Output Width has to be the same as out.dim3");
    static_assert(OutputMatrixTypePermuted::dim3 == OutputChannels, "OutputChannels has to be the same as out.dim2");

#pragma GCC unroll(1)
    for (Dim_size_t batch = 0; batch < Batch; batch++) {
#pragma GCC unroll(1)
        for (Dim_size_t input_width = -padding; input_width < InputWidth - Kernel + 1 + padding; input_width += stride) {
#pragma GCC unroll(1)
            for (Dim_size_t output_channel = 0; output_channel < OutputChannels / KernelParallel; output_channel++) {
                AccumulationType sum[KernelParallel]{Bias.at(output_channel * KernelParallel + KernelIndexes)...};

                //                 auto start = (input_width >= 0) ? 0 : -input_width;
                //                 auto end   = (input_width + Kernel < InputWidth) ? Kernel : InputWidth - input_width;
                // #pragma GCC unroll(1)
                //                 for (Dim_size_t kernel = start; kernel < end; kernel++) {
#pragma GCC unroll(1)
                for (Dim_size_t kernel = 0; kernel < Kernel; kernel++) {
                    if (input_width + kernel < 0 || input_width + kernel >= InputWidth)
                        continue;
                    else {
#pragma GCC unroll(1)
                        for (Dim_size_t input_channel = 0; input_channel < InputChannels / Unrolled; input_channel++) {
                            const InputType input[Unrolled]{Input.template at<DimensionOrder::D3_Batch_Width_Channel>(batch, input_width + kernel, input_channel*Unrolled + UnrollIndexes)...};
#pragma GCC unroll(65534) // 65534 is the maximum unroll value for GCC, so full unroll
                            for (Dim_size_t kernel_parallel = 0; kernel_parallel < KernelParallel; kernel_parallel++) {
                                const WeightType weights[Unrolled]{Weights.template at<DimensionOrder::D5_OutChannel_Kernel_InChannel_KernelParallel_Unrolled>(output_channel, kernel, input_channel,
                                                                                                                                                               kernel_parallel, UnrollIndexes)...};
#pragma GCC unroll(65534)
                                for (Dim_size_t unrolled = 0; unrolled < Unrolled; unrolled++) {
                                    sum[kernel_parallel] += static_cast<AccumulationType>(input[unrolled]) * static_cast<AccumulationType>(weights[unrolled]);
                                }
                            }
                        }
                    }
                }
#pragma GCC unroll(65534)
                for (Dim_size_t kernel_parallel = 0; kernel_parallel < KernelParallel; kernel_parallel++) {
                    out.template at<DimensionOrder::D3_Batch_Width_Channel>(batch, (input_width + padding) / stride, output_channel* KernelParallel + kernel_parallel) =
                            Lambda::template Act<AccumulationType, OutputType, ActivationInformation...>(sum[kernel_parallel], ActivationParameters.at(output_channel* KernelParallel + kernel_parallel)...);
                }
            }
        }
    }
}

// Wrapper for the unrolled and parallelized Conv1d function, to automatically infer KernelParallel and Unrolled
template <Dim_size_t stride,
          Dim_size_t padding,
          typename InputMatrixType,
          typename WeightMatrixType,
          typename BiasMatrixType,
          typename OutputMatrixType,
          class Lambda,
          typename... ActivationInformation,
          std::enable_if_t<WeightMatrixType::dims == 5, int> = 0>
__attribute__((always_inline)) inline void Conv1d(
        const InputMatrixType &Input, OutputMatrixType &out, const WeightMatrixType &Weights, const BiasMatrixType &Bias, const Lambda &Act, const ActivationInformation &...ActivationParameters) {
    constexpr auto KernelParallel = WeightMatrixType::dim4;
    constexpr auto Unrolled       = WeightMatrixType::dim5;
    Conv1d<stride, padding>(Input, out, Weights, Bias, std::make_index_sequence<KernelParallel>{}, std::make_index_sequence<Unrolled>{}, Act, ActivationParameters...);
}

/*
Simple Conv1d Layer, uses stack memory,
wrapps the Conv1d function
*/
template <Dim_size_t stride,
          Dim_size_t padding,
          typename OutputType       = float,
          typename InputType        = float,
          typename WeightType       = float,
          typename AccumulationType = AccumulationType_helper<InputType, WeightType>,
          typename InputMatrixType,
          typename InputMatrixTypePermuted = typename InputMatrixType::template Permutation<DimensionOrder::D3_Batch_Channel_Width>,
          Dim_size_t Batch                 = InputMatrixTypePermuted::dim1, // Input batch
          Dim_size_t InputChannels         = InputMatrixTypePermuted::dim2, // Input Channels
          Dim_size_t InputWidth            = InputMatrixTypePermuted::dim3, // Input Width
          Dim_size_t OutputChannels,                                        // Weight Output Channels
          // Dim_size_t M2_2,	// Weight Input Channels has to be the same as InputChannels
          Dim_size_t Kernel, // Weight Kernel shape
          class Lambda,
          //   Matrix<InputType, DimensionOrder::D3_Batch_Channel_Width, Batch, InputChannels, InputWidth> _ = InputMatrixType(),
          typename... ActivationInformation>
__attribute__((always_inline)) inline Matrix<OutputType, DimensionOrder::D3_Batch_Channel_Width, Batch, OutputChannels, ((InputWidth - Kernel + 2 * padding) / stride + 1)> Conv1d(
        const InputMatrixType                                                                                           &Input,
        const Matrix<WeightType, DimensionOrder::D3_OutChannel_InChannel_Kernel, OutputChannels, InputChannels, Kernel> &Weights,
        const Matrix<AccumulationType, DimensionOrder::D1_Channel, OutputChannels>                                      &Bias,
        const Lambda                                                                                                    &Act,
        const Matrix<ActivationInformation, DimensionOrder::D1_Channel, OutputChannels> &...ActivationParameters) {
    Matrix<OutputType, DimensionOrder::D3_Batch_Channel_Width, Batch, OutputChannels, ((InputWidth - Kernel + 2 * padding) / stride + 1)> out;
    Conv1d<stride, padding>(Input, out, Weights, Bias, Act, ActivationParameters...);
    return out;
}

} // namespace conv1d
} // namespace functions