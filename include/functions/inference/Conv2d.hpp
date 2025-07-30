#pragma once

#include "../../Matrix.hpp"
#include "../../helpers/AccumulationTypes.hpp"
#include "../../helpers/Algorithm.hpp"

namespace functions {
namespace conv2d {
/*
Simple Conv2d Layer, uses mostly passed memory
*/
template < // Template parameters starting with manual setting infere
        Dim_size_t stride,
        Dim_size_t padding,
        // template parameters auto infered
        typename InputMatrixType,
        typename OutputMatrixType,
        typename WeightMatrixType,
        typename InputMatrixTypePermuted  = typename InputMatrixType::template Permutation<DimensionOrder::D4_Batch_Width_Height_Channel>,
        typename OutputMatrixTypePermuted = typename OutputMatrixType::template Permutation<DimensionOrder::D4_Batch_Width_Height_Channel>,
        typename WeightMatrixTypePermuted = typename WeightMatrixType::template Permutation<DimensionOrder::D4_OutChannel_InChannel_KernelWidth_KernelHeight>, // suboptimal
        Dim_size_t OutputChannels /*automatically inferred from Bias*/,
        typename AccumulationType /*automatically inferred from Bias*/,
        typename InputType       = typename InputMatrixType::type,
        typename WeightType      = typename WeightMatrixType::type,
        typename OutputType      = typename OutputMatrixType::type,
        Dim_size_t Batch         = InputMatrixTypePermuted::dim1,
        Dim_size_t InputWidth    = InputMatrixTypePermuted::dim2,
        Dim_size_t InputHeight   = InputMatrixTypePermuted::dim3,
        Dim_size_t InputChannels = InputMatrixTypePermuted::dim4,
        Dim_size_t KernelWidth   = WeightMatrixTypePermuted::dim3,
        Dim_size_t KernelHeight  = WeightMatrixTypePermuted::dim4,
        class Lambda,
        std::enable_if_t<WeightMatrixType::dims == 4, int> = 0,
        typename... ActivationInformation>
__attribute__((always_inline)) inline void Conv2d( // Function Parameters
        const InputMatrixType                                                      &Input,
        OutputMatrixType                                                           &out,
        const WeightMatrixType                                                     &Weights,
        const Matrix<AccumulationType, DimensionOrder::D1_Channel, OutputChannels> &Bias,
        const Lambda                                                               &Act,
        const Matrix<ActivationInformation, DimensionOrder::D1_Channel, OutputChannels> &...ActivationParameters) {

    static_assert(std::is_same_v<AccumulationType_helper<InputType, WeightType>, void> == false, "AccumulationType is not defined for the given Input and Weight types");
    static_assert(std::is_same_v<AccumulationType_helper<InputType, WeightType>, AccumulationType> == true, "Bias does not use recommended AccumulationType");

    static_assert(InputChannels == WeightMatrixTypePermuted::dim2, "InputChannels has to be the same as WeightMatrixType::dim2");
    static_assert(InputWidth + 2 * padding >= KernelWidth, "InputWidth + 2*padding has to be greater or equal to KernelWidth");
    static_assert(InputHeight + 2 * padding >= KernelHeight, "InputHeight + 2*padding has to be greater or equal to KernelHeight");
    static_assert(OutputMatrixTypePermuted::dim1 == Batch, "OutputMatrixType::dim1 has to be the same as InputMatrixType::dim1");
    static_assert(OutputMatrixTypePermuted::dim2 == (InputWidth - KernelWidth + 2 * padding) / stride + 1, "OutputMatrixType::dim2 has to be (InputWidth - KernelWidth + 2 * padding) / stride + 1");
    static_assert(OutputMatrixTypePermuted::dim3 == (InputHeight - KernelHeight + 2 * padding) / stride + 1,
                  "OutputMatrixType::dim3 has to be (InputHeight - KernelHeight + 2 * padding) / stride + 1");
    static_assert(OutputMatrixTypePermuted::dim4 == OutputChannels, "OutputMatrixType::dim4 has to be OutputChannels");

    for (Dim_size_t batch = 0; batch < Batch; batch++) {
        for (Dim_size_t output_channel = 0; output_channel < OutputChannels; output_channel++) {
            for (Dim_size_t input_width = -padding; input_width < InputWidth - KernelWidth + 1 + padding; input_width += stride) {
                for (Dim_size_t input_height = -padding; input_height < InputHeight - KernelHeight + 1 + padding; input_height += stride) {
                    // Preload the bias
                    AccumulationType sum{Bias.at(output_channel)};
                    // The convolution step
                    for (Dim_size_t kernel_width = 0; kernel_width < KernelWidth; kernel_width++) {
                        for (Dim_size_t kernel_height = 0; kernel_height < KernelHeight; kernel_height++) {
                            if (input_width + kernel_width < 0 || input_width + kernel_width >= InputWidth || input_height + kernel_height < 0 || input_height + kernel_height >= InputHeight)
                                continue;
                            else {
                                for (Dim_size_t input_channel = 0; input_channel < InputChannels; input_channel++) {
                                    sum += static_cast<AccumulationType>(
                                                   Input.template at<DimensionOrder::D4_Batch_Width_Height_Channel>(batch, input_width + kernel_width, input_height + kernel_height, input_channel)) *
                                           static_cast<AccumulationType>(
                                                   Weights.template at<DimensionOrder::D4_OutChannel_InChannel_KernelWidth_KernelHeight>(output_channel, input_channel, kernel_width, kernel_height));
                                }
                            }
                        }
                    }
                    out.template at<DimensionOrder::D4_Batch_Width_Height_Channel>(batch, (input_width + padding) / stride, (input_height + padding) / stride, output_channel) =
                            Lambda::template Act<AccumulationType, OutputType, ActivationInformation...>(sum, ActivationParameters.at(output_channel)...);
                }
            }
        }
    }
}

// Weight unroll and parallelization constexpr
template <Dim_size_t KernelParallelSuggested,
          Dim_size_t UnrolledSuggested,
          typename InputMatrixType,
          typename InputMatrixTypePermuted = typename InputMatrixType::template Permutation<DimensionOrder::D4_OutChannel_KernelWidth_KernelHeight_InChannel>,
          typename Type                    = typename InputMatrixTypePermuted::type,
          Dim_size_t OutputChannels        = InputMatrixTypePermuted::dim1,
          Dim_size_t KernelWidth           = InputMatrixTypePermuted::dim2,
          Dim_size_t KernelHeight          = InputMatrixTypePermuted::dim3,
          Dim_size_t InputChannels         = InputMatrixTypePermuted::dim4,
          Dim_size_t KernelParallel        = helpers::highest_restless_division_factor_up_to(OutputChannels, KernelParallelSuggested),
          Dim_size_t Unrolled              = helpers::highest_restless_division_factor_up_to(InputChannels, UnrolledSuggested),
          typename OutputMatrixType        = Matrix<Type,
                                                    DimensionOrder::D6_OutChannel_KernelWidth_KernelHeight_InChannel_KernelParallel_Unrolled,
                                                    OutputChannels / KernelParallel,
                                                    KernelWidth,
                                                    KernelHeight,
                                                    InputChannels / Unrolled,
                                                    KernelParallel,
                                                    Unrolled>>
constexpr OutputMatrixType WeightUnrollParallel(const InputMatrixType &Weights) {
    static_assert(OutputChannels % KernelParallel == 0, "OutputChannels must be divisible by KernelParallel");
    static_assert(InputChannels % Unrolled == 0, "InputChannels must be divisible by Unrolled");

    OutputMatrixType WeightsUnrolled;

    for (Dim_size_t output_channel = 0; output_channel < OutputChannels / KernelParallel; output_channel++) {
        for (Dim_size_t kernel_width = 0; kernel_width < KernelWidth; kernel_width++) {
            for (Dim_size_t kernel_height = 0; kernel_height < KernelHeight; kernel_height++) {
                for (Dim_size_t input_channel = 0; input_channel < InputChannels / Unrolled; input_channel++) {
                    for (Dim_size_t kernel_parallel = 0; kernel_parallel < KernelParallel; kernel_parallel++) {
                        for (Dim_size_t unrolled = 0; unrolled < Unrolled; unrolled++) {
                            WeightsUnrolled.data[output_channel][kernel_width][kernel_height][input_channel][kernel_parallel][unrolled] =
                                    Weights.template at<DimensionOrder::D4_OutChannel_KernelWidth_KernelHeight_InChannel>(output_channel * KernelParallel + kernel_parallel, kernel_width,
                                                                                                                          kernel_height, input_channel * Unrolled + unrolled);
                        }
                    }
                }
            }
        }
    }
    return WeightsUnrolled;
}

/*
Simple Conv2d Layer, unrolled
*/
template < // Template parameters starting with manual setting infere
        Dim_size_t stride,
        Dim_size_t padding,
        // template parameters auto infered
        typename InputMatrixType,
        typename OutputMatrixType,
        typename WeightMatrixType,
        typename InputMatrixTypePermuted  = typename InputMatrixType::template Permutation<DimensionOrder::D4_Batch_Width_Height_Channel>,
        typename OutputMatrixTypePermuted = typename OutputMatrixType::template Permutation<DimensionOrder::D4_Batch_Width_Height_Channel>,
        Dim_size_t OutputChannels /*automatically inferred from Bias*/,
        typename AccumulationType /*automatically inferred from Bias*/,
        Dim_size_t KernelWidth,    /*automatically inferred from Weights*/
        Dim_size_t KernelHeight,   /*automatically inferred from Weights*/
        Dim_size_t KernelParallel, /*automatically inferred from Weights*/
        Dim_size_t Unrolled,       /*automatically inferred from Weights*/
        typename InputType       = typename InputMatrixType::type,
        typename WeightType      = typename WeightMatrixType::type,
        typename OutputType      = typename OutputMatrixType::type,
        Dim_size_t Batch         = InputMatrixTypePermuted::dim1,
        Dim_size_t InputWidth    = InputMatrixTypePermuted::dim2,
        Dim_size_t InputHeight   = InputMatrixTypePermuted::dim3,
        Dim_size_t InputChannels = InputMatrixTypePermuted::dim4,
        class Lambda,
        size_t... KernelIndexes,
        size_t... UnrollIndexes,
        typename... ActivationInformation>
__attribute__((always_inline)) inline void Conv2d( // Function Parameters
        const InputMatrixType                                                      &Input,
        OutputMatrixType                                                           &out,
        const Matrix<WeightType,
                     DimensionOrder::D6_OutChannel_KernelWidth_KernelHeight_InChannel_KernelParallel_Unrolled,
                     OutputChannels / KernelParallel,
                     KernelWidth,
                     KernelHeight,
                     InputChannels / Unrolled,
                     KernelParallel,
                     Unrolled>                                                     &Weights,
        const Matrix<AccumulationType, DimensionOrder::D1_Channel, OutputChannels> &Bias,
        std::index_sequence<KernelIndexes...>,
        std::index_sequence<UnrollIndexes...>,
        const Lambda &Act,
        const Matrix<ActivationInformation, DimensionOrder::D1_Channel, OutputChannels> &...ActivationParameters) {

    static_assert(std::is_same_v<AccumulationType_helper<InputType, WeightType>, void> == false, "AccumulationType is not defined for the given Input and Weight types");
    static_assert(std::is_same_v<AccumulationType_helper<InputType, WeightType>, AccumulationType> == true, "Bias does not use recommended AccumulationType");

    static_assert(InputWidth + 2 * padding >= KernelWidth, "InputWidth + 2*padding has to be greater or equal to KernelWidth");
    static_assert(InputHeight + 2 * padding >= KernelHeight, "InputHeight + 2*padding has to be greater or equal to KernelHeight");
    static_assert(OutputMatrixTypePermuted::dim1 == Batch, "OutputMatrixType::dim1 has to be the same as InputMatrixType::dim1");
    static_assert(OutputMatrixTypePermuted::dim2 == (InputWidth - KernelWidth + 2 * padding) / stride + 1, "OutputMatrixType::dim2 has to be (InputWidth - KernelWidth + 2 * padding) / stride + 1");
    static_assert(OutputMatrixTypePermuted::dim3 == (InputHeight - KernelHeight + 2 * padding) / stride + 1,
                  "OutputMatrixType::dim3 has to be (InputHeight - KernelHeight + 2 * padding) / stride + 1");
    static_assert(OutputMatrixTypePermuted::dim4 == OutputChannels, "OutputMatrixType::dim4 has to be OutputChannels");

#pragma GCC unroll(1)
    for (Dim_size_t batch = 0; batch < Batch; batch++) {
#pragma GCC unroll(1)
        for (Dim_size_t input_width = -padding; input_width < InputWidth - KernelWidth + 1 + padding; input_width += stride) {
#pragma GCC unroll(1)
            for (Dim_size_t input_height = -padding; input_height < InputHeight - KernelHeight + 1 + padding; input_height += stride) {
#pragma GCC unroll(1)
                for (Dim_size_t output_channel = 0; output_channel < OutputChannels / KernelParallel; output_channel++) {
                    // Preload the bias
                    AccumulationType sum[KernelParallel]{Bias.at(output_channel * KernelParallel + KernelIndexes)...};
                    // The convolution step
#pragma GCC unroll(1)
                    for (Dim_size_t kernel_width = 0; kernel_width < KernelWidth; kernel_width++) {
#pragma GCC unroll(1)
                        for (Dim_size_t kernel_height = 0; kernel_height < KernelHeight; kernel_height++) {
                            if (input_width + kernel_width < 0 || input_width + kernel_width >= InputWidth || input_height + kernel_height < 0 || input_height + kernel_height >= InputHeight)
                                continue;
                            else {
#pragma GCC unroll(1)
                                for (Dim_size_t input_channel = 0; input_channel < InputChannels / Unrolled; input_channel++) {
                                    const InputType input[Unrolled]{Input.template at<DimensionOrder::D4_Batch_Width_Height_Channel>(batch, input_width + kernel_width, input_height + kernel_height,
                                                                                                                                     input_channel * Unrolled + UnrollIndexes)...};
#pragma GCC unroll(65534) // 65534 is the maximum unroll value for GCC, so full unroll
                                    for (Dim_size_t kernel_parallel = 0; kernel_parallel < KernelParallel; kernel_parallel++) {
                                        const WeightType WeightUnrollParallel[Unrolled]{Weights.template at<DimensionOrder::D6_OutChannel_KernelWidth_KernelHeight_InChannel_KernelParallel_Unrolled>(
                                                output_channel * KernelParallel + kernel_parallel, kernel_width, kernel_height, input_channel * Unrolled, kernel_parallel, UnrollIndexes)...};
#pragma GCC unroll(65534)
                                        for (Dim_size_t unrolled = 0; unrolled < Unrolled; unrolled++) {
                                            sum[kernel_parallel] += static_cast<AccumulationType>(input[unrolled]) * static_cast<AccumulationType>(WeightUnrollParallel[unrolled]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    for (Dim_size_t kernel_parallel = 0; kernel_parallel < KernelParallel; kernel_parallel++) {
                        out.template at<DimensionOrder::D4_Batch_Width_Height_Channel>(batch, (input_width + padding) / stride, (input_height + padding) / stride,
                                                                                       output_channel * KernelParallel + kernel_parallel) =
                                Lambda::template Act<AccumulationType, OutputType, ActivationInformation...>(sum[kernel_parallel], ActivationParameters.at(output_channel)...);
                    }
                }
            }
        }
    }
}

// Wrapper for the unrolled and parallelized Conv2d function, to automatically infer KernelParallel and Unrolled
template <Dim_size_t stride,
          Dim_size_t padding,
          typename InputMatrixType,
          typename WeightMatrixType,
          typename BiasMatrixType,
          typename OutputMatrixType,
          class Lambda,
          typename... ActivationInformation,
          std::enable_if_t<WeightMatrixType::dims == 6, int> = 0>
__attribute__((always_inline)) inline void Conv2d(
        const InputMatrixType &Input, OutputMatrixType &out, const WeightMatrixType &Weights, const BiasMatrixType &Bias, const Lambda &Act, const ActivationInformation &...ActivationParameters) {
    constexpr auto KernelParallel = WeightMatrixType::dim5;
    constexpr auto Unrolled       = WeightMatrixType::dim6;
    Conv2d<stride, padding>(Input, out, Weights, Bias, std::make_index_sequence<KernelParallel>{}, std::make_index_sequence<Unrolled>{}, Act, ActivationParameters...);
}

/*
Simple Conv2d Layer, uses stack memory,
wrapps the Conv2d function
*/
template < // Template parameters
        Dim_size_t stride,
        Dim_size_t padding,
        typename OutputType,
        typename WeightType,
        typename AccumulationType,
        typename InputMatrixType,
        typename InputMatrixTypePermuted = typename InputMatrixType::template Permutation<DimensionOrder::D4_Batch_Channel_Width_Height>,
        Dim_size_t Batch                 = InputMatrixTypePermuted::dim1, // Input batch
        Dim_size_t InputWidth            = InputMatrixTypePermuted::dim3, // Input Width
        Dim_size_t InputHeight           = InputMatrixTypePermuted::dim4, // Input height
        Dim_size_t InputChannels         = InputMatrixTypePermuted::dim2, // Input Channels
        typename InputType               = typename InputMatrixType::type,
        Dim_size_t OutputChannels, // Weight Output Channels
        // Dim_size_t M2_2,	// Weight Input Channels has to be the same as InputChannels
        Dim_size_t KernelWidth,  // Weight Kernel shape
        Dim_size_t KernelHeight, // Weight Kernel shape
        class Lambda,
        typename... ActivationInformation>
__attribute__((always_inline)) inline auto Conv2d(
        // Function Parameters
        const InputMatrixType                                                                                                                                &Input,
        const Matrix<WeightType, DimensionOrder::D4_OutChannel_InChannel_KernelWidth_KernelHeight, OutputChannels, InputChannels, KernelWidth, KernelHeight> &Weights,
        const Matrix<AccumulationType, DimensionOrder::D1_Channel, OutputChannels>                                                                           &Bias,
        const Lambda                                                                                                                                         &Act,
        const Matrix<ActivationInformation, DimensionOrder::D1_Channel, OutputChannels> &...ActivationParameters) {
    Matrix<OutputType, DimensionOrder::D4_Batch_Channel_Width_Height, Batch, OutputChannels, (InputWidth - KernelWidth + 2 * padding) / stride + 1,
           (InputHeight - KernelHeight + 2 * padding) / stride + 1>
            out;
    Conv2d<stride, padding>(Input, out, Weights, Bias, Act, ActivationParameters...);
    return out;
}

} // namespace conv2d
} // namespace functions