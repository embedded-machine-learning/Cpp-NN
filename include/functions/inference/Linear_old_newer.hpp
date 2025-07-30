#pragma once

#include "../../MAC.hpp"
#include "../../Matrix.hpp"
#include "../../helpers/AccumulationTypes.hpp"

// #include "../../helpers/TestHelpers.hpp"
#include "../../helpers/Algorithm.hpp"

#ifndef Number_Of_Registers
#define Number_Of_Registers 32
#endif


namespace functions {
namespace linear {
/*
Best Linear Configuration for channel Order is D2_Batch_Channel
*/

/*
Simple Linear Layer, uses mostly passed memory
*/
template < // Template Parameters
        typename InputMatrixType,
        typename OutputMatrixType,
        typename WeightMatrixType,
        typename InputMatrixTypePermuted  = typename InputMatrixType::template Permutation<DimensionOrder::D2_Batch_Channel>,
        typename WeightMatrixTypePermuted = typename WeightMatrixType::template Permutation<DimensionOrder::D2_OutChannel_InChannel>,
        typename OutputMatrixTypePermuted = typename OutputMatrixType::template Permutation<DimensionOrder::D2_Batch_Channel>,
        Dim_size_t OutputChannels, // automatically inferred from Bias
        typename AccumulationType, // automatically inferred from Bias
        Dim_size_t Batch         = InputMatrixTypePermuted::dim1,
        Dim_size_t InputChannels = InputMatrixTypePermuted::dim2,
        typename InputType       = typename InputMatrixType::type,
        typename WeightType      = typename WeightMatrixType::type,
        typename OutputType      = typename OutputMatrixType::type,
        typename Lambda,
        std::enable_if_t<WeightMatrixType::dims == 2, int> = 0,
        typename... ActivationInformation>
__attribute__((always_inline)) inline void Linear( // Function Parameters
        const InputMatrixType                                                      &Input,
        OutputMatrixType                                                           &out,
        const WeightMatrixType                                                     &Weights,
        const Matrix<AccumulationType, DimensionOrder::D1_Channel, OutputChannels> &Bias,
        const Lambda                                                               &Act,
        const Matrix<ActivationInformation, DimensionOrder::D1_Channel, OutputChannels> &...ActivationParameters) {
    static_assert(std::is_same_v<AccumulationType_helper<InputType, WeightType>, void> == false, "AccumulationType is not defined for the given Input and Weight types");
    static_assert(std::is_same_v<AccumulationType_helper<InputType, WeightType>, AccumulationType> == true, "Bias does not use recommended AccumulationType");

    static_assert(InputChannels == WeightMatrixTypePermuted::dim2, "InputChannels must be equal to Weights.dim2");
    static_assert(OutputChannels == WeightMatrixTypePermuted::dim1, "OutputChannels must be equal to Weights.dim1");
    static_assert(Batch == OutputMatrixTypePermuted::dim1, "Batch must be equal to Input.dim1");

    for (Dim_size_t batch = 0; batch < Batch; batch++) {
        for (Dim_size_t output_channel = 0; output_channel < OutputChannels; output_channel++) {
            AccumulationType sum{Bias.at(output_channel)};
            for (Dim_size_t input_channel = 0; input_channel < InputChannels; input_channel += 1) {
                sum += static_cast<AccumulationType>(Input.template at<DimensionOrder::D2_Batch_Channel>(batch, input_channel)) *
                       static_cast<AccumulationType>(Weights.template at<DimensionOrder::D2_OutChannel_InChannel>(output_channel, input_channel));
            }
            out.template at<DimensionOrder::D2_Batch_Channel>(batch, output_channel) =
                    Lambda::template Act<AccumulationType, OutputType, ActivationInformation...>(sum, ActivationParameters.at(output_channel)...);
        }
    }
}

// Weight unroll and parallelization constexpr
template <
        Dim_size_t KernelParallelSuggested,
        Dim_size_t UnrolledSuggested,
        typename InputMatrixType,
        typename InputMatrixTypePermuted = typename InputMatrixType::template Permutation<DimensionOrder::D2_OutChannel_InChannel>,
        typename Type                    = typename InputMatrixType::type,
        Dim_size_t OutputChannels        = InputMatrixTypePermuted::dim1,
        Dim_size_t InputChannels         = InputMatrixTypePermuted::dim2,
        Dim_size_t KernelParallel        = helpers::highest_restless_division_factor_up_to(OutputChannels, KernelParallelSuggested),
        Dim_size_t Unrolled              = helpers::highest_restless_division_factor_up_to(InputChannels, UnrolledSuggested),
        typename OutputMatrixType = Matrix<Type, DimensionOrder::D4_OutChannel_InChannel_KernelParallel_Unrolled, OutputChannels / KernelParallel, InputChannels / Unrolled, KernelParallel, Unrolled>,
        std::enable_if_t<OutputChannels % KernelParallel == 0, int> = 0>
__attribute__((always_inline)) inline constexpr OutputMatrixType WeightUnrollParallel_old(const InputMatrixType &Weights) {
    static_assert(OutputChannels % KernelParallel == 0, "OutputChannels must be divisible by KernelParallel, how the fuck did you manage to fail this assert like it is impossible!");
    static_assert(InputChannels % Unrolled == 0, "InputChannels must be divisible by Unrolled, how the fuck did you manage to fail this assert like it is impossible!");
    OutputMatrixType WeightsUnrolled;

    for (Dim_size_t output_channel = 0; output_channel < OutputChannels / KernelParallel; output_channel++) {
        for (Dim_size_t input_channel = 0; input_channel < InputChannels / Unrolled; input_channel++) {
            for (Dim_size_t kernel_parallel = 0; kernel_parallel < KernelParallel; kernel_parallel++) {
                for (Dim_size_t unrolled = 0; unrolled < Unrolled; unrolled++) {
                    WeightsUnrolled.data[output_channel][input_channel][kernel_parallel][unrolled] =
                            Weights.template at<DimensionOrder::D2_OutChannel_InChannel>(output_channel * KernelParallel + kernel_parallel, input_channel * Unrolled + unrolled);
                }
            }
        }
    }
    return WeightsUnrolled;
}

// Weight unroll and parallelization constexpr
template <
        Dim_size_t KernelParallelSuggested,
        Dim_size_t UnrolledSuggested,
        typename InputMatrixType,
        typename InputMatrixTypePermuted = typename InputMatrixType::template Permutation<DimensionOrder::D2_OutChannel_InChannel>,
        typename Type                    = typename InputMatrixType::type,
        Dim_size_t OutputChannels        = InputMatrixTypePermuted::dim1,
        Dim_size_t InputChannels         = InputMatrixTypePermuted::dim2,
        Dim_size_t KernelParallel        = (KernelParallelSuggested > OutputChannels) ? OutputChannels : KernelParallelSuggested,
        Dim_size_t Unrolled              = helpers::highest_restless_division_factor_up_to(InputChannels, UnrolledSuggested),
        typename OutputMatrixType = Matrix<Type, DimensionOrder::D4_OutChannel_InChannel_KernelParallel_Unrolled, OutputChannels / KernelParallel, InputChannels / Unrolled, KernelParallel, Unrolled>,
        typename SpillType        = std::conditional_t<OutputChannels - (OutputChannels / KernelParallel) * KernelParallel == 0, void, Type>,
        typename OutputMatrixSpillType = Matrix<SpillType,
                                                DimensionOrder::D4_OutChannel_InChannel_KernelParallel_Unrolled,
                                                1,
                                                InputChannels / Unrolled,
                                                OutputChannels - (OutputChannels / KernelParallel) * KernelParallel,
                                                Unrolled>>
__attribute__((always_inline)) inline constexpr std::tuple<const OutputMatrixType, const OutputMatrixSpillType> WeightUnrollParallel(const InputMatrixType &Weights) {
    // static_assert(OutputChannels % KernelParallel == 0, "OutputChannels must be divisible by KernelParallel, how the fuck did you manage to fail this assert like it is impossible!");
    static_assert(InputChannels % Unrolled == 0, "InputChannels must be divisible by Unrolled, how the fuck did you manage to fail this assert like it is impossible!");
    static_assert(OutputMatrixSpillType::dim3 + OutputMatrixType::dim1 * OutputMatrixType::dim3 == OutputChannels, "Total of divided OutputChannels must be the same as the OutputChannels");
    OutputMatrixType      WeightsUnrolled;
    OutputMatrixSpillType WeightsUnrolledSpilled;

    for (Dim_size_t output_channel = 0; output_channel < OutputChannels / KernelParallel; output_channel++) {
        for (Dim_size_t input_channel = 0; input_channel < InputChannels / Unrolled; input_channel++) {
            for (Dim_size_t kernel_parallel = 0; kernel_parallel < KernelParallel; kernel_parallel++) {
                for (Dim_size_t unrolled = 0; unrolled < Unrolled; unrolled++) {
                    WeightsUnrolled.data[output_channel][input_channel][kernel_parallel][unrolled] =
                            Weights.template at<DimensionOrder::D2_OutChannel_InChannel>(output_channel * KernelParallel + kernel_parallel, input_channel * Unrolled + unrolled);
                }
            }
        }
    }
    if constexpr (OutputMatrixSpillType::dim3 > 0) {
        Dim_size_t start = (OutputChannels / KernelParallel) * KernelParallel;
        for (Dim_size_t input_channel = 0; input_channel < InputChannels / Unrolled; input_channel++) {
            for (Dim_size_t kernel_parallel = 0; kernel_parallel < OutputChannels - start; kernel_parallel++) {
                for (Dim_size_t unrolled = 0; unrolled < Unrolled; unrolled++) {
                    WeightsUnrolledSpilled.data[0][input_channel][kernel_parallel][unrolled] =
                            Weights.template at<DimensionOrder::D2_OutChannel_InChannel>(start + kernel_parallel, input_channel * Unrolled + unrolled);
                }
            }
        }
    }
    return std::make_tuple(WeightsUnrolled, WeightsUnrolledSpilled);
}

/*
Simple Linear Layer, uses mostly passed memory with partially unrolled and parallelized weight matrix
*/
template < // Template Parameters
        typename InputMatrixType,
        typename OutputMatrixType,
        typename InputMatrixTypePermuted = typename InputMatrixType::template Permutation<DimensionOrder::D2_Batch_Channel>,
        Dim_size_t Batch                 = InputMatrixTypePermuted::dim1,
        Dim_size_t InputChannels         = InputMatrixTypePermuted::dim2,
        Dim_size_t KernelParallel, // automatically inferred from Weights
        Dim_size_t Unrolled,       // automatically inferred from Weights
        Dim_size_t OutputChannels, // automatically inferred from Bias
        typename InputType = typename InputMatrixType::type,
        typename WeightType, // automatically inferred from Weights
        typename OutputType       = typename OutputMatrixType::type,
        typename AccumulationType = AccumulationType_helper<InputType, WeightType>,
        typename Lambda,
        size_t... KernelIndexes,
        size_t... UnrollIndexes,
        typename... ActivationInformation>
// __attribute__((always_inline)) inline
void Linear_old( // Function Parameters
        const InputMatrixType                                                                                                                                                          &Input,
        OutputMatrixType                                                                                                                                                               &out,
        const Matrix<WeightType, DimensionOrder::D4_OutChannel_InChannel_KernelParallel_Unrolled, OutputChannels / KernelParallel, InputChannels / Unrolled, KernelParallel, Unrolled> &Weights,
        const Matrix<AccumulationType, DimensionOrder::D1_Channel, OutputChannels>                                                                                                     &Bias,
        std::index_sequence<KernelIndexes...>,
        std::index_sequence<UnrollIndexes...>,
        const Lambda &Act,
        const Matrix<ActivationInformation, DimensionOrder::D1_Channel, OutputChannels> &...ActivationParameters) {
    static_assert(std::is_same_v<AccumulationType_helper<InputType, WeightType>, void> == false, "AccumulationType is not defined for the given Input and Weight types");
    static_assert(std::is_same_v<AccumulationType_helper<InputType, WeightType>, AccumulationType> == true, "Bias does not use recommended AccumulationType");

#pragma GCC unroll(1)
    for (Dim_size_t batch = 0; batch < Batch; batch++) {
#pragma GCC unroll(1)
        for (Dim_size_t output_channel = 0; output_channel < OutputChannels / KernelParallel; output_channel++) {
            AccumulationType sum[KernelParallel]{Bias.at(output_channel * KernelParallel + KernelIndexes)...};
#pragma GCC unroll(1)
            for (Dim_size_t input_channel = 0; input_channel < InputChannels / Unrolled; input_channel++) {
                const InputType input[Unrolled] = {Input.template at<DimensionOrder::D2_Batch_Channel>(batch, input_channel * Unrolled + UnrollIndexes)...};
                // auto InputLambda = [&](Dim_size_t pos){return Input.data[batch][input_channel * Unrolled + pos];};
                //  #pragma unroll
#pragma GCC unroll(65534) // 65534 is the maximum unroll value for GCC, so full unroll
                for (Dim_size_t kernel_parallel = 0; kernel_parallel < KernelParallel; kernel_parallel++) {
                    const WeightType weights[Unrolled] = {Weights.at(output_channel, input_channel, kernel_parallel, UnrollIndexes)...};
                    // auto WeightLambda = [&](Dim_size_t pos){return Weights.data[output_channel][input_channel][kernel_parallel][pos];};
                    //  #pragma GCC unroll(65534)
                    //                      for (Dim_size_t unrolled = 0; unrolled < Unrolled; unrolled++) {
                    //                          sum[kernel_parallel] += static_cast<AccumulationType>(input[unrolled]) * static_cast<AccumulationType>(weights[unrolled]);
                    //                      }
                    sum[kernel_parallel] = MAC<Unrolled, InputType, WeightType, AccumulationType, UnrollIndexes...>::OP(input, weights, sum[kernel_parallel], std::index_sequence<UnrollIndexes...>{});
                }
            }
// #pragma unroll
#pragma GCC unroll(65534)
            for (Dim_size_t kernel_parallel = 0; kernel_parallel < KernelParallel; kernel_parallel++) {
                out.template at<DimensionOrder::D2_Batch_Channel>(batch, output_channel * KernelParallel + kernel_parallel) =
                        Lambda::template Act<AccumulationType, OutputType, ActivationInformation...>(sum[kernel_parallel],
                                                                                                     ActivationParameters.at(output_channel * KernelParallel + kernel_parallel)...);
            }
        }
    }
}

template < // Template Parameters
        typename InputMatrixType,
        typename WeightMatrixType,
        typename WeightMatrixSpillType,
        typename OutputMatrixType,
        typename InputMatrixTypePermuted = typename InputMatrixType::template Permutation<DimensionOrder::D2_Batch_Channel>,
        Dim_size_t Batch                 = InputMatrixTypePermuted::dim1,
        Dim_size_t InputChannels         = InputMatrixTypePermuted::dim2,
        Dim_size_t KernelParallel        = WeightMatrixType::dim3,
        Dim_size_t Unrolled              = WeightMatrixType::dim4,
        Dim_size_t OutputChannels, // automatically inferred from Bias
        Dim_size_t KernelRest     = WeightMatrixSpillType::dim3,
        typename InputType        = typename InputMatrixType::type,
        typename WeightType       = typename WeightMatrixType::type,
        typename OutputType       = typename OutputMatrixType::type,
        typename AccumulationType = AccumulationType_helper<InputType, WeightType>,
        typename Lambda,
        size_t... KernelIndexes,
        size_t... UnrollIndexes,
        size_t... KernelRestIndexes,
        size_t... PermanentInputIndexes,
        typename... ActivationInformation>
__attribute__((always_inline)) inline
void Linear( // Function Parameters
        const InputMatrixType                                                      &Input,
        OutputMatrixType                                                           &out,
        const std::tuple<WeightMatrixType, WeightMatrixSpillType>                  &WeightsTuple,
        const Matrix<AccumulationType, DimensionOrder::D1_Channel, OutputChannels> &Bias,
        std::index_sequence<KernelIndexes...>,
        std::index_sequence<UnrollIndexes...>,
        std::index_sequence<KernelRestIndexes...>,
        std::index_sequence<PermanentInputIndexes...>,
        const Lambda &Act,
        const Matrix<ActivationInformation, DimensionOrder::D1_Channel, OutputChannels> &...ActivationParameters) {
    static_assert(std::is_same_v<AccumulationType_helper<InputType, WeightType>, void> == false, "AccumulationType is not defined for the given Input and Weight types");
    static_assert(std::is_same_v<AccumulationType_helper<InputType, WeightType>, AccumulationType> == true, "Bias does not use recommended AccumulationType");

    const WeightMatrixType      &Weights      = std::get<0>(WeightsTuple);
    const WeightMatrixSpillType &WeightsSpill = std::get<1>(WeightsTuple);

    constexpr auto AmountOfPermanentInputs = sizeof...(PermanentInputIndexes);

    static_assert(AmountOfPermanentInputs>=0 && AmountOfPermanentInputs <= InputChannels , "WTF");

#pragma GCC unroll(1)
    for (Dim_size_t batch = 0; batch < Batch; batch++) {
        InputType input_permanent[AmountOfPermanentInputs][Unrolled];
        if constexpr (AmountOfPermanentInputs > 0) {
#pragma GCC unroll(65534)
            for (Dim_size_t input_channel = 0; input_channel < AmountOfPermanentInputs; input_channel++) {
#pragma GCC unroll(65534)
                for (Dim_size_t unrolled = 0; unrolled < Unrolled; unrolled++) {
                    input_permanent[input_channel][unrolled] = Input.template at<DimensionOrder::D2_Batch_Channel>(batch, input_channel * Unrolled + unrolled);
                }
            }
        }
#pragma GCC unroll(1)
        for (Dim_size_t output_channel = 0; output_channel < OutputChannels / KernelParallel; output_channel++) {
            AccumulationType sum[KernelParallel]{Bias.at(output_channel * KernelParallel + KernelIndexes)...};
            if constexpr (AmountOfPermanentInputs > 0) {
                // permanent Inputs
#pragma GCC unroll(65534) // 65534 is the maximum unroll value for GCC, so full unroll
                for (Dim_size_t input_channel = 0; input_channel < AmountOfPermanentInputs; input_channel++) {
#pragma GCC unroll(65534) // 65534 is the maximum unroll value for GCC, so full unroll
                    for (Dim_size_t kernel_parallel = 0; kernel_parallel < KernelParallel; kernel_parallel++) {
                        const WeightType weights[Unrolled] = {Weights.at(output_channel, input_channel, kernel_parallel, UnrollIndexes)...};
                        sum[kernel_parallel]               = MAC<Unrolled, InputType, WeightType, AccumulationType, UnrollIndexes...>::OP(input_permanent[input_channel], weights, sum[kernel_parallel],
                                                                                                                                          std::index_sequence<UnrollIndexes...>{});
                    }
                }
            }
           // Looped Inputs
            if constexpr (AmountOfPermanentInputs != InputChannels ){
#pragma GCC unroll(1)
				for (Dim_size_t input_channel = AmountOfPermanentInputs; input_channel < InputChannels / Unrolled; input_channel++) {
					const InputType input[Unrolled] = {Input.template at<DimensionOrder::D2_Batch_Channel>(batch, input_channel * Unrolled + UnrollIndexes)...};
					//  #pragma unroll
#pragma GCC unroll(65534) // 65534 is the maximum unroll value for GCC, so full unroll
					for (Dim_size_t kernel_parallel = 0; kernel_parallel < KernelParallel; kernel_parallel++) {
						const WeightType weights[Unrolled] = {Weights.at(output_channel, input_channel, kernel_parallel, UnrollIndexes)...};
						sum[kernel_parallel] = MAC<Unrolled, InputType, WeightType, AccumulationType, UnrollIndexes...>::OP(input, weights, sum[kernel_parallel], std::index_sequence<UnrollIndexes...>{});
					}
				}
            }
// #pragma unroll
#pragma GCC unroll(65534)
            for (Dim_size_t kernel_parallel = 0; kernel_parallel < KernelParallel; kernel_parallel++) {
                out.template at<DimensionOrder::D2_Batch_Channel>(batch, output_channel * KernelParallel + kernel_parallel) =
                        Lambda::template Act<AccumulationType, OutputType, ActivationInformation...>(sum[kernel_parallel],
                                                                                                     ActivationParameters.at(output_channel * KernelParallel + kernel_parallel)...);
            }
        }
        if constexpr (KernelRest > 0) {
            Dim_size_t       output_channel = (OutputChannels / KernelParallel) * KernelParallel;
            AccumulationType sum[KernelRest]{Bias.at(output_channel + KernelRestIndexes)...};
            if constexpr (AmountOfPermanentInputs > 0) {
                // permanent Inputs
#pragma GCC unroll(65534) // 65534 is the maximum unroll value for GCC, so full unroll
                for (Dim_size_t input_channel = 0; input_channel < AmountOfPermanentInputs; input_channel++) {
#pragma GCC unroll(65534) // 65534 is the maximum unroll value for GCC, so full unroll
                    for (Dim_size_t kernel_rest = 0; kernel_rest < KernelRest; kernel_rest++) {
                        const WeightType weights[Unrolled] = {WeightsSpill.at(0, input_channel, kernel_rest, UnrollIndexes)...};
                        sum[kernel_rest]                   = MAC<Unrolled, InputType, WeightType, AccumulationType, UnrollIndexes...>::OP(input_permanent[input_channel], weights, sum[kernel_rest],
                                                                                                                                          std::index_sequence<UnrollIndexes...>{});
                    }
                }
            }
            // Looped Inputs
            if constexpr (AmountOfPermanentInputs != InputChannels ){
#pragma GCC unroll(1)
				for (Dim_size_t input_channel = AmountOfPermanentInputs; input_channel < InputChannels / Unrolled; input_channel++) {
					const InputType input[Unrolled] = {Input.template at<DimensionOrder::D2_Batch_Channel>(batch, input_channel * Unrolled + UnrollIndexes)...};
					//  #pragma unroll
#pragma GCC unroll(65534) // 65534 is the maximum unroll value for GCC, so full unroll
					for (Dim_size_t kernel_rest = 0; kernel_rest < KernelRest; kernel_rest++) {
						const WeightType weights[Unrolled] = {WeightsSpill.at(0, input_channel, kernel_rest, UnrollIndexes)...};
						sum[kernel_rest] = MAC<Unrolled, InputType, WeightType, AccumulationType, UnrollIndexes...>::OP(input, weights, sum[kernel_rest], std::index_sequence<UnrollIndexes...>{});
					}
				}
            }
// #pragma unroll
#pragma GCC unroll(65534)
            for (Dim_size_t kernel_rest = 0; kernel_rest < KernelRest; kernel_rest++) {
                out.template at<DimensionOrder::D2_Batch_Channel>(batch, output_channel + kernel_rest) =
                        Lambda::template Act<AccumulationType, OutputType, ActivationInformation...>(sum[kernel_rest], ActivationParameters.at(output_channel + kernel_rest)...);
            }
        }
    }
}

// Wrapper for the unrolled and parallelized Linear function, to automatically infer KernelParallel and Unrolled
template < // Template Parameters
        typename InputMatrixType,
        typename WeightMatrixType,
        typename BiasMatrixType,
        typename OutputMatrixType,
        typename Lambda,
        typename... ActivationMatrixType,
        std::enable_if_t<WeightMatrixType::dims == 4, int> = 0>
__attribute__((always_inline)) inline void Linear_old( // Function Parameters
        const InputMatrixType  &Input,
        OutputMatrixType       &out,
        const WeightMatrixType &Weights,
        const BiasMatrixType   &Bias,
        const Lambda           &Act,
        const ActivationMatrixType &...ActivationParameters) {
    constexpr auto KernelParallel = WeightMatrixType::dim3;
    constexpr auto Unrolled       = WeightMatrixType::dim4;
    Linear_old(Input, out, Weights, Bias, std::make_index_sequence<KernelParallel>{}, std::make_index_sequence<Unrolled>{}, Act, ActivationParameters...);
}

// Wrapper for the unrolled and parallelized Linear function, to automatically infer KernelParallel and Unrolled
template < // Template Parameters
        typename InputMatrixType,
        typename WeightMatrixType,
        typename WeightMatrixSpillType,
        typename BiasMatrixType,
        typename OutputMatrixType,
        typename Lambda,
        typename... ActivationMatrixType,
        std::enable_if_t<WeightMatrixType::dims == 4, int> = 0>
__attribute__((always_inline)) inline void Linear( // Function Parameters
        const InputMatrixType                                     &Input,
        OutputMatrixType                                          &out,
        const std::tuple<WeightMatrixType, WeightMatrixSpillType> &Weights,
        const BiasMatrixType                                      &Bias,
        const Lambda                                              &Act,
        const ActivationMatrixType &...ActivationParameters) {
    constexpr auto KernelParallel = WeightMatrixType::dim3;
    constexpr auto KernelRest     = WeightMatrixSpillType::dim3;
    constexpr auto Unrolled       = WeightMatrixType::dim4;

    constexpr int AmountOfPermanentInputs_tmp = (int)Number_Of_Registers - (int)3 - (int)KernelParallel;
    //constexpr int  AmountOfPermanentInputs_tmp_2 =(AmountOfPermanentInputs_tmp < (int)InputMatrixType::dim2)? (int)AmountOfPermanentInputs_tmp : (int)InputMatrixType::dim2 ;
    constexpr int  AmountOfPermanentInputs_tmp_2 =(AmountOfPermanentInputs_tmp < (int)InputMatrixType::dim2)? (int)0 : (int)InputMatrixType::dim2 ;
    constexpr int AmountOfPermanentInputs = (AmountOfPermanentInputs_tmp_2>=0)? (int)AmountOfPermanentInputs_tmp_2 : (int)0;

    Linear(Input, out, Weights, Bias, std::make_index_sequence<KernelParallel>{}, std::make_index_sequence<Unrolled>{}, std::make_index_sequence<KernelRest>{},
           std::make_index_sequence<AmountOfPermanentInputs>{}, Act, ActivationParameters...);
}

/*
Simple Linear Layer, uses stack memory,
wrapps the Linear function
*/
template < // Template Parameters
        typename OutputType       = float,
        typename InputType        = float,
        typename WeightType       = float,
        typename AccumulationType = AccumulationType_helper<InputType, WeightType>,
        typename InputMatrixType,
        typename InputMatrixTypePermuted = typename InputMatrixType::template Permutation<DimensionOrder::D2_Batch_Channel>,
        Dim_size_t Batch                 = InputMatrixTypePermuted::dim1,
        Dim_size_t InputChannels         = InputMatrixTypePermuted::dim2,
        Dim_size_t OutputChannels,
        typename Lambda,
        // Matrix<InputType, DimensionOrder::D2_Batch_Channel, Batch, InputChannels> _ = InputMatrixType(),
        typename... ActivationInformation>
__attribute__((always_inline)) inline Matrix<OutputType, DimensionOrder::D2_Batch_Channel, Batch, OutputChannels> Linear( // Function Parameters
        const InputMatrixType                                                                            &Input,
        const Matrix<WeightType, DimensionOrder::D2_OutChannel_InChannel, OutputChannels, InputChannels> &Weights,
        const Matrix<AccumulationType, DimensionOrder::D1_Channel, OutputChannels>                       &Bias,
        const Lambda                                                                                     &Act,
        const Matrix<ActivationInformation, DimensionOrder::D1_Channel, OutputChannels> &...ActivationParameters) {

    Matrix<OutputType, DimensionOrder::D2_Batch_Channel, Batch, OutputChannels> out;
    Linear(Input, out, Weights, Bias, Act, ActivationParameters...);
    return out;
}
} // namespace linear
} // namespace functions
