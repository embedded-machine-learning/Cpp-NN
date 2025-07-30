#pragma once

#include "../../MAC.hpp"
#include "../../Matrix.hpp"
#include "../../helpers/AccumulationTypes.hpp"

// #include "../../helpers/TestHelpers.hpp"
#include "../../helpers/Algorithm.hpp"

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
    static_assert(isAccumulationTypeSupported<InputType, WeightType>, "AccumulationType is not defined for the given Input and Weight types");
    static_assert(isAccumulationTypeValid<InputType, WeightType, AccumulationType>, "Bias does not use recommended AccumulationType");

    static_assert(InputChannels == WeightMatrixTypePermuted::dim2, "InputChannels must be equal to Weights.dim2");
    static_assert(OutputChannels == WeightMatrixTypePermuted::dim1, "OutputChannels must be equal to Weights.dim1");
    static_assert(Batch == OutputMatrixTypePermuted::dim1, "Batch must be equal to Input.dim1");

    for (Dim_size_t batch = 0; batch < Batch; batch++) {
        for (Dim_size_t output_channel = 0; output_channel < OutputChannels; output_channel++) {
            AccumulationType sum{Bias.at(output_channel)};
            for (Dim_size_t input_channel = 0; input_channel < InputChannels; input_channel += 1) {
                const InputType input[1] = {Input.template at<DimensionOrder::D2_Batch_Channel>(batch, input_channel)};
                const WeightType weights[1] = {Weights.template at<DimensionOrder::D2_OutChannel_InChannel>(output_channel, input_channel)};
                sum = MAC<1, InputType, WeightType, AccumulationType,0>::OP(input, weights, sum, std::make_index_sequence<1>{});

                // sum += static_cast<AccumulationType>(Input.template at<DimensionOrder::D2_Batch_Channel>(batch, input_channel)) *
                //        static_cast<AccumulationType>(Weights.template at<DimensionOrder::D2_OutChannel_InChannel>(output_channel, input_channel));
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

template<typename Type>
constexpr Dim_size_t getInputDim=0;

template<typename Type, DimensionOrder Order, Dim_size_t dim1, Dim_size_t dim2>
constexpr Dim_size_t getInputDim<Matrix<Type,Order,dim1,dim2>> = Matrix<Type,Order,dim1,dim2>::template Permutation<DimensionOrder::D2_OutChannel_InChannel>::dim2;

template<typename Type, Dim_size_t dim1, Dim_size_t dim2, Dim_size_t dim3, Dim_size_t dim4>
constexpr Dim_size_t getInputDim<Matrix<Type,DimensionOrder::D4_OutChannel_InChannel_KernelParallel_Unrolled,dim1,dim2,dim3,dim4>> = dim2*dim4;

template<typename Type1, typename Type2>
constexpr Dim_size_t getInputDim<std::tuple<Type1,Type2>> = getInputDim<Type1> + getInputDim<Type2>;


template<typename Type>
struct un_unrolled_Matrix_helper_struct;

template<typename Type, DimensionOrder Order, Dim_size_t dim1, Dim_size_t dim2>
struct un_unrolled_Matrix_helper_struct<Matrix<Type,Order,dim1, dim2>>{
    using type = typename Matrix<Type,Order,dim1, dim2>::template Permutation<DimensionOrder::D2_OutChannel_InChannel>;
};

template<typename Type, DimensionOrder Order, Dim_size_t dim1>
struct un_unrolled_Matrix_helper_struct<Matrix<Type,Order,dim1>>{
    using type = Matrix<Type,Order,dim1>;
};

template<typename Type, Dim_size_t OutputChannels, Dim_size_t InputChannels ,Dim_size_t KernelParallel,Dim_size_t Unrolled>
struct un_unrolled_Matrix_helper_struct<Matrix<Type,DimensionOrder::D4_OutChannel_InChannel_KernelParallel_Unrolled,OutputChannels,InputChannels,KernelParallel,Unrolled>>{
    using type = Matrix<Type,DimensionOrder::D2_OutChannel_InChannel,OutputChannels * KernelParallel,InputChannels*Unrolled>;
};

template<typename Type1, typename Type2>
struct un_unrolled_Matrix_helper_struct<std::tuple<Type1,Type2>>{
    using Mtype1 = typename un_unrolled_Matrix_helper_struct<remove_cvref_t<Type1>>::type;
    using Mtype2 = typename un_unrolled_Matrix_helper_struct<remove_cvref_t<Type2>>::type;
    using type = Matrix<typename Mtype1::type,Mtype1::order,Mtype1::dim1 + Mtype2::dim1,Mtype1::dim2>;
};

template<typename Type>
using get_un_unrolled_Matrix = typename un_unrolled_Matrix_helper_struct<Type>::type;


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
    static_assert(isAccumulationTypeSupported<InputType, WeightType>, "AccumulationType is not defined for the given Input and Weight types");
    static_assert(isAccumulationTypeValid<InputType, WeightType,AccumulationType>, "Bias does not use recommended AccumulationType");

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
        typename... ActivationInformation>
// __attribute__((always_inline)) inline
void Linear_Multiple_Output_Channels( // Function Parameters
        const InputMatrixType                                                      &Input,
        OutputMatrixType                                                           &out,
        const std::tuple<WeightMatrixType, WeightMatrixSpillType>                  &WeightsTuple,
        const Matrix<AccumulationType, DimensionOrder::D1_Channel, OutputChannels> &Bias,
        std::index_sequence<KernelIndexes...>,
        std::index_sequence<UnrollIndexes...>,
        std::index_sequence<KernelRestIndexes...>,
        const Lambda &Act,
        const Matrix<ActivationInformation, DimensionOrder::D1_Channel, OutputChannels> &...ActivationParameters) {
    static_assert(isAccumulationTypeSupported<InputType, WeightType>, "AccumulationType is not defined for the given Input and Weight types");
    static_assert(isAccumulationTypeValid<InputType, WeightType,AccumulationType>, "Bias does not use recommended AccumulationType");

    const WeightMatrixType      &Weights      = std::get<0>(WeightsTuple);
    const WeightMatrixSpillType &WeightsSpill = std::get<1>(WeightsTuple);
#pragma GCC unroll(1)
    for (Dim_size_t batch = 0; batch < Batch; batch++) {

#pragma GCC unroll(1)
        for (Dim_size_t output_channel = 0; output_channel < OutputChannels / KernelParallel; output_channel++) {
            AccumulationType sum[KernelParallel]{Bias.at(output_channel * KernelParallel + KernelIndexes)...};
            // __builtin_prefetch(&Weights.at(output_channel,0, 0,0), 0, 0); 
            // Looped Inputs
#pragma GCC unroll(1)
            for (Dim_size_t input_channel = 0; input_channel < InputChannels / Unrolled; input_channel++) {
                const InputType input[Unrolled] = {Input.template at<DimensionOrder::D2_Batch_Channel>(batch, input_channel * Unrolled + UnrollIndexes)...};
                // __builtin_prefetch(&Weights.at(output_channel, input_channel, 0,0), 0, 3); 
                //  #pragma unroll
#pragma GCC unroll(65534) // 65534 is the maximum unroll value for GCC, so full unroll
                for (Dim_size_t kernel_parallel = 0; kernel_parallel < KernelParallel; kernel_parallel++) {
                    // (__builtin_prefetch(&Weights.at(output_channel, input_channel, kernel_parallel, UnrollIndexes), 0, 0),...); // very bad
                    const WeightType weights[Unrolled] = {Weights.at(output_channel, input_channel, kernel_parallel, UnrollIndexes)...};
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
        if constexpr (KernelRest > 0) {
            Dim_size_t       output_channel = (OutputChannels / KernelParallel) * KernelParallel;
            AccumulationType sum[KernelRest]{Bias.at(output_channel + KernelRestIndexes)...};

            // Looped Inputs
#pragma GCC unroll(1)
            for (Dim_size_t input_channel = 0; input_channel < InputChannels / Unrolled; input_channel++) {
                const InputType input[Unrolled] = {Input.template at<DimensionOrder::D2_Batch_Channel>(batch, input_channel * Unrolled + UnrollIndexes)...};
                //  #pragma unroll
#pragma GCC unroll(65534) // 65534 is the maximum unroll value for GCC, so full unroll
                for (Dim_size_t kernel_rest = 0; kernel_rest < KernelRest; kernel_rest++) {
                    const WeightType weights[Unrolled] = {WeightsSpill.at(0, input_channel, kernel_rest, UnrollIndexes)...};
                    sum[kernel_rest] = MAC<Unrolled, InputType, WeightType, AccumulationType, UnrollIndexes...>::OP(input, weights, sum[kernel_rest], std::index_sequence<UnrollIndexes...>{});
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
// __attribute__((always_inline)) inline
void Linear_Input_Permanent( // Function Parameters
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
    static_assert(isAccumulationTypeSupported<InputType, WeightType>, "AccumulationType is not defined for the given Input and Weight types");
    static_assert(isAccumulationTypeValid<InputType, WeightType,AccumulationType>, "Bias does not use recommended AccumulationType");
    static_assert(sizeof...(PermanentInputIndexes) == sizeof...(UnrollIndexes), "All Permanent Inputs need to be unrolled");
    static_assert(WeightMatrixType::dim2 == 1, "No additional Inputs other than the unrolled ones are allowed");

    const WeightMatrixType      &Weights      = std::get<0>(WeightsTuple);
    const WeightMatrixSpillType &WeightsSpill = std::get<1>(WeightsTuple);

    // constexpr auto AmountOfPermanentInputs = sizeof...(PermanentInputIndexes);

#pragma GCC unroll(1)
    for (Dim_size_t batch = 0; batch < Batch; batch++) {
        const InputType input_permanent[Unrolled][1]{{Input.template at<DimensionOrder::D2_Batch_Channel>(batch, UnrollIndexes)}...};

#pragma GCC unroll(1)
        for (Dim_size_t output_channel = 0; output_channel < OutputChannels / KernelParallel; output_channel++) {
            // permanent Inputs
#pragma GCC unroll(65534) // 65534 is the maximum unroll value for GCC, so full unroll
            for (Dim_size_t kernel_parallel = 0; kernel_parallel < KernelParallel; kernel_parallel++) {
                AccumulationType sum{Bias.at(output_channel * KernelParallel + kernel_parallel)};
#pragma GCC unroll(65534) // 65534 is the maximum unroll value for GCC, so full unroll
                for (Dim_size_t unrolled = 0; unrolled < Unrolled; unrolled++) {
                    const WeightType weights[1] = {Weights.at(output_channel, 0, kernel_parallel, unrolled)};
                    sum                         = MAC<1, InputType, WeightType, AccumulationType, 0>::OP(input_permanent[unrolled], weights, sum, std::index_sequence<0>{});
                }
                out.template at<DimensionOrder::D2_Batch_Channel>(batch, output_channel * KernelParallel + kernel_parallel) =
                        Lambda::template Act<AccumulationType, OutputType, ActivationInformation...>(sum, ActivationParameters.at(output_channel * KernelParallel + kernel_parallel)...);
            }
        }
        if constexpr (KernelRest > 0) {
            Dim_size_t       output_channel = (OutputChannels / KernelParallel) * KernelParallel;
            AccumulationType sum[KernelRest]{Bias.at(output_channel + KernelRestIndexes)...};
            // permanent Inputs
#pragma GCC unroll(65534) // 65534 is the maximum unroll value for GCC, so full unroll
            for (Dim_size_t kernel_rest = 0; kernel_rest < KernelRest; kernel_rest++) {
#pragma GCC unroll(65534) // 65534 is the maximum unroll value for GCC, so full unroll
                for (Dim_size_t unrolled = 0; unrolled < Unrolled; unrolled++) {
                    const WeightType weights[1] = {WeightsSpill.at(0, 0, kernel_rest, unrolled)};
                    sum[kernel_rest]            = MAC<1, InputType, WeightType, AccumulationType, 0>::OP(input_permanent[unrolled], weights, sum[kernel_rest], std::index_sequence<0>{});
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

    constexpr int AmountOfPermanentInputs = (int)InputMatrixType::dim2;

    //if constexpr (Unrolled != AmountOfPermanentInputs) {
        Linear_Multiple_Output_Channels(Input, out, Weights, Bias, std::make_index_sequence<KernelParallel>{}, std::make_index_sequence<Unrolled>{}, std::make_index_sequence<KernelRest>{}, Act,
                                        ActivationParameters...);
    //} else {
    //    Linear_Input_Permanent(Input, out, Weights, Bias, std::make_index_sequence<KernelParallel>{}, std::make_index_sequence<Unrolled>{}, std::make_index_sequence<KernelRest>{},
    //                           std::make_index_sequence<AmountOfPermanentInputs>{}, Act, ActivationParameters...);
    //}
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
