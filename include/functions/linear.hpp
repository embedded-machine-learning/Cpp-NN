#pragma once

#include "../MAC.hpp"
#include "../Matrix.hpp"
#include "../MatrixOperations.hpp"
#include "../helpers/cpp_helpers.hpp"
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#ifndef __LINEAR_FORCE_INLINE__
#warning "__LINEAR_FORCE_INLINE__ is not defined, using default inlining behavior. Consider defining it."
#define __LINEAR_FORCE_INLINE__ false
#endif

#warning "TODO: check the reference implenetation for questianble std::move"

namespace functions::linear {
/*
Reference implementation of a Linear Layer.
This function performs a linear transformation on the input matrix using the provided weights and bias.
It applies an activation function to the output and supports broadcasting of input, weights, and bias matrices
to handle different batch sizes.
This implementation is designed to be flexible, allowing for various activation functions and activation parameters.
It is NOT optimized for performance and is intended as a ground truth reference for testing and validation purposes.

Inputs:
- Input: The input matrix, with the order of 'BC' (Batch, Channel) or 'C' (Channel).
- Output: The output matrix, with the order of 'BC' (Batch, Channel) or 'C' (Channel).
- Weights: The weight matrix, with the order of 'BIO' (Batch, Input, Output) or 'IO' (Input, Output).
- Bias: The bias matrix, with the order of 'BC' (Batch, Channel) or 'C' (Channel).
- Act: The activation function to apply to the output.
- ActivationParameters: Additional parameters for the activation function, if needed.
*/
template <std::size_t SuggestedSubBatchSize                         = 1, // ununsed
          template <typename, typename, typename> class MACOperator = DefaultMACOperation,
          IsMatrixType InputMatrixType                              = Matrix<int, "BC", 1, 1>,
          IsMatrixType OutputMatrixType                             = Matrix<int, "BC", 1, 1>,
          IsMatrixType WeightMatrixType                             = Matrix<int, "IO", 1, 1>,
          IsMatrixType BiasMatrixType                               = Matrix<int, "BC", 1, 1>,
          typename Lambda                                           = decltype([]() {}),
          IsMatrixType... ActivationInformationMatrixType>
    requires(IsMACOperation<MACOperator, typename InputMatrixType::value_type, typename WeightMatrixType::value_type, typename BiasMatrixType::value_type>)
#if __LINEAR_FORCE_INLINE__
__attribute__((always_inline)) inline // Force inlining for performance
#else
inline // Let the compiler decide the inlining
#endif
                                      // __attribute__((noinline))
        void
        Linear( // Function Parameters
                const InputMatrixType  &Input,
                OutputMatrixType       &Output,
                const WeightMatrixType &Weights,
                const BiasMatrixType   &Bias,
                const Lambda           &Act,
                const ActivationInformationMatrixType &...ActivationParameters) {

    static_assert(InputMatrixType::order.contains('C'), "InputMatrixType must contain 'C' for input channels");
    static_assert(OutputMatrixType::order.contains('C'), "OutputMatrixType must contain 'O' for output channels");
    static_assert(WeightMatrixType::order.contains('I'), "WeightMatrixType must contain 'I' for input channels");
    static_assert(WeightMatrixType::order.contains('O'), "WeightMatrixType must contain 'O' for output channels");
    static_assert(BiasMatrixType::order.contains('C') != BiasMatrixType::order.contains('E'),
                  "BiasMatrixType must contain 'C' for output channels or 'E' single element bias parameters, not both not neither");
    static_assert(InputMatrixType::order.contains('B') == OutputMatrixType::order.contains('B'), "Either both Input and Output must contain 'B' for batch or neither");
    static_assert(((ActivationInformationMatrixType::order.contains('C') || ActivationInformationMatrixType::order.contains('E')) && ...),
                  "ActivationParameters must contain 'C' for output channels or 'E' single element activation parameters");
    static_assert(((ActivationInformationMatrixType::order.contains('C') != ActivationInformationMatrixType::order.contains('E')) && ...),
                  "ActivationParameters must contain 'C' for output channels or 'E' single element activation parameters, not both not neither");

    static_assert(InputMatrixType::order.remove("BC").length() == 0, "Input has to be in the order of 'BC' (Batch, Channel) or 'C' (Channel) only");
    static_assert(OutputMatrixType::order.remove("BC").length() == 0, "Output has to be in the order of 'BC' (Batch, Channel) or 'C' (Channel) only");
    static_assert(WeightMatrixType::order.remove("BIO").length() == 0, "Weights has to be in the order of 'BIO' (Batch, Input, Output) or 'IO' (Input, Output) only");
    static_assert(BiasMatrixType::order.remove("BCE").length() == 0, "Bias has to be in the order of 'BC' (Batch, Channel) or 'C' (Channel) only or 'E' (Element) only");
    static_assert(((ActivationInformationMatrixType::order.remove("BCE").length() == 0) && ...),
                  "ActivationParameters has to be in the order of 'BC' (Batch, Channel) or 'C' (Channel) only, or 'E' (Element) only");

    constexpr Dim_size_t batch_size            = (InputMatrixType::order.contains('B') ? InputMatrixType::dimensions[InputMatrixType::order.indexOf('B')] : 1);
    constexpr Dim_size_t batch_size_tmp        = (OutputMatrixType::order.contains('B') ? OutputMatrixType::dimensions[OutputMatrixType::order.indexOf('B')] : 1);
    constexpr Dim_size_t input_channels        = (InputMatrixType::order.contains('C') ? InputMatrixType::dimensions[InputMatrixType::order.indexOf('C')] : 1);
    constexpr Dim_size_t input_channels_tmp    = (WeightMatrixType::order.contains('I') ? WeightMatrixType::dimensions[WeightMatrixType::order.indexOf('I')] : 1);
    constexpr Dim_size_t output_channels       = (OutputMatrixType::order.contains('C') ? OutputMatrixType::dimensions[OutputMatrixType::order.indexOf('C')] : 1);
    constexpr Dim_size_t output_channels_tmp   = (WeightMatrixType::order.contains('O') ? WeightMatrixType::dimensions[WeightMatrixType::order.indexOf('O')] : 1);
    constexpr Dim_size_t output_channels_tmp_2 = (BiasMatrixType::order.contains('C') ? BiasMatrixType::dimensions[BiasMatrixType::order.indexOf('C')] : output_channels_tmp);

    // check for correct dimensions
    static_assert(batch_size == batch_size_tmp, "Batch size of Input and Output must match");
    static_assert(input_channels == input_channels_tmp, "Input channels of Input and Weights must match");
    static_assert(output_channels == output_channels_tmp, "Output channels of Output and Weights must match");
    static_assert(output_channels == output_channels_tmp_2, "Output channels of Output and Bias must match");
    static_assert((((ActivationInformationMatrixType::order.contains('E') ? ActivationInformationMatrixType::dimensions[BiasMatrixType::order.indexOf('E')] : 1) == 1) && ...),
                  "ActivationParameters using single value 'E' must have a dimension of 1");

    using MACOperator_ = MACOperator<typename InputMatrixType::value_type, typename WeightMatrixType::value_type, typename BiasMatrixType::value_type>;

    // change interpretations of passed matrices
    const auto input_broadcasted   = conditionalBroadcast<"B", {batch_size}>(Input);
    const auto bias_broadcasted    = conditionalBroadcast<"B", {batch_size}>(conditionalReplace<"E", "C">(conditionalReplicate<"E", {output_channels}>(Bias)));
    const auto weights_broadcasted = conditionalBroadcast<"B", {batch_size}>(Weights);
    auto       out_broadcasted     = conditionalBroadcast<"B", {batch_size}>(Output);

    // unifiing the order of matrices
    const auto input_broadcasted_permuted   = permute<"BC">(input_broadcasted);
    const auto bias_broadcasted_permuted    = permute<"BC">(bias_broadcasted);
    const auto weights_broadcasted_permuted = permute<"BIO">(weights_broadcasted);
    auto       out_broadcasted_permuted     = permute<"BC", decltype(out_broadcasted) &>(out_broadcasted);

    // activation_parameters reinterpretations
    // Questionalble move here, TODO: review
    [[maybe_unused]]
    const auto broadcast_permute =
            [=](const auto &matrix) { return std::move(permute<"BC">(conditionalBroadcast<"B", {batch_size}>(conditionalReplace<"E", "C">(conditionalReplicate<"E", {output_channels}>(matrix))))); };

    // get data types
    // using AccumulationType = typename BiasMatrixType::value_type;
    using AccumulationType = typename MACOperator_::AccumulationType_;

    // ((std::cout << broadcast_permute(ActivationParameters) << std::endl), ...);

    for (Dim_size_t batch = 0; batch < batch_size; batch++) {
        for (Dim_size_t output_channel = 0; output_channel < output_channels; output_channel++) {
            AccumulationType sum = MACOperator_::pre_processing(bias_broadcasted_permuted.at(batch, output_channel));
            for (Dim_size_t input_channel = 0; input_channel < input_channels; input_channel++) {
                // const AccumulationType input_value  = static_cast<AccumulationType>(input_broadcasted_permuted.at(batch, input_channel));
                // const AccumulationType weight_value = static_cast<AccumulationType>(weights_broadcasted_permuted.at(batch, input_channel, output_channel));
                // sum += input_value * weight_value;
                MACOperator_::lambda(sum, input_broadcasted_permuted.at(batch, input_channel), weights_broadcasted_permuted.at(batch, input_channel, output_channel));
            }
            out_broadcasted_permuted.at(batch, output_channel) = Act(MACOperator_::post_processing(sum), broadcast_permute(ActivationParameters).at(batch, output_channel)...);
        }
    }
}

template <Dim_size_t SuggestedSubInputChannels, Dim_size_t SuggestedSubOutputChannels, IsMatrixType WeightMatrixType>
struct WeightSubDivideHelper {
    using WeightMatrixTypeNoRefConst            = std::remove_cvref_t<WeightMatrixType>;
    using value_type                            = typename WeightMatrixTypeNoRefConst::value_type;
    static constexpr Dim_size_t input_channels  = (WeightMatrixTypeNoRefConst::order.contains('I') ? WeightMatrixTypeNoRefConst::dimensions[WeightMatrixTypeNoRefConst::order.indexOf('I')] : 1);
    static constexpr Dim_size_t output_channels = (WeightMatrixTypeNoRefConst::order.contains('O') ? WeightMatrixTypeNoRefConst::dimensions[WeightMatrixTypeNoRefConst::order.indexOf('O')] : 1);
    static constexpr Dim_size_t sub_input_size  = (input_channels >= SuggestedSubInputChannels ? SuggestedSubInputChannels : input_channels);
    static constexpr Dim_size_t sub_output_size = (output_channels >= SuggestedSubOutputChannels ? SuggestedSubOutputChannels : output_channels);
    static constexpr Dim_size_t sub_input_channels_count      = input_channels / sub_input_size;
    static constexpr Dim_size_t sub_output_channels_count     = output_channels / sub_output_size;
    static constexpr Dim_size_t sub_input_channels_remainder  = input_channels % sub_input_size;
    static constexpr Dim_size_t sub_output_channels_remainder = output_channels % sub_output_size;
    static constexpr Dim_size_t optimal_input_range           = input_channels - sub_input_channels_remainder;
    static constexpr Dim_size_t optimal_output_range          = output_channels - sub_output_channels_remainder;

    using AMatrixType = Matrix<value_type, "OIoi", sub_output_channels_count, sub_input_channels_count, sub_output_size, sub_input_size>;
    using BMatrixType = Matrix<value_type, "OIoi", sub_output_channels_count, 1, sub_output_size, sub_input_channels_remainder>;
    using CMatrixType = Matrix<value_type, "OIoi", 1, sub_input_channels_count, sub_output_channels_remainder, sub_input_size>;
    using DMatrixType = Matrix<value_type, "OIoi", 1, 1, sub_output_channels_remainder, sub_input_channels_remainder>;

    using WeightMatrixTypeSplit = AlignedMatrixCollection<4, AMatrixType, BMatrixType, CMatrixType, DMatrixType>;
};

template <Dim_size_t SuggestedSubInputChannels, Dim_size_t SuggestedSubOutputChannels, IsMatrixType WeightMatrixType>
using SubBioWeightMatrixType = typename WeightSubDivideHelper<SuggestedSubInputChannels, SuggestedSubOutputChannels, WeightMatrixType>::WeightMatrixTypeSplit;

template <Dim_size_t SuggestedSubInputChannels, Dim_size_t SuggestedSubOutputChannels, IsMatrixType WeightMatrixType = Matrix<int, "IO", 1, 1>>
    requires(std::remove_cvref_t<WeightMatrixType>::order.remove("IO").length() == 0)
__attribute__((always_inline)) constexpr inline auto weightSubBio(WeightMatrixType &&weights) {
    using WeightMatrixTypeNoRefConst = std::remove_cvref_t<WeightMatrixType>;
    static_assert(WeightMatrixTypeNoRefConst::order.contains('I'), "WeightMatrixType must contain 'I' for input channels");
    static_assert(WeightMatrixTypeNoRefConst::order.contains('O'), "WeightMatrixType must contain 'O' for output channels");
    static_assert(!WeightMatrixTypeNoRefConst::order.containsAny("io"), "WeightMatrixType must not contain 'i' (sub-Input) or 'o' (sub-Output) dimensions");
    static_assert(WeightMatrixTypeNoRefConst::order.remove("IO").length() == 0, "Weights has to be in the order of 'IO' (Input, Output) only");

    constexpr Dim_size_t input_channels  = (WeightMatrixTypeNoRefConst::order.contains('I') ? WeightMatrixTypeNoRefConst::dimensions[WeightMatrixTypeNoRefConst::order.indexOf('I')] : 1);
    constexpr Dim_size_t output_channels = (WeightMatrixTypeNoRefConst::order.contains('O') ? WeightMatrixTypeNoRefConst::dimensions[WeightMatrixTypeNoRefConst::order.indexOf('O')] : 1);

    constexpr Dim_size_t sub_input_size  = (input_channels >= SuggestedSubInputChannels ? SuggestedSubInputChannels : input_channels);
    constexpr Dim_size_t sub_output_size = (output_channels >= SuggestedSubOutputChannels ? SuggestedSubOutputChannels : output_channels);

    constexpr Dim_size_t sub_input_channels_count  = input_channels / sub_input_size;
    constexpr Dim_size_t sub_output_channels_count = output_channels / sub_output_size;

    constexpr Dim_size_t sub_input_channels_remainder  = input_channels % sub_input_size;
    constexpr Dim_size_t sub_output_channels_remainder = output_channels % sub_output_size;

    constexpr Dim_size_t optimal_input_range  = input_channels - sub_input_channels_remainder;
    constexpr Dim_size_t optimal_output_range = output_channels - sub_output_channels_remainder;

    const auto weight_expanded = conditionalBroadcast<"I", {input_channels}>(conditionalBroadcast<"O", {output_channels}>(weights));

    /*
    Weight Partitioning:
          --> Input
     |   [A ..... A] [B]
     |   [A ..... A] [B]
     V   [A ..... A] [B]
    out  [C ..... C] [D]

    Matrix A: optimal allignment of sub-Batch, sub-Input, sub-Output
    Matrix B: optimal allignment of sub-Batch,            sub-Output
    Matrix C: optimal allignment of sub-Batch, sub-Input
    Matrix D: optimal allignment of sub-Batch
    */

    // clang-format off
    const auto Weight_A = slice<"IO", optimal_input_range,          optimal_output_range>         (weight_expanded, {0,                   0});
    const auto Weight_B = slice<"IO", sub_input_channels_remainder, optimal_output_range>         (weight_expanded, {optimal_input_range, 0});
    const auto Weight_C = slice<"IO", optimal_input_range,          sub_output_channels_remainder>(weight_expanded, {0,                   optimal_output_range});
    const auto Weight_D = slice<"IO", sub_input_channels_remainder, sub_output_channels_remainder>(weight_expanded, {optimal_input_range, optimal_output_range});
    
    // split the weights into sub-matrices
    const auto Weight_A_split = permute<"OIoi">(split<"O", "Oo", sub_output_channels_count, sub_output_size>              (split<"I", "Ii", sub_input_channels_count, sub_input_size>              (Weight_A)));
    const auto Weight_B_split = permute<"OIoi">(split<"O", "Oo", sub_output_channels_count, sub_output_size>              (split<"I", "Ii", 1,                        sub_input_channels_remainder>(Weight_B)));
    const auto Weight_C_split = permute<"OIoi">(split<"O", "Oo", 1,                         sub_output_channels_remainder>(split<"I", "Ii", sub_input_channels_count, sub_input_size>              (Weight_C)));
    const auto Weight_D_split = permute<"OIoi">(split<"O", "Oo", 1,                         sub_output_channels_remainder>(split<"I", "Ii", 1,                        sub_input_channels_remainder>(Weight_D)));

    // clang-format on

    // return std::make_tuple(materialize(Weight_A_split), materialize(Weight_B_split), materialize(Weight_C_split), materialize(Weight_D_split));
    return makeAlignedMatrixCollection<4>(materialize(Weight_A_split), materialize(Weight_B_split), materialize(Weight_C_split), materialize(Weight_D_split));
}

template <Dim_size_t SuggestedSubInputChannels, Dim_size_t SuggestedSubOutputChannels, IsMatrixType WeightMatrixType = Matrix<int, "IO", 1, 1>>
    requires(std::remove_cvref_t<WeightMatrixType>::order.remove("IO").length() != 0)
__attribute__((always_inline)) constexpr inline WeightMatrixType weightSubBio(WeightMatrixType &&weights) {
    return std::forward<WeightMatrixType>(weights); // If the matrix is not in the expected order, just return it as is
}

template <typename WeightMatrixType>
struct WeightSubDivideInverseHelper {
    using WeightMatrixTypeNoRefConst = std::remove_cvref_t<WeightMatrixType>;
    using AMatrixType                = std::tuple_element_t<0, WeightMatrixTypeNoRefConst>;
    using BMatrixType                = std::tuple_element_t<1, WeightMatrixTypeNoRefConst>;
    using CMatrixType                = std::tuple_element_t<2, WeightMatrixTypeNoRefConst>;
    using DMatrixType                = std::tuple_element_t<3, WeightMatrixTypeNoRefConst>;

    using value_type = typename AMatrixType::value_type;

    static constexpr Dim_size_t sub_input_size                = (AMatrixType::dimensions[AMatrixType::order.indexOf('i')]);
    static constexpr Dim_size_t sub_output_size               = (AMatrixType::dimensions[AMatrixType::order.indexOf('o')]);
    static constexpr Dim_size_t sub_input_channels_count      = (AMatrixType::dimensions[AMatrixType::order.indexOf('I')]);
    static constexpr Dim_size_t sub_output_channels_count     = (AMatrixType::dimensions[AMatrixType::order.indexOf('O')]);
    static constexpr Dim_size_t sub_input_channels_remainder  = (BMatrixType::dimensions[BMatrixType::order.indexOf('i')]);
    static constexpr Dim_size_t sub_output_channels_remainder = (CMatrixType::dimensions[CMatrixType::order.indexOf('o')]);

    static constexpr Dim_size_t input_channels  = sub_input_size * sub_input_channels_count + sub_input_channels_remainder;
    static constexpr Dim_size_t output_channels = sub_output_size * sub_output_channels_count + sub_output_channels_remainder;

    using WeightMatrixTypeRecombined = Matrix<value_type, "OI", output_channels, input_channels>;
};

template <IsMatrixType WeightMatrixType>
struct WeightSubDivideInverseHelper<WeightMatrixType> {
    using WeightMatrixTypeRecombined = WeightMatrixType;
};

template <typename WeightMatrixSplitType>
using InverseWeightSubBioMatrixType = typename WeightSubDivideInverseHelper<WeightMatrixSplitType>::WeightMatrixTypeRecombined;

template <typename Matrices>
__attribute__((always_inline)) constexpr inline auto inverseWeightSubBio(const Matrices &weights) {
    static_assert(std::tuple_size<std::remove_cvref_t<Matrices>>::value == 4, "weights must be a tuple of 4 matrices");
    auto & Matrix_A = std::get<0>(weights);
    auto & Matrix_B = std::get<1>(weights);
    auto & Matrix_C = std::get<2>(weights);
    auto & Matrix_D = std::get<3>(weights);

    auto Matrix_A_collapsed = permute<"IO">(collapse<"Oo", "O">(collapse<"Ii", "I">(Matrix_A)));
    auto Matrix_B_collapsed = permute<"IO">(collapse<"Oo", "O">(collapse<"Ii", "I">(Matrix_B)));
    auto Matrix_C_collapsed = permute<"IO">(collapse<"Oo", "O">(collapse<"Ii", "I">(Matrix_C)));
    auto Matrix_D_collapsed = permute<"IO">(collapse<"Oo", "O">(collapse<"Ii", "I">(Matrix_D)));

    auto Matrix_AB = concatenate<0>(std::move(Matrix_A_collapsed), std::move(Matrix_B_collapsed));
    auto Matrix_CD = concatenate<0>(std::move(Matrix_C_collapsed), std::move(Matrix_D_collapsed));

    auto Matrix_A_B_C_D = concatenate<1>(std::move(Matrix_AB), std::move(Matrix_CD));

    return Matrix_A_B_C_D;
}

template <typename MatrixType>
    requires(IsMatrixType<MatrixType>)
__attribute__((always_inline)) constexpr inline auto inverseWeightSubBio(const MatrixType &weights) {
    return std::move(weights); // If the matrix is not a tuple, just return it as is
}

template <typename... Matrices>
__attribute__((always_inline)) constexpr inline auto pack(const Matrices &...weights) {
    static_assert(((std::tuple_size<std::remove_cvref_t<Matrices>>::value == 4) && ...), "weights must be a tuple of 4 matrices");
    using FirstPack                = std::tuple_element_t<0, std::tuple<std::remove_cvref_t<Matrices>...>>;
    constexpr Dim_size_t alignment = FirstPack::align;
    // there are still 4 matrixes per pack, they should split and all packs are fused, then repacked to aligned collection

    return makeAlignedMatrixCollectionNoSafeGuard<alignment>(      // forced linebreak
            (fuse(materialize(std::get<0>(weights))...)), // forced linebreak
            (fuse(materialize(std::get<1>(weights))...)), // forced linebreak
            (fuse(materialize(std::get<2>(weights))...)), // forced linebreak
            (fuse(materialize(std::get<3>(weights))...)));
}

template <typename... Matrices>
    requires((IsMatrixType<Matrices> && ...))
__attribute__((always_inline)) constexpr inline auto pack(const Matrices &... weights) {
    return materialize(fuse(weights...));   // If the matrices are not tuples, just fuse them as is
}

template <Dim_size_t SuggestedSubInputChannels, Dim_size_t SuggestedSubOutputChannels, IsMatrixType... WeightMatrixType>
    requires((std::remove_cvref_t<WeightMatrixType>::order.remove("IO").length() == 0)&& ...)
__attribute__((always_inline)) constexpr inline auto weightSubBioEarlyFusion(const WeightMatrixType &... weights) {
    return weightSubBio<SuggestedSubInputChannels,SuggestedSubOutputChannels>(fuse(weights...));
}

template <Dim_size_t SuggestedSubInputChannels, Dim_size_t SuggestedSubOutputChannels, IsMatrixType... WeightMatrixType>
    requires((std::remove_cvref_t<WeightMatrixType>::order.remove("IO").length() == 0)&& ...)
__attribute__((always_inline)) constexpr inline auto weightSubBioLateFusion(const WeightMatrixType &... weights) {
    return pack(weightSubBio<SuggestedSubInputChannels,SuggestedSubOutputChannels>(weights)...);
}



template <std::size_t SuggestedSubBatchSize                         = 1,
          template <typename, typename, typename> class MACOperator = DefaultMACOperation,
          IsMatrixType InputMatrixType                              = Matrix<int, "BC", 1, 1>,
          IsMatrixType OutputMatrixType                             = Matrix<int, "BC", 1, 1>,
          tuple_like   WeightMatrixType                             = AlignedMatrixCollection<4, Matrix<int, "OIoi", 1, 1, 1, 1>>,
          IsMatrixType BiasMatrixType                               = Matrix<int, "BC", 1, 1>,
          typename Lambda                                           = decltype([]() {}),
          IsMatrixType... ActivationInformationMatrixType>
    requires(std::tuple_size_v<std::remove_cvref_t<WeightMatrixType>> == 4)
#if __LINEAR_FORCE_INLINE__
__attribute__((always_inline)) inline // Force inlining for performance
#else
inline // Let the compiler decide the inlining
#endif
        void
        Linear( // Function Parameters
                const InputMatrixType  &Input,
                OutputMatrixType       &Output,
                const WeightMatrixType &Weights,
                const BiasMatrixType   &Bias,
                const Lambda           &Act,
                const ActivationInformationMatrixType &...ActivationParameters) {

    using BaseWeightMatrixType = decltype(materialize(inverseWeightSubBio(Weights)));
    static_assert(InputMatrixType::order.contains('C'), "InputMatrixType must contain 'C' for input channels");
    static_assert(OutputMatrixType::order.contains('C'), "OutputMatrixType must contain 'O' for output channels");
    static_assert(BaseWeightMatrixType::order.contains('I'), "BaseWeightMatrixType must contain 'I' for input channels");
    static_assert(BaseWeightMatrixType::order.contains('O'), "BaseWeightMatrixType must contain 'O' for output channels");
    static_assert(BiasMatrixType::order.contains('C') != BiasMatrixType::order.contains('E'),
                  "BiasMatrixType must contain 'C' for output channels or 'E' single element bias parameters, not both not neither");
    static_assert(InputMatrixType::order.contains('B') == OutputMatrixType::order.contains('B'), "Either both Input and Output must contain 'B' for batch or neither");
    static_assert(((ActivationInformationMatrixType::order.contains('C') || ActivationInformationMatrixType::order.contains('E')) && ...),
                  "ActivationParameters must contain 'C' for output channels or 'E' single element activation parameters");

    static_assert(InputMatrixType::order.remove("BC").length() == 0, "Input may only use the dimensions 'BC' (Batch, Channel)");
    static_assert(OutputMatrixType::order.remove("BC").length() == 0, "Output may only use the dimensions 'BC' (Batch Channel)");
    static_assert(BaseWeightMatrixType::order.remove("BIiOo").length() == 0, "Weights may only use the dimensions 'BbIiOo' (Batch, sub-Batch, Input, sub-Input, Output, sub-Output)");
    static_assert(BiasMatrixType::order.remove("BCE").length() == 0, "Bias may only use the dimensions 'BCE' (Batch, Channel, Element)");
    static_assert(((ActivationInformationMatrixType::order.remove("BCE").length() == 0) && ...), "ActivationParameters may have the dimensions 'BbCc' (Batch, Channel) or only 'E' (Element)");

    const auto &weight_matrix_a = std::get<0>(Weights);
    const auto &weight_matrix_b = std::get<1>(Weights);
    const auto &weight_matrix_c = std::get<2>(Weights);
    const auto &weight_matrix_d = std::get<3>(Weights);

    using WeightMatrixTypeA = std::remove_cvref_t<std::tuple_element_t<0, WeightMatrixType>>;
    using WeightMatrixTypeB = std::remove_cvref_t<std::tuple_element_t<1, WeightMatrixType>>;
    using WeightMatrixTypeC = std::remove_cvref_t<std::tuple_element_t<2, WeightMatrixType>>;
    using WeightMatrixTypeD = std::remove_cvref_t<std::tuple_element_t<3, WeightMatrixType>>;

    constexpr Dim_size_t batch_size     = (InputMatrixType::order.contains('B') ? InputMatrixType::dimensions[InputMatrixType::order.indexOf('B')] : 1);
    constexpr Dim_size_t batch_size_tmp = (OutputMatrixType::order.contains('B') ? OutputMatrixType::dimensions[OutputMatrixType::order.indexOf('B')] : 1);

    constexpr Dim_size_t input_channels              = (InputMatrixType::order.contains('C') ? InputMatrixType::dimensions[InputMatrixType::order.indexOf('C')] : 1);
    constexpr Dim_size_t input_channels_tmp          = (BaseWeightMatrixType::order.contains('I') ? BaseWeightMatrixType::dimensions[BaseWeightMatrixType::order.indexOf('I')] : 1);
    constexpr Dim_size_t sub_input_channels          = (WeightMatrixTypeA::order.contains('i') ? WeightMatrixTypeA::dimensions[WeightMatrixTypeA::order.indexOf('i')] : 1);
    constexpr Dim_size_t sub_input_channels_rest     = (WeightMatrixTypeB::order.contains('i') ? WeightMatrixTypeB::dimensions[WeightMatrixTypeB::order.indexOf('i')] : 1);
    constexpr Dim_size_t sub_input_channels_rest_tmp = (WeightMatrixTypeD::order.contains('i') ? WeightMatrixTypeD::dimensions[WeightMatrixTypeD::order.indexOf('i')] : 1);

    constexpr Dim_size_t output_channels              = (OutputMatrixType::order.contains('C') ? OutputMatrixType::dimensions[OutputMatrixType::order.indexOf('C')] : 1);
    constexpr Dim_size_t output_channels_tmp          = (BaseWeightMatrixType::order.contains('O') ? BaseWeightMatrixType::dimensions[BaseWeightMatrixType::order.indexOf('O')] : 1);
    constexpr Dim_size_t sub_output_channels          = (WeightMatrixTypeA::order.contains('o') ? WeightMatrixTypeA::dimensions[WeightMatrixTypeA::order.indexOf('o')] : 1);
    constexpr Dim_size_t sub_output_channels_rest     = (WeightMatrixTypeC::order.contains('o') ? WeightMatrixTypeC::dimensions[WeightMatrixTypeC::order.indexOf('o')] : 1);
    constexpr Dim_size_t sub_output_channels_rest_tmp = (WeightMatrixTypeD::order.contains('o') ? WeightMatrixTypeD::dimensions[WeightMatrixTypeD::order.indexOf('o')] : 1);
    constexpr Dim_size_t output_channels_tmp_2        = (BiasMatrixType::order.contains('C') ? BiasMatrixType::dimensions[BiasMatrixType::order.indexOf('C')] : output_channels_tmp);

    // check for correct dimensions
    static_assert(batch_size == batch_size_tmp, "Batch size of Input and Output must match");
    static_assert(input_channels == input_channels_tmp, "Input channels of Input and Weights must match");
    static_assert(input_channels % sub_input_channels == sub_input_channels_rest, "Input channels of Input and Weights must match sub-Input channels");
    static_assert(output_channels == output_channels_tmp_2, "Output channels of Output and Bias must match");
    static_assert(output_channels % sub_output_channels == sub_output_channels_rest, "Output channels of Output and Weights must match sub-Output channels");
    static_assert(output_channels == output_channels_tmp, "Output channels of Output and Weights must match");
    static_assert((((ActivationInformationMatrixType::order.contains('E') ? ActivationInformationMatrixType::dimensions[BiasMatrixType::order.indexOf('E')] : 1) == 1) && ...),
                  "ActivationParameters using single value 'E' must have a dimension of 1");

    static_assert(sub_input_channels_rest == sub_input_channels_rest_tmp, "Sub-Input channels of Weight Matrix B and D must match");
    static_assert(sub_output_channels_rest == sub_output_channels_rest_tmp, "Sub-Output channels of Weight Matrix C and D must match");

    // change interpretations of passed matrices
    const auto input_broadcasted     = conditionalBroadcast<"B", {batch_size}>(Input);
    const auto bias_broadcasted      = conditionalBroadcast<"B", {batch_size}>(conditionalReplace<"E", "C">(conditionalReplicate<"E", {output_channels}>(Bias)));
    const auto weights_a_broadcasted = conditionalBroadcast<"B", {batch_size}>(weight_matrix_a);
    const auto weights_b_broadcasted = conditionalBroadcast<"B", {batch_size}>(weight_matrix_b);
    const auto weights_c_broadcasted = conditionalBroadcast<"B", {batch_size}>(weight_matrix_c);
    const auto weights_d_broadcasted = conditionalBroadcast<"B", {batch_size}>(weight_matrix_d);
    auto       out_broadcasted       = conditionalBroadcast<"B", {batch_size}>(Output);

    constexpr Dim_size_t sub_batch_size      = ((batch_size >= SuggestedSubBatchSize) ? SuggestedSubBatchSize : batch_size);
    constexpr Dim_size_t sub_batch_size_rest = batch_size % sub_batch_size;

    using MACOperator_ = MACOperator<typename InputMatrixType::value_type, typename WeightMatrixTypeA::value_type, typename BiasMatrixType::value_type>;

    // using AccumulationType = typename BiasMatrixType::value_type;
    using AccumulationType = typename MACOperator_::AccumulationType_;

// Unrolled Sub Batch
#pragma GCC unroll(1)
    for (Dim_size_t batch = 0; batch < batch_size - sub_batch_size_rest; batch += sub_batch_size) {
#pragma GCC unroll(1)
        for (Dim_size_t output_channel = 0; output_channel < output_channels - sub_output_channels_rest; output_channel += sub_output_channels) {
            // slice to subbatch
            // slice output channels
            const auto input_sliced_batch   = replace<"BC", "bc">(slice<"B", sub_batch_size>(input_broadcasted, {batch}));
            const auto weights_sliced_batch = replace<"B", "b">(collapse<"Oo", "o">(slice<"BO", sub_batch_size, 1>(weights_a_broadcasted, {batch, output_channel / sub_output_channels})));
            const auto bias_sliced          = replace<"BC", "bo">(slice<"BC", sub_batch_size, sub_output_channels>(bias_broadcasted, {batch, output_channel}));
            auto       out_sliced           = replace<"BC", "bo">(slice<"BC", sub_batch_size, sub_output_channels>(out_broadcasted, {batch, output_channel}));

            auto sum_matrix = materializeUnrolled(permute<"bo">(bias_sliced), MACOperator_::pre_processing);
#pragma GCC unroll(1)
            for (Dim_size_t input_channel = 0; input_channel < input_channels - sub_input_channels_rest; input_channel += sub_input_channels) {
                const auto input_sliced_channel   = replace<"c", "i">(slice<"c", sub_input_channels>(input_sliced_batch, {input_channel}));
                const auto weights_sliced_channel = collapse<"Ii", "i">(slice<"I", 1>(weights_sliced_batch, {input_channel / sub_input_channels}));

                // // Perform Tensor MAC
                MACOperator_::op(sum_matrix, input_sliced_channel, weights_sliced_channel);
            }
            if constexpr (sub_input_channels_rest > 0) {
                // take care of matrix b
                const auto weights_b_sliced_batch =
                        replace<"B", "b">(collapse<"Ii", "i">(collapse<"Oo", "o">(slice<"BO", sub_batch_size, 1>(weights_b_broadcasted, {batch, output_channel / sub_output_channels}))));
                const auto input_sliced_channel_rest = replace<"c", "i">(slice<"c", sub_input_channels_rest>(input_sliced_batch, {input_channels - sub_input_channels_rest}));

                // // Perform Tensor MAC
                MACOperator_::op(sum_matrix, input_sliced_channel_rest, weights_b_sliced_batch);
            }

            // activation_parameters reinterpretations
            [[maybe_unused]]
            const auto broadcast_permute = [=](const auto &matrix) {
                return replace<"BC", "bo">(slice<"BC", sub_batch_size, sub_output_channels>(
                        conditionalBroadcast<"B", {batch_size}>(conditionalReplace<"E", "C">(conditionalReplicate<"E", {output_channels}>(matrix))), {batch, output_channel}));
            };

            // Apply activation function
            loopUnrolled([=](auto &&a, const AccumulationType &b, const auto &...activation_parameters) { a = Act(MACOperator_::post_processing(b), activation_parameters...); }, out_sliced,
                         sum_matrix, broadcast_permute(ActivationParameters)...);
        }
        if constexpr (sub_output_channels_rest > 0) {
            // slice output channels
            // take care of weight matrix c and d
            const auto input_sliced_batch = replace<"BC", "bc">(slice<"B", sub_batch_size>(input_broadcasted, {batch}));
            const auto weights_c_batch    = replace<"B", "b">(collapse<"Oo", "o">(slice<"BO", sub_batch_size, 1>(weights_c_broadcasted, {batch, 0})));
            const auto bias_sliced        = replace<"BC", "bo">(slice<"BC", sub_batch_size, sub_output_channels_rest>(bias_broadcasted, {batch, output_channels - sub_output_channels_rest}));
            auto       out_sliced         = replace<"BC", "bo">(slice<"BC", sub_batch_size, sub_output_channels_rest>(out_broadcasted, {batch, output_channels - sub_output_channels_rest}));

            auto sum_matrix = materializeUnrolled(permute<"bo">(bias_sliced), MACOperator_::pre_processing);
#pragma GCC unroll(1)
            for (Dim_size_t input_channel = 0; input_channel < input_channels - sub_input_channels_rest; input_channel += sub_input_channels) {
                const auto input_sliced_channel   = replace<"c", "i">(slice<"c", sub_input_channels>(input_sliced_batch, {input_channel}));
                const auto weights_sliced_channel = collapse<"Ii", "i">(slice<"I", 1>(weights_c_batch, {input_channel / sub_input_channels}));
                // // Perform Tensor MAC
                MACOperator_::op(sum_matrix, input_sliced_channel, weights_sliced_channel);
            }
            if constexpr (sub_input_channels_rest > 0) {
                // take care of matrix d
                const auto weights_d_batch           = replace<"B", "b">(collapse<"Ii", "i">(collapse<"Oo", "o">(slice<"BO", sub_batch_size, 1>(weights_d_broadcasted, {batch, 0}))));
                const auto input_sliced_channel_rest = replace<"c", "i">(slice<"c", sub_input_channels_rest>(input_sliced_batch, {input_channels - sub_input_channels_rest}));
                // // Perform Tensor MAC
                MACOperator_::op(sum_matrix, input_sliced_channel_rest, weights_d_batch);
            }

            // activation_parameters reinterpretations
            [[maybe_unused]]
            const auto broadcast_permute = [=](const auto &matrix) {
                return replace<"BC", "bo">(slice<"BC", sub_batch_size, sub_output_channels_rest>(
                        conditionalBroadcast<"B", {batch_size}>(conditionalReplace<"E", "C">(conditionalReplicate<"E", {output_channels}>(matrix))),
                        {batch, output_channels - sub_output_channels_rest}));
            };
            // Apply activation function
            loopUnrolled([&](auto &&a, const AccumulationType &b, const auto &...activation_parameters) { a = Act(MACOperator_::post_processing(b), activation_parameters...); }, out_sliced,
                         sum_matrix, broadcast_permute(ActivationParameters)...);
        }
    }

    // Rest Batch
    if constexpr (sub_batch_size_rest > 0) {
#pragma GCC unroll(1)
        for (Dim_size_t output_channel = 0; output_channel < output_channels - sub_output_channels_rest; output_channel += sub_output_channels) {
            // slice to subbatch
            // slice output channels
            const auto input_sliced_batch = replace<"BC", "bc">(slice<"B", sub_batch_size_rest>(input_broadcasted, {batch_size - sub_batch_size_rest}));
            const auto weights_sliced_batch =
                    replace<"B", "b">(collapse<"Oo", "o">(slice<"BO", sub_batch_size_rest, 1>(weights_a_broadcasted, {batch_size - sub_batch_size_rest, output_channel / sub_output_channels})));
            const auto bias_sliced = replace<"BC", "bo">(slice<"BC", sub_batch_size_rest, sub_output_channels>(bias_broadcasted, {batch_size - sub_batch_size_rest, output_channel}));
            auto       out_sliced  = replace<"BC", "bo">(slice<"BC", sub_batch_size_rest, sub_output_channels>(out_broadcasted, {batch_size - sub_batch_size_rest, output_channel}));

            auto sum_matrix = materializeUnrolled(permute<"bo">(bias_sliced), MACOperator_::pre_processing);
#pragma GCC unroll(1)
            for (Dim_size_t input_channel = 0; input_channel < input_channels - sub_input_channels_rest; input_channel += sub_input_channels) {
                const auto input_sliced_channel   = replace<"c", "i">(slice<"c", sub_input_channels>(input_sliced_batch, {input_channel}));
                const auto weights_sliced_channel = collapse<"Ii", "i">(slice<"I", 1>(weights_sliced_batch, {input_channel / sub_input_channels}));

                // // Perform Tensor MAC
                MACOperator_::op(sum_matrix, input_sliced_channel, weights_sliced_channel);
            }
            if constexpr (sub_input_channels_rest > 0) {
                // take care of matrix b
                const auto weights_b_sliced_batch = replace<"B", "b">(
                        collapse<"Ii", "i">(collapse<"Oo", "o">(slice<"BO", sub_batch_size_rest, 1>(weights_b_broadcasted, {batch_size - sub_batch_size_rest, output_channel / sub_output_channels}))));
                const auto input_sliced_channel_rest = replace<"c", "i">(slice<"c", sub_input_channels_rest>(input_sliced_batch, {input_channels - sub_input_channels_rest}));

                // // Perform Tensor MAC
                MACOperator_::op(sum_matrix, input_sliced_channel_rest, weights_b_sliced_batch);
            }

            // activation_parameters reinterpretations
            [[maybe_unused]]
            const auto broadcast_permute = [=](const auto &matrix) {
                return replace<"BC", "bo">(slice<"BC", sub_batch_size_rest, sub_output_channels>(
                        conditionalBroadcast<"B", {batch_size}>(conditionalReplace<"E", "C">(conditionalReplicate<"E", {output_channels}>(matrix))),
                        {batch_size - sub_batch_size_rest, output_channel}));
            };
            // Apply activation function
            loopUnrolled([&](auto &&a, const AccumulationType &b, const auto &...activation_parameters) { a = Act(MACOperator_::post_processing(b), activation_parameters...); }, out_sliced,
                         sum_matrix, broadcast_permute(ActivationParameters)...);
        }
        if constexpr (sub_output_channels_rest > 0) {
            // slice output channels
            // take care of weight matrix c and d
            const auto input_sliced_batch = replace<"BC", "bc">(slice<"B", sub_batch_size_rest>(input_broadcasted, {batch_size - sub_batch_size_rest}));
            const auto weights_c_batch    = replace<"B", "b">(collapse<"Oo", "o">(slice<"BO", sub_batch_size_rest, 1>(weights_c_broadcasted, {batch_size - sub_batch_size_rest, 0})));
            const auto bias_sliced =
                    replace<"BC", "bo">(slice<"BC", sub_batch_size_rest, sub_output_channels_rest>(bias_broadcasted, {batch_size - sub_batch_size_rest, output_channels - sub_output_channels_rest}));
            auto out_sliced =
                    replace<"BC", "bo">(slice<"BC", sub_batch_size_rest, sub_output_channels_rest>(out_broadcasted, {batch_size - sub_batch_size_rest, output_channels - sub_output_channels_rest}));

            auto sum_matrix = materializeUnrolled(permute<"bo">(bias_sliced), MACOperator_::pre_processing);
#pragma GCC unroll(1)
            for (Dim_size_t input_channel = 0; input_channel < input_channels - sub_input_channels_rest; input_channel += sub_input_channels) {
                const auto input_sliced_channel   = replace<"c", "i">(slice<"c", sub_input_channels>(input_sliced_batch, {input_channel}));
                const auto weights_sliced_channel = collapse<"Ii", "i">(slice<"I", 1>(weights_c_batch, {input_channel / sub_input_channels}));
                // // Perform Tensor MAC
                MACOperator_::op(sum_matrix, input_sliced_channel, weights_sliced_channel);
            }
            if constexpr (sub_input_channels_rest > 0) {
                // take care of matrix d
                const auto weights_d_batch =
                        replace<"B", "b">(collapse<"Ii", "i">(collapse<"Oo", "o">(slice<"BO", sub_batch_size_rest, 1>(weights_d_broadcasted, {batch_size - sub_batch_size_rest, 0}))));
                const auto input_sliced_channel_rest = replace<"c", "i">(slice<"c", sub_input_channels_rest>(input_sliced_batch, {input_channels - sub_input_channels_rest}));
                // // Perform Tensor MAC
                MACOperator_::op(sum_matrix, input_sliced_channel_rest, weights_d_batch);
            }

            // activation_parameters reinterpretations
            [[maybe_unused]]
            const auto broadcast_permute = [=](const auto &matrix) {
                return replace<"BC", "bo">(slice<"BC", sub_batch_size_rest, sub_output_channels_rest>(
                        conditionalBroadcast<"B", {batch_size}>(conditionalReplace<"E", "C">(conditionalReplicate<"E", {output_channels}>(matrix))),
                        {batch_size - sub_batch_size_rest, output_channels - sub_output_channels_rest}));
            };
            // Apply activation function
            loopUnrolled([=](auto &&a, const AccumulationType &b, const auto &...activation_parameters) { a = Act(MACOperator_::post_processing(b), activation_parameters...); }, out_sliced,
                         sum_matrix, broadcast_permute(ActivationParameters)...);
        }
    }
}

}; // namespace functions::linear
