#pragma once

#include <concepts>
#include <stddef.h>
#include <type_traits>

#include "../Matrix.hpp"
#include "./BaseLayer.hpp"

namespace layers {

template < // Foreced linebreak
        DimensionOrder SmoothingOrder = "S",
        typename OutputType           = float,
        typename WeightMatrixType     = Matrix<float, "E", 1>,
        typename SmoothingLambdaType  = decltype([](auto &ret, const auto &input, const auto &weights) { ret = ret * weights + (static_cast<decltype(weights)>(1) - weights) * input; }),
        typename OutputLambdaType     = decltype([](auto &ret, const auto &state) { ret = state; }),
        IsMatrixType... ActivationMatrixInformation>
class EWMAReductionLayer {
  private:
    using WeightMatrixType_ = std::remove_cvref_t<WeightMatrixType>;

    static_assert(!WeightMatrixType_::order.containsAny(SmoothingOrder), "The weight matrix may not contain the SmoothingOrder in its order");
    static_assert(SmoothingOrder.length() == 1, "SmoothingOrder must have a length of 1");

    using WeightMatrixType_stored = std::remove_cvref_t<WeightMatrixType>;

  public:
    const WeightMatrixType_stored                                               weights_;
    const SmoothingLambdaType                                                   smoothing_lambda_;
    const OutputLambdaType                                                      output_lambda_;
    const std::tuple<const std::remove_cvref_t<ActivationMatrixInformation>...> activation_parameters_;

    using ExampleInputMatrix = Matrix<float, SmoothingOrder + "C", 1, 1>;
    template <IsMatrixType InputMatrix>
    using OutputMatrix = MaterializedMatrix<InputMatrix>;

    template <IsMatrixType InputMatrix>
    using StateMatrixType = MaterializedMatrix<OverrideRemoveDimensionMatrix<MaterializedMatrix<InputMatrix>, SmoothingOrder>>;

    // Memory Requirements of forward pass
    template <IsMatrixType InputMatrix>
    static constexpr std::size_t memory_minimal = std::max(sizeof(MaterializedMatrix<InputMatrix>), sizeof(MaterializedMatrix<OutputMatrix<InputMatrix>>));
    // Can it reuse the input Memory region?
    static constexpr bool memory_inlined = true;
    // Required Buffer size
    // the amount of Memory required for temporary storage
    template <IsMatrixType InputMatrix>
    static constexpr std::size_t memory_buffer = 0;
    // Permanent Memory which is required for some layers, if used in a time series model
    template <IsMatrixType InputMatrix>
    static constexpr std::size_t memory_permanent = sizeof(MaterializedMatrix<StateMatrixType<InputMatrix>>);

    // Constructor
    constexpr EWMAReductionLayer(WeightMatrixType &&Weights, SmoothingLambdaType &&SmoothingLambda, OutputLambdaType &&OutputLambda, ActivationMatrixInformation &&...ActivationParameters) noexcept
            : weights_(std::forward<WeightMatrixType>(Weights)), smoothing_lambda_(std::forward<SmoothingLambdaType>(SmoothingLambda)), output_lambda_(std::forward<OutputLambdaType>(OutputLambda)),
              activation_parameters_(std::forward<ActivationMatrixInformation>(ActivationParameters)...) {};

    template <bool ContinueAfter = true, IsMatrixType InputMatrixType, IsMatrixType OutputMatrixType, IsBaseMatrixType BufferMatrixType, IsBaseMatrixType PermanentMatrixType, std::size_t... I>
    __attribute__((always_inline)) inline void operator()(const InputMatrixType             &Input,
                                                          OutputMatrixType                  &Out,
                                                          [[maybe_unused]] BufferMatrixType &buffer,
                                                          PermanentMatrixType               &permanent,
                                                          const std::index_sequence<I...> = std::make_index_sequence<sizeof...(ActivationMatrixInformation)>()) const noexcept {
        static_assert(InputMatrixType::order.containsAll(SmoothingOrder), "Input must contain the SmoothingOrder in its order");
        static_assert(OutputMatrixType::order.containsAny(SmoothingOrder), "Output must contain the SmoothingOrder in its order");
        static_assert(InputMatrixType::order.containsAll(OutputMatrixType::order), "Input must contain all dimensions of Output");
        static_assert(InputMatrixType::dimensions == PermutedMatrix<InputMatrixType::order, OutputMatrixType>::dimensions , "Dimensions of Input and Output must match after permutation");

        static_assert(sizeof(permanent.data) >= sizeof(StateMatrixType<InputMatrixType>), "Permanent Memory Size does not match the required size for OutputMatrixType");

        auto &state = *reinterpret_cast<StateMatrixType<InputMatrixType> *>(&permanent.data[0]);

        auto state_expanded = permute<InputMatrixType::order>(conditionalBroadcast<SmoothingOrder, {InputMatrixType::dimensions[InputMatrixType::order.indexOf(SmoothingOrder.order[0])]}>(state));
        const auto weights_broadcasted = replicate<SmoothingOrder, {InputMatrixType::dimensions[InputMatrixType::order.indexOf(SmoothingOrder.order[0])]}>(
                conditionalBroadcast<SmoothingOrder, {1}>(conditionalReplace<"E", SmoothingOrder>(weights_)));
        const auto weights_broadcasted_full = conditionalFullBroadcast<InputMatrixType::order, InputMatrixType::dimensions>(weights_broadcasted);

        constexpr Dim_size_t input_smooting_dimension_size = InputMatrixType::dimensions[InputMatrixType::order.indexOf(SmoothingOrder.order[0])];

        for (Dim_size_t i = 0; i < input_smooting_dimension_size; ++i) {
            auto input_sliced = slice<SmoothingOrder, 1>(Input, {i});
            auto state_sliced = slice<SmoothingOrder, 1>(state_expanded, {i});
            auto out_sliced   = slice<SmoothingOrder, 1>(Out, {i});
            auto weights_sliced = slice<SmoothingOrder, 1>(weights_broadcasted_full, {i});

            loopUnrolled(smoothing_lambda_, state_sliced, input_sliced, weights_sliced);

            if constexpr (ContinueAfter) {
                loopUnrolled(output_lambda_, out_sliced, state_sliced, std::get<I>(activation_parameters_)...);
            }
        }
    }
};

static_assert(IsValidLayer<EWMAReductionLayer<>>, "BaseLayer does not meet the requirements of a valid layer");

template < // Foreced linebreak
        DimensionOrder SmoothingOrder = "S",
        typename OutputType           = float,
        typename WeightMatrixType     = Matrix<float, "E", 1>,
        typename SmoothingLambdaType  = decltype([](auto &ret, const auto &input, const auto &weights) { ret = ret * weights + (static_cast<decltype(weights)>(1) - weights) * input; }),
        typename OutputLambdaType     = decltype([](auto &ret, const auto &state) { ret = state; }),
        IsMatrixType... ActivationMatrixInformation>
__attribute__((always_inline)) inline constexpr auto EWMAReduction(WeightMatrixType    &&Weights,
                                                                   SmoothingLambdaType &&SmoothingLambda = {},
                                                                   OutputLambdaType    &&OutputLambda    = {},
                                                                   ActivationMatrixInformation &&...ActivationParameters) noexcept {
    return EWMAReductionLayer<SmoothingOrder, OutputType, WeightMatrixType, SmoothingLambdaType, OutputLambdaType, ActivationMatrixInformation...>(
            std::forward<WeightMatrixType>(Weights), std::forward<SmoothingLambdaType>(SmoothingLambda), std::forward<OutputLambdaType>(OutputLambda),
            std::forward<ActivationMatrixInformation>(ActivationParameters)...);
}

} // namespace layers