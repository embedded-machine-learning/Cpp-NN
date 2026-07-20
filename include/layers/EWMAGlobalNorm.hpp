#pragma once

#include <algorithm>
#include <concepts>
#include <cstdlib>
#include <iostream>
#include <stddef.h>
#include <type_traits>
#include <utility>

#include "../Matrix.hpp"
#include "./BaseLayer.hpp"

namespace layers {

template < // Foreced linebreak
        DimensionOrder SmoothingOrder     = "S",
        DimensionOrder ReductionOrder     = "C",
        typename OutputType               = float,
        typename WeightMatrixType         = Matrix<float, "E", 1>,
        typename InputReductionResetType  = decltype([](auto &ret) { ret = 0; }),
        typename InputReductionLambdaType = decltype([](auto &ret, const auto &input) { ret = std::max(ret, input); }),
        typename SmoothingLambdaType      = decltype([](auto &ret, const auto &input, const auto &weights) { ret = ret * weights + (static_cast<decltype(weights)>(1) - weights) * input; }),
        typename OutputLambdaType         = decltype([](auto &ret, const auto &state, const auto &input) { ret = input / (state + 1e-6); }),
        IsMatrixType... ActivationMatrixInformation>
class EWMAGlobalNormLayer {
  private:
    using WeightMatrixType_ = std::remove_cvref_t<WeightMatrixType>;

    static_assert(!WeightMatrixType_::order.containsAny(SmoothingOrder), "The weight matrix may not contain the SmoothingOrder in its order");
    static_assert(SmoothingOrder.length() == 1, "SmoothingOrder must have a length of 1");

    using WeightMatrixType_stored = std::remove_cvref_t<WeightMatrixType>;

  public:
    const WeightMatrixType_stored                                               weights_;
    const InputReductionResetType                                               input_reset_lambda_;
    const InputReductionLambdaType                                              input_reduction_lambda_;
    const SmoothingLambdaType                                                   smoothing_lambda_;
    const OutputLambdaType                                                      output_lambda_;
    const std::tuple<const std::remove_cvref_t<ActivationMatrixInformation>...> activation_parameters_;

    using ExampleInputMatrix = Matrix<float, SmoothingOrder + "C", 1, 1>;
    template <IsMatrixType InputMatrix>
    using OutputMatrix = MaterializedMatrix<InputMatrix>;

    template <IsMatrixType InputMatrix>
    using StateMatrixType = MaterializedMatrix<OverrideDimensionMatrix<OverrideRemoveDimensionMatrix<MaterializedMatrix<InputMatrix>, SmoothingOrder>, ReductionOrder, 1>>;

    template <IsMatrixType InputMatrix>
    using ReducedInputType = MaterializedMatrix<OverrideDimensionMatrix<OverrideRemoveDimensionMatrix<MaterializedMatrix<InputMatrix>, SmoothingOrder>, ReductionOrder, 1>>;

    // Memory Requirements of forward pass
    template <IsMatrixType InputMatrix>
    static constexpr std::size_t memory_minimal = sizeof(MaterializedMatrix<InputMatrix>) + sizeof(MaterializedMatrix<ReducedInputType<InputMatrix>>);
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
    constexpr EWMAGlobalNormLayer( // Forced Linebreak
            WeightMatrixType         &&Weights,
            InputReductionResetType  &&InputReductionReset,
            InputReductionLambdaType &&InputReductionLambda,
            SmoothingLambdaType      &&SmoothingLambda,
            OutputLambdaType         &&OutputLambda,
            ActivationMatrixInformation &&...ActivationParameters) noexcept
            : weights_(std::forward<WeightMatrixType>(Weights)), input_reset_lambda_(std::forward<InputReductionResetType>(InputReductionReset)),
              input_reduction_lambda_(std::forward<InputReductionLambdaType>(InputReductionLambda)), smoothing_lambda_(std::forward<SmoothingLambdaType>(SmoothingLambda)),
              output_lambda_(std::forward<OutputLambdaType>(OutputLambda)), activation_parameters_(std::forward<ActivationMatrixInformation>(ActivationParameters)...) {};

    template <bool ContinueAfter = true, IsMatrixType InputMatrixType, IsMatrixType OutputMatrixType, IsBaseMatrixType BufferMatrixType, IsBaseMatrixType PermanentMatrixType, std::size_t... I>
    __attribute__((always_inline)) inline void operator()(const InputMatrixType &Input,
                                                          OutputMatrixType      &Out,
                                                          BufferMatrixType      &buffer,
                                                          PermanentMatrixType   &permanent,
                                                          const std::index_sequence<I...> = std::make_index_sequence<sizeof...(ActivationMatrixInformation)>()) const noexcept {
        static_assert(InputMatrixType::order.containsAll(SmoothingOrder), "Input must contain the SmoothingOrder in its order");
        static_assert(OutputMatrixType::order.containsAny(SmoothingOrder), "Output must contain the SmoothingOrder in its order");
        static_assert(InputMatrixType::order.containsAll(OutputMatrixType::order), "Input must contain all dimensions of Output");
        static_assert(InputMatrixType::dimensions == PermutedMatrix<InputMatrixType::order, OutputMatrixType>::dimensions, "Dimensions of Input and Output must match after permutation");

        static_assert(sizeof(permanent.data) >= sizeof(StateMatrixType<InputMatrixType>), "Permanent Memory Size does not match the required size for OutputMatrixType");

        auto &state = *reinterpret_cast<StateMatrixType<InputMatrixType> *>(&permanent.data[0]);
        auto state_expanded = permute<InputMatrixType::order>(conditionalBroadcast<SmoothingOrder, {InputMatrixType::dimensions[InputMatrixType::order.indexOf(SmoothingOrder.order[0])]}>(state));
        
        auto &InputReduced = *reinterpret_cast<ReducedInputType<InputMatrixType> *>(&buffer.data[0]);
        auto  InputReduced_expanded =
                permute<InputMatrixType::order>(conditionalBroadcast<SmoothingOrder, {InputMatrixType::dimensions[InputMatrixType::order.indexOf(SmoothingOrder.order[0])]}>(InputReduced));

        const auto weights_broadcasted = replicate<SmoothingOrder, {InputMatrixType::dimensions[InputMatrixType::order.indexOf(SmoothingOrder.order[0])]}>(
                conditionalBroadcast<SmoothingOrder, {1}>(conditionalReplace<"E", SmoothingOrder>(weights_)));
        const auto weights_broadcasted_full         = conditionalFullBroadcast<ReducedInputType<InputMatrixType>::order, ReducedInputType<InputMatrixType>::dimensions>(weights_broadcasted);

        constexpr Dim_size_t input_smooting_dimension_size = InputMatrixType::dimensions[InputMatrixType::order.indexOf(SmoothingOrder.order[0])];

        std::cout << "InputReduced Type: " << human_readable_type<decltype(InputReduced)> << std::endl;
        std::cout << "InputReduced_expanded Type: " << human_readable_type<decltype(InputReduced_expanded)> << std::endl;
        std::cout << "InputReduced_expanded Type: " << human_readable_type<MaterializedMatrix<decltype(InputReduced_expanded)>> << std::endl;

        for (Dim_size_t i = 0; i < input_smooting_dimension_size; ++i) {
            auto input_sliced            = slice<SmoothingOrder, 1>(Input, {i});
            auto state_sliced            = slice<SmoothingOrder, 1>(state_expanded, {i});
            auto state_sliced_replicated = replicate<ReductionOrder, {InputMatrixType::dimensions[InputMatrixType::order.indexOf(ReductionOrder.order[0])]}>(state_sliced);
            auto weights_sliced          = slice<SmoothingOrder, 1>(weights_broadcasted_full, {i});
            auto out_sliced              = slice<SmoothingOrder, 1>(Out, {i});

            auto input_reduced_sliced            = slice<SmoothingOrder, 1>(InputReduced_expanded, {i});
            auto input_reduced_sliced_replicated = replicate<ReductionOrder, {InputMatrixType::dimensions[InputMatrixType::order.indexOf(ReductionOrder.order[0])]}>(input_reduced_sliced);

        //     std::cout << "=====================================================================" << std::endl;
        //     std::cout << "input_sliced Type: " << human_readable_type<MaterializedMatrix<decltype(input_sliced)>> << std::endl;
        //     std::cout << "state_sliced Type: " << human_readable_type<MaterializedMatrix<decltype(state_sliced)>> << std::endl;
        //     std::cout << "state_sliced_replicated Type: " << human_readable_type<MaterializedMatrix<decltype(state_sliced_replicated)>> << std::endl;
        //     std::cout << "out_sliced Type: " << human_readable_type<MaterializedMatrix<decltype(out_sliced)>> << std::endl;
        //     std::cout << "input_reduced_sliced Type: " << human_readable_type<MaterializedMatrix<decltype(input_reduced_sliced)>> << std::endl;
            //     std::cout << "input_reduced_sliced_replicated Type: " << human_readable_type<MaterializedMatrix<decltype(input_reduced_sliced_replicated)>> << std::endl;
        //     std::cout << "weights_sliced Type: " << human_readable_type<MaterializedMatrix<decltype(weights_sliced)>> << std::endl;
        //     std::cout << "=====================================================================" << std::endl;

            loopUnrolled(input_reset_lambda_, input_reduced_sliced);
            loopUnrolled(input_reduction_lambda_, input_reduced_sliced_replicated, input_sliced);

            loopUnrolled(smoothing_lambda_, state_sliced, input_reduced_sliced, weights_sliced);

        //     std::cout << "=====================================================================" << std::endl;
        //     std::cout << "input_sliced: " << std::endl;
        //     printNDMatrix(input_sliced);
        //     std::cout << "input_reduced_sliced_replicated: " << std::endl;
        //     printNDMatrix(input_reduced_sliced_replicated);
        //     std::cout << "input_reduced_sliced: " << std::endl;
        //     printNDMatrix(input_reduced_sliced);
        //     std::cout << "state_sliced: " << std::endl;
        //     printNDMatrix(state_sliced);
        //     std::cout << "state_sliced_replicated: " << std::endl;
        //     printNDMatrix(state_sliced_replicated);
        //     std::cout << "weights_sliced: " << std::endl;
        //     printNDMatrix(weights_sliced);

            if constexpr (ContinueAfter) {
                loopUnrolled(output_lambda_, out_sliced, state_sliced_replicated, input_sliced, std::get<I>(activation_parameters_)...);
            }
        }
    }
};

static_assert(IsValidLayer<EWMAGlobalNormLayer<>>, "BaseLayer does not meet the requirements of a valid layer");

template < // Foreced linebreak
        DimensionOrder SmoothingOrder     = "S",
        DimensionOrder ReductionOrder     = "C",
        typename OutputType               = float,
        typename WeightMatrixType         = Matrix<float, "E", 1>,
        typename InputReductionLambdaType = decltype([](auto &ret, const auto &input) { ret = std::max(ret, input); }),
        typename InputReductionResetType  = decltype([](auto &ret) { ret = 0; }),
        typename SmoothingLambdaType      = decltype([](auto &ret, const auto &input, const auto &weights) { ret = ret * weights + (static_cast<decltype(weights)>(1) - weights) * input; }),
        typename OutputLambdaType         = decltype([](auto       &ret,
                                                const auto &state,
                                                [[maybe_unused]]
                                                const auto &input) { ret = input / (state + 1e-6); }),
        IsMatrixType... ActivationMatrixInformation>
__attribute__((always_inline)) inline constexpr auto EWMAGlobalNorm(
        WeightMatrixType         &&Weights,
        InputReductionResetType  &&InputReductionReset  ,
        InputReductionLambdaType &&InputReductionLambda ,
        SmoothingLambdaType      &&SmoothingLambda,
        OutputLambdaType         &&OutputLambda,
        ActivationMatrixInformation &&...ActivationParameters) noexcept {
    return EWMAGlobalNormLayer<                                                                       // Forced Linebreak
            SmoothingOrder, ReductionOrder,                                                           // Orders
            OutputType, WeightMatrixType,                                                             // datatypes
            InputReductionResetType, InputReductionLambdaType, SmoothingLambdaType, OutputLambdaType, // Lambdas
            ActivationMatrixInformation...>(std::forward<WeightMatrixType>(Weights), std::forward<InputReductionResetType>(InputReductionReset),
                                            std::forward<InputReductionLambdaType>(InputReductionLambda), std::forward<SmoothingLambdaType>(SmoothingLambda),
                                            std::forward<OutputLambdaType>(OutputLambda), std::forward<ActivationMatrixInformation>(ActivationParameters)...);
}

} // namespace layers