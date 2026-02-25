#pragma once

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "../MAC.hpp"
#include "../Matrix.hpp"
#include "../functions/linear.hpp"
#include "BaseLayer.hpp"

namespace layers {
/*
Class for the Linear Layer
*/
template <typename OutputType                                       = float,
          std::size_t    SuggestedSubBatchSize                      = 1,
          DimensionOrder CollapsedOrder                             = "B", // All Dimensions that will be collapsed to 'B'
          template <typename, typename, typename> class MACOperator = DefaultMACOperation,
          typename WeightMatrixType                                 = Matrix<float, "OI", 1, 1>,
          IsMatrixType BiasMatrixType                               = Matrix<OutputType, "BC", 1, 1>,
          typename Lambda                                           = decltype([]() {}),
          IsMatrixType... ActivationMatrixInformation>
class LinearLayer {
  private:
    using WeightMatrixType_ = std::remove_cvref_t<WeightMatrixType>;
    using BiasMatrixType_   = std::remove_cvref_t<BiasMatrixType>;

    // static_assert(IsAlignedMatrixCollection<WeightMatrixType> || IsMatrixType<WeightMatrixType>, "WeightMatrixType must be an AlignedMatrixCollection");
    using WeightMatrixTypeCollapsed = typename functions::linear::InverseWeightSubBioMatrixType<WeightMatrixType_>;
    static_assert(WeightMatrixTypeCollapsed::order.remove("IO").length() == 0, "Weights may only have IO in a colapsed way");
    static_assert(BiasMatrixType_::order.remove("BCE").length() == 0, "BiasMatrixType must be in the order of 'BC' (Batch, Channel) or 'C' (Channel) only or 'E' (Element) only");

    using WeightMatrixTypeCollapsedOrdered = MaterializedMatrix<PermutedMatrix<"OI", WeightMatrixTypeCollapsed>>;

    using AccumulationType                                = BiasMatrixType_::value_type;
    constexpr static Dim_size_t  input_channels           = WeightMatrixTypeCollapsedOrdered::dimensions[1];
    constexpr static Dim_size_t  output_channels          = WeightMatrixTypeCollapsedOrdered::dimensions[0];
    constexpr static std::size_t suggested_sub_batch_size = SuggestedSubBatchSize;

    using BiasMatrixType_stored   = std::remove_cvref_t<BiasMatrixType>;
    using WeightMatrixType_stored = std::remove_cvref_t<WeightMatrixType>;

  public:
    /* Layer store Data */
    const BiasMatrixType_stored   bias_;
    const WeightMatrixType_stored weights_;

    const Lambda                                                                Act;
    const std::tuple<const std::remove_cvref_t<ActivationMatrixInformation>...> activation_parameters_;

    // Type information
    using ExampleInputMatrix = Matrix<AccumulationType, "BC", 1, input_channels>;
    template <IsMatrixType InputMatrix>
    using OutputMatrix = OverrideDimensionMatrix<InputMatrix, "C", output_channels>;

    using BufferMatrix = Matrix<char, "E", 0>;

    // Memory Requirements of forward pass
    template <IsMatrixType InputMatrix>
    static constexpr size_t memory_minimal = sizeof(MaterializedMatrix<InputMatrix>) + sizeof(MaterializedMatrix<OutputMatrix<InputMatrix>>);
    // Can it reuse the input Memory region?
    static constexpr bool memory_inlined = false;
    // Required Buffer size
    template <IsMatrixType InputMatrix>
    static constexpr size_t memory_buffer = 0;
    // Permanent Memory, Linear layers do not require permanent memory
    template <IsMatrixType InputMatrix>
    static constexpr size_t memory_permanent = 0;

    // Constructor
    constexpr LinearLayer(WeightMatrixType &&Weights, BiasMatrixType &&Bias, Lambda &&Act, ActivationMatrixInformation &...ActivationParameters) noexcept
            : bias_(std::forward<BiasMatrixType>(Bias)), weights_(std::forward<WeightMatrixType>(Weights)), Act(std::forward<Lambda>(Act)),
              activation_parameters_(std::forward<ActivationMatrixInformation>(ActivationParameters)...) {};

    template <bool             ContinueAfter             = true,
              std::size_t      UsedSuggestedSubBatchSize = suggested_sub_batch_size,
              IsMatrixType     InputMatrixType,
              IsMatrixType     OutputMatrixType,
              IsBaseMatrixType BufferMatrixType    = Matrix<char, "E", 0>,
              IsBaseMatrixType PermanentMatrixType = Matrix<char, "E", 0>,
              std::size_t... I>
    __attribute__((always_inline)) inline void operator()(const InputMatrixType &Input,
                                                          OutputMatrixType      &Out,
                                                          BufferMatrixType             && = {},
                                                          PermanentMatrixType          && = {},
                                                          const std::index_sequence<I...> = std::make_index_sequence<sizeof...(ActivationMatrixInformation)>()) const noexcept {
        static_assert(InputMatrixType::order.remove("C").remove(CollapsedOrder).length() == 0, "Input may only use the dimensions 'BC' (Batch, Channel), rest not implemented");
        const auto input_collapsed  = collapse<CollapsedOrder, "B">(Input); // Collapse the input to the specified order
        auto       output_collapsed = collapse<CollapsedOrder, "B">(Out);   // Collapse the output to the specified order
        functions::linear::Linear<UsedSuggestedSubBatchSize, MACOperator>(input_collapsed, output_collapsed, weights_, bias_, Act, std::get<I>(activation_parameters_)...);
    }

    template <bool             ContinueAfter             = true,
              std::size_t      UsedSuggestedSubBatchSize = suggested_sub_batch_size,
              IsMatrixType     InputMatrixType,
              IsMatrixType     OutputMatrixType,
              IsBaseMatrixType BufferMatrixType    = Matrix<char, "E", 0>,
              IsBaseMatrixType PermanentMatrixType = Matrix<char, "E", 0>>
    __attribute__((always_inline)) inline void operator()(const InputMatrixType &Input,
                                                          OutputMatrixType      &Out,
                                                    BufferMatrixType             &Buffer,
                                                          PermanentMatrixType          &PermanentBuffer) const noexcept {
                                                            this->operator()(Input, Out, Buffer, PermanentBuffer, std::make_index_sequence<sizeof...(ActivationMatrixInformation)>());
                                                          }

};

static_assert(IsValidLayer<LinearLayer<>>, "LinearLayer does not meet the requirements of a valid layer");

template <typename OutputType                                       = float,
          std::size_t    SuggestedSubBatchSize                      = 1,
          DimensionOrder CollapsedOrder                             = "B", // All Dimensions that will be collapsed to 'B'
          template <typename, typename, typename> class MACOperator = DefaultMACOperation,
          typename WeightMatrixType                                 = decltype(functions::linear::weightSubBio<1, 1>(Matrix<float, "OI", 1, 1>())),
          IsMatrixType BiasMatrixType                               = Matrix<OutputType, "BC", 1, 1>,
          typename Lambda                                           = decltype([](const OutputType a) { return a; }),
          IsMatrixType... ActivationMatrixInformation>
__attribute__((always_inline)) inline constexpr auto Linear( // Function Parameters
        WeightMatrixType &&Weights,
        BiasMatrixType   &&Bias,
        Lambda           &&Act = {},
        ActivationMatrixInformation &&...ActivationParameters) {
    return LinearLayer<OutputType, SuggestedSubBatchSize, CollapsedOrder, MACOperator, WeightMatrixType, BiasMatrixType, Lambda, ActivationMatrixInformation...>(
            std::forward<WeightMatrixType>(Weights), std::forward<BiasMatrixType>(Bias), std::forward<Lambda>(Act), std::forward<ActivationMatrixInformation>(ActivationParameters)...);
}
} // namespace layers
