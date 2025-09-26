#pragma once

#warning "Work in progress, do not use yet"
#warning "Order is not optimized"
#warning "High performance Linear op never tested"
#warning "TODO: add padding, move transfomation to matrix.hpp"

#include <array>
#include <concepts>
#include <cstddef>
#include <stddef.h>
#include <tuple>
#include <type_traits>
#include <utility>

#include "../Matrix.hpp"

#include "./BaseLayer.hpp"

#include "../functions/linear.hpp"

#include "../helpers/print.hpp"

template <IsMatrixType                                   BaseMatrixType,
          DimensionOrder                                 TraverseOrder,
          DimensionOrder                                 TraverseAddedOrder,
          std::array<Dim_size_t, TraverseOrder.length()> KernelSizes,
          std::array<Dim_size_t, TraverseOrder.length()> Strides,
          std::array<Dim_size_t, TraverseOrder.length()> Dilations,
          typename TraverseVariadicIndices,
          typename TraverseRestVariadicIndices,
          typename OriginalVariadicIndices,
          typename VariadicIndices>
    requires(TraverseOrder.unique() && TraverseOrder.length() > 0 &&
             std::remove_cvref_t<BaseMatrixType>::order.containsAll(TraverseOrder) && // TraverseOrder must be in the order of the dimensions that will be traversed
             TraverseAddedOrder.unique() && TraverseAddedOrder.length() == TraverseOrder.length() &&
             !std::remove_cvref_t<BaseMatrixType>::order.containsAny(TraverseAddedOrder) &&                      // TraverseAddedOrder expands the TraverseOrder
             !TraverseOrder.containsAny(TraverseAddedOrder) && !TraverseAddedOrder.containsAny(TraverseOrder) && // TraverseOrder and TraverseAddedOrder must not overlap
             true                                                                                                // Just to make the requires clause easier to read
             )
struct Image2ColMatrixType;

/*
Strcutured Slicing and stacking of matrix
Example: 2D Convolution
Order: BWHC
TraverseOrder: WH
TraverseAddedOrder: wh
KernelSizes: {A,B}
Strides: {Q,D}
Dilations: {E,F}
The resulting Matrix will have the order: BWwHhC
with the dimensions:
B: Original B
W: (Original W + 2*Padding - E*(A-1) -1)/Q + 1
H: (Original H + 2*Padding - F*(B-1) -1)/D + 1
w: A
h: B
C: Original C
*/

template <IsMatrixType                                   BaseMatrixType,
          DimensionOrder                                 TraverseOrder,
          DimensionOrder                                 TraverseAddedOrder,
          std::array<Dim_size_t, TraverseOrder.length()> KernelSizes,
          std::array<Dim_size_t, TraverseOrder.length()> Strides,
          std::array<Dim_size_t, TraverseOrder.length()> Dilations,
          std::size_t... TraverseVariadicIndices,
          std::size_t... TraverseRestVariadicIndices,
          std::size_t... OriginalVariadicIndices,
          std::size_t... VariadicIndices>
struct Image2ColMatrixType< // Focred linebreak
        BaseMatrixType,
        TraverseOrder,
        TraverseAddedOrder,
        KernelSizes,
        Strides,
        Dilations,
        std::index_sequence<TraverseVariadicIndices...>,
        std::index_sequence<TraverseRestVariadicIndices...>,
        std::index_sequence<OriginalVariadicIndices...>,
        std::index_sequence<VariadicIndices...>> {
    using BaseMatrixTypeNoRef = std::remove_cvref_t<BaseMatrixType>;
    using value_type          = typename BaseMatrixTypeNoRef::value_type;
    using storage_type = std::conditional_t<std::is_rvalue_reference_v<BaseMatrixType>, std::remove_reference_t<BaseMatrixType>, std::add_lvalue_reference_t<std::remove_reference_t<BaseMatrixType>>>;

    static constexpr std::size_t    number_of_dimensions = BaseMatrixTypeNoRef::number_of_dimensions + TraverseOrder.length();
    static constexpr DimensionOrder order                = BaseMatrixTypeNoRef::order.template multiInsert<TraverseOrder.length()>(
            {TraverseOrder[TraverseVariadicIndices]...}, {DimensionOrder(TraverseOrder[TraverseVariadicIndices]) + DimensionOrder(TraverseAddedOrder[TraverseVariadicIndices])...});

    static constexpr std::size_t                                           original_number_of_dimensions = BaseMatrixTypeNoRef::number_of_dimensions;
    static constexpr std::array<Dim_size_t, original_number_of_dimensions> original_dimensions           = BaseMatrixTypeNoRef::dimensions;
    static constexpr DimensionOrder                                        original_order                = BaseMatrixTypeNoRef::order;

    static constexpr std::array<Dim_size_t, TraverseOrder.length()> traverse_sizes = {
            (                                                                                            // variadic braces to expand the array
                    (original_dimensions[original_order.indexOf(TraverseOrder[TraverseVariadicIndices])] // Original Size
                     + 2 * 0                                                                             // Padding is always zero for now
                     - Dilations[TraverseVariadicIndices] * (KernelSizes[TraverseVariadicIndices] - 1)   // Effective Kernel Size
                     - 1                                                                                 // -1 because we start counting at zero
                     ) / Strides[TraverseVariadicIndices]                                                // Stride
                    + 1                                                                                  // +1 because we want to include the last step
                    )...                                                                                 // Expand the array
    };

    static constexpr DimensionOrder order_traverse_ez_at_back = order.remove(TraverseOrder + TraverseAddedOrder) + TraverseOrder + TraverseAddedOrder;
    static constexpr auto           permutation_order = (original_order.remove(TraverseOrder) + TraverseOrder).template permutationOrderComputation<original_number_of_dimensions>(original_order);
    static constexpr auto           permutation_order_inverse = order.template permutationOrderComputation<number_of_dimensions>(order_traverse_ez_at_back);

    static constexpr std::array<Dim_size_t, number_of_dimensions> tmp_dimensions = {original_dimensions[permutation_order[TraverseRestVariadicIndices]]..., traverse_sizes[TraverseVariadicIndices]...,
                                                                                    KernelSizes[TraverseVariadicIndices]...};

    static constexpr std::array<Dim_size_t, number_of_dimensions> dimensions           = {tmp_dimensions[permutation_order_inverse[VariadicIndices]]...};
    static constexpr bool                                         k_has_zero_dimension = ((dimensions[VariadicIndices] <= 0) || ...);

    storage_type data;

    Image2ColMatrixType(reference_or_rvalue<BaseMatrixType> ref) : data(ref) {
    }

    constexpr Image2ColMatrixType(const_reference_or_rvalue<BaseMatrixType> ref) : data(ref) {
    }

    template <IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions)
    __attribute__((always_inline)) inline std::array<Dim_size_t, original_number_of_dimensions> calculatePositions(const DimTypes... dim) const {
        constexpr auto                                       permutation_order = order_traverse_ez_at_back.template permutationOrderComputation<number_of_dimensions>(order);
        const std::array<Dim_size_t, number_of_dimensions>   dims              = {std::get<permutation_order[VariadicIndices]>(std::make_tuple(dim...))...};
        const std::array<Dim_size_t, TraverseOrder.length()> traverse_indices  = {dims[number_of_dimensions - 2 * TraverseOrder.length() + TraverseVariadicIndices]...};
        const std::array<Dim_size_t, TraverseOrder.length()> kernel_indices    = {dims[number_of_dimensions - TraverseOrder.length() + TraverseVariadicIndices]...};

        const std::array<Dim_size_t, original_number_of_dimensions> original_indices = {
                dims[TraverseRestVariadicIndices]..., // Non-traversed dimensions stay the same
                (traverse_indices[TraverseVariadicIndices] * Strides[TraverseVariadicIndices] + kernel_indices[TraverseVariadicIndices] * Dilations[TraverseVariadicIndices])...};

        constexpr auto permutation_to_original = original_order.template permutationOrderComputation<original_number_of_dimensions>(original_order.remove(TraverseOrder) + TraverseOrder);

        const std::array<Dim_size_t, original_number_of_dimensions> full_dims_original_order = {original_indices[permutation_to_original[OriginalVariadicIndices]]...};
        return full_dims_original_order;
    }

    template <IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions && !std::is_const_v<std::remove_reference_t<storage_type>>)
    __attribute__((always_inline)) inline value_type &at(const DimTypes... dim) {
        checkBounds<dimensions[VariadicIndices]...>({(Dim_size_t)dim...});
        const auto full_dims_original_order = calculatePositions(dim...);
        return data.template at<original_order>(std::get<OriginalVariadicIndices>(full_dims_original_order)...);
    }

    template <IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions)
    __attribute__((always_inline)) constexpr inline value_type at(const DimTypes... dim) const {
        checkBounds<dimensions[VariadicIndices]...>({(Dim_size_t)dim...});
        const auto full_dims_original_order = calculatePositions(dim...);
        return data.template at<original_order>(std::get<OriginalVariadicIndices>(full_dims_original_order)...);
    }

    template <DimensionOrder InterpretedDimensionOrder, IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions && InterpretedDimensionOrder.length() == order.length() && !std::is_const_v<std::remove_reference_t<storage_type>>)
    __attribute__((always_inline)) inline value_type &at(DimTypes... dim) {
        constexpr std::array<Dim_size_t, number_of_dimensions> permutation_order = order.template permutationOrderComputation<number_of_dimensions>(InterpretedDimensionOrder);
        return this->at(std::get<permutation_order[VariadicIndices]>(std::make_tuple(dim...))...);
    }

    template <DimensionOrder InterpretedDimensionOrder, IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions && InterpretedDimensionOrder.length() == order.length())
    __attribute__((always_inline)) constexpr inline value_type at(DimTypes... dim) const {
        constexpr std::array<Dim_size_t, number_of_dimensions> permutation_order = order.template permutationOrderComputation<number_of_dimensions>(InterpretedDimensionOrder);
        return this->at(std::get<permutation_order[VariadicIndices]>(std::make_tuple(dim...))...);
    }
};

template <IsMatrixType                                   BaseMatrixType,
          DimensionOrder                                 TraverseOrder,
          DimensionOrder                                 TraverseAddedOrder,
          std::array<Dim_size_t, TraverseOrder.length()> KernelSizes,
          std::array<Dim_size_t, TraverseOrder.length()> Strides,
          std::array<Dim_size_t, TraverseOrder.length()> Dilations>
using Image2ColMatrix = Image2ColMatrixType<BaseMatrixType,
                                            TraverseOrder,
                                            TraverseAddedOrder,
                                            KernelSizes,
                                            Strides,
                                            Dilations,
                                            std::make_index_sequence<TraverseOrder.length()>,
                                            std::make_index_sequence<std::remove_cvref_t<BaseMatrixType>::number_of_dimensions - TraverseOrder.length()>,
                                            std::make_index_sequence<std::remove_cvref_t<BaseMatrixType>::number_of_dimensions>,
                                            std::make_index_sequence<std::remove_cvref_t<BaseMatrixType>::number_of_dimensions + TraverseOrder.length()>>;

static_assert(IsMatrixType<Image2ColMatrix<Matrix<float, "12", 2, 3>, "12", "34", {1, 1}, {1, 1}, {1, 1}>>, "Example Assert, for sanity, __TestMatrix is not a valid MatrixType");

template <DimensionOrder                                 TraverseOrder,
          DimensionOrder                                 TraverseAddedOrder,
          std::array<Dim_size_t, TraverseOrder.length()> KernelSizes    = makeFilledArray<Dim_size_t, TraverseOrder.length()>(1),
          std::array<Dim_size_t, TraverseOrder.length()> Strides        = makeFilledArray<Dim_size_t, TraverseOrder.length()>(1),
          std::array<Dim_size_t, TraverseOrder.length()> Dilations      = makeFilledArray<Dim_size_t, TraverseOrder.length()>(1),
          IsMatrixType                                   BaseMatrixType = Matrix<char, "E", 0>>
    requires(TraverseOrder.length() == TraverseAddedOrder.length() && TraverseOrder.length() == KernelSizes.size() && TraverseOrder.length() == Strides.size() &&
             TraverseOrder.length() == Dilations.size())
constexpr conditional_const<std::is_const_v<std::remove_reference_t<BaseMatrixType>>, Image2ColMatrix<BaseMatrixType &&, TraverseOrder, TraverseAddedOrder, KernelSizes, Strides, Dilations>> im2col(
        BaseMatrixType &&base_matrix) {
    return Image2ColMatrix<BaseMatrixType &&, TraverseOrder, TraverseAddedOrder, KernelSizes, Strides, Dilations>(std::forward<BaseMatrixType>(base_matrix));
}

namespace layers {

template < // Focred linebreak
        typename OutputType                                                    = float,
        DimensionOrder                                      TraverseDimensions = "S", // or WH for 2D convolutions
        std::array<Dim_size_t, TraverseDimensions.length()> KernelSizes        = makeFilledArray<Dim_size_t, TraverseDimensions.length()>(1),
        std::array<Dim_size_t, TraverseDimensions.length()> Strides            = makeFilledArray<Dim_size_t, TraverseDimensions.length()>(1),
        std::array<Dim_size_t, TraverseDimensions.length()> Dilations          = makeFilledArray<Dim_size_t, TraverseDimensions.length()>(1),
        DimensionOrder InternalTraverseAddedDimensions            = TraverseDimensions.toLowerCase(), // or wh for 2D convolutions, override manually if using lowercase letters in TraverseDimensions
        std::size_t    SuggestedSubBatchSize                      = 1,                                // For now should be increased later as batch processing is default for convolutions
        DimensionOrder CollapsedOrder                             = DimensionOrder("B") + InternalTraverseAddedDimensions, // All Dimensions that will be collapsed to 'B'
        template <typename, typename, typename> class MACOperator = DefaultMACOperation,
        //   typename WeightMatrixType                                 = decltype(functions::linear::weightSubBio<1, 1>(std::declval<Matrix<float, "OI", 1, 1>>())),
        typename WeightMatrixType   = Matrix<float, TraverseDimensions + "OI", 1, 1, 1>,
        IsMatrixType BiasMatrixType = Matrix<OutputType, "BC", 1, 1>,
        typename Lambda             = decltype([](const OutputType a) { return a; }),
        IsMatrixType... ActivationMatrixInformation>
class ConvolutionLayer {
  private:
    using WeightMatrixType_ = std::remove_cvref_t<WeightMatrixType>;
    using BiasMatrixType_   = std::remove_cvref_t<BiasMatrixType>;

    // static_assert(IsAlignedMatrixCollection<WeightMatrixType> || IsMatrixType<WeightMatrixType>, "WeightMatrixType must be an AlignedMatrixCollection");
    // using WeightMatrixTypeCollapsed = typename functions::linear::InverseWeightSubBioMatrixType<WeightMatrixType_>; // TODO
    using WeightMatrixTypeCollapsed = WeightMatrixType_;
    static_assert(WeightMatrixTypeCollapsed::order.remove(TraverseDimensions + "IO").length() == 0, "Weights may only have IO in a colapsed way");
    static_assert(BiasMatrixType_::order.remove("BCE").length() == 0, "BiasMatrixType must be in the order of 'BC' (Batch, Channel) or 'C' (Channel) only or 'E' (Element) only");

    using WeightMatrixTypeCollapsedOrdered = MaterializedMatrix<PermutedMatrix<DimensionOrder("OI") + TraverseDimensions, WeightMatrixTypeCollapsed>>;

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

    /* Helpers */
    template <IsMatrixType Matrix>
    using Im2ColTransformedType = Image2ColMatrix<Matrix, TraverseDimensions, InternalTraverseAddedDimensions, KernelSizes, Strides, Dilations>;

    template <IsMatrixType Matrix>
    using PaddedMatrixType = Matrix; // TODO

    template <IsMatrixType InputMatrix>
    using OutputMatrix = MaterializedMatrix<                                      // Materialize to have a concrete type
            OverrideDimensionMatrix<                                              // Override the channel dimension 'C' to the output channels
                    OverrideRemoveDimensionMatrix<                                // Remove the kernel dimensions
                            Im2ColTransformedType<PaddedMatrixType<InputMatrix>>, // Transform the input matrix to im2col, with padding
                            InternalTraverseAddedDimensions>,
                    "C",
                    output_channels>>;

    // Memory Requirements of forward pass
    template <IsMatrixType InputMatrix>
    static constexpr std::size_t memory_minimal = sizeof(MaterializedMatrix<InputMatrix>) + sizeof(MaterializedMatrix<OutputMatrix<InputMatrix>>);
    // Can it reuse the input Memory region?
    static constexpr bool memory_inlined = false;
    // Required Buffer size
    // the amount of Memory required for temporary storage
    template <IsMatrixType InputMatrix>
    static constexpr std::size_t memory_buffer = 0;
    // Permanent Memory which is required for some layers, if used in a time series model
    template <IsMatrixType InputMatrix>
    static constexpr std::size_t memory_permanent = 0;

    using ExampleInputMatrix = Matrix<char, "BSC", 1, 1, 1>;

    // Constructor
    constexpr ConvolutionLayer(WeightMatrixType &&Weights, BiasMatrixType &&Bias, Lambda &&Act = {}, ActivationMatrixInformation &&...ActivationParameters)
            : bias_(std::forward<BiasMatrixType>(Bias)), weights_(std::forward<WeightMatrixType>(Weights)), Act(std::forward<Lambda>(Act)),
              activation_parameters_(std::forward<ActivationMatrixInformation>(ActivationParameters)...) {
    }

    // To catch all the cases where the operator is not implemented, at the same time define the parameters
    template <IsMatrixType InputMatrixType, IsMatrixType OutputMatrixType, IsMatrixType BufferMatrixType, IsMatrixType PermanentMatrixType, std::size_t... I>
    __attribute__((always_inline)) inline void operator()(const InputMatrixType &Input,
                                                          OutputMatrixType      &Out,
                                                          BufferMatrixType      &buffer,
                                                          PermanentMatrixType   &permanent,
                                                          const std::index_sequence<I...> = std::make_index_sequence<sizeof...(ActivationMatrixInformation)>()) const noexcept {
        // constexpr Dim_size_t input_batch    = (InputMatrixType::order.contains('B') ? InputMatrixType::dimensions[InputMatrixType::order.indexOf('B')] : 1);
        // constexpr Dim_size_t input_sequence = (InputMatrixType::order.contains('S') ? InputMatrixType::dimensions[InputMatrixType::order.indexOf('S')] : 1);

        const auto Input_expanded  = conditionalBroadcast<"B">(Input);
        auto       Output_expanded = conditionalBroadcast<"B">(Out);

        auto im2_col_transformed = im2col<TraverseDimensions, InternalTraverseAddedDimensions, KernelSizes, Strides, Dilations>(Input_expanded);
        // std::cout << "Im2Col Transformed: " << im2_col_transformed << std::endl;
        auto im2_col_transformed_collapsed = collapse<DimensionOrder("B") + TraverseDimensions, "B">(collapse<InternalTraverseAddedDimensions + "C", "C">(im2_col_transformed));
        // std::cout << "Im2Col Transformed Collapsed: " << im2_col_transformed_collapsed << std::endl;

        // std::cout << "weights: " << weights_ << std::endl;
        auto weights_matrix_transformed = collapse<TraverseDimensions + "I", "I">(weights_);
        // std::cout << "weights transformed: " << weights_matrix_transformed << std::endl;

        // std::cout << "output: " << Out << std::endl;
        auto out2_col_transformed = collapse<DimensionOrder("B") + TraverseDimensions, "B">(Output_expanded);
        // std::cout << "output collapsed: " << out2_col_transformed << std::endl;
        functions::linear::Linear(im2_col_transformed_collapsed, out2_col_transformed, weights_matrix_transformed, bias_, Act, std::get<I>(activation_parameters_)...);
    }
};

static_assert(IsValidLayer<ConvolutionLayer<>>, "BaseLayer does not meet the requirements of a valid layer");

template <typename OutputType                                                    = float,
          DimensionOrder                                      TraverseDimensions = "S", // or WH for 2D convolutions
          std::array<Dim_size_t, TraverseDimensions.length()> KernelSizes        = makeFilledArray<Dim_size_t, TraverseDimensions.length()>(1),
          std::array<Dim_size_t, TraverseDimensions.length()> Strides            = makeFilledArray<Dim_size_t, TraverseDimensions.length()>(1),
          std::array<Dim_size_t, TraverseDimensions.length()> Dilations          = makeFilledArray<Dim_size_t, TraverseDimensions.length()>(1),
          DimensionOrder InternalTraverseAddedDimensions            = TraverseDimensions.toLowerCase(), // or wh for 2D convolutions, override manually if using lowercase letters in TraverseDimensions
          std::size_t    SuggestedSubBatchSize                      = 1,                                // For now should be increased later as batch processing is default for convolutions
          DimensionOrder CollapsedOrder                             = DimensionOrder("B") + InternalTraverseAddedDimensions, // All Dimensions that will be collapsed to 'B'
          template <typename, typename, typename> class MACOperator = DefaultMACOperation,
          //   typename WeightMatrixType                                 = decltype(functions::linear::weightSubBio<1, 1>(std::declval<Matrix<float, "OI", 1, 1>>())),
          typename WeightMatrixType   = Matrix<float, TraverseDimensions + "OI", 1, 1>,
          IsMatrixType BiasMatrixType = Matrix<OutputType, "BC", 1, 1>,
          typename Lambda             = decltype([](const OutputType a) { return a; }),
          IsMatrixType... ActivationMatrixInformation>
__attribute__((always_inline)) inline constexpr auto Convolution( // Function Parameters
        WeightMatrixType &&Weights,
        BiasMatrixType   &&Bias,
        Lambda           &&Act = {},
        ActivationMatrixInformation &&...ActivationParameters) {
    return ConvolutionLayer<OutputType, TraverseDimensions, KernelSizes, Strides, Dilations, InternalTraverseAddedDimensions, SuggestedSubBatchSize, CollapsedOrder, MACOperator, WeightMatrixType,
                            BiasMatrixType, Lambda, ActivationMatrixInformation...>(std::forward<WeightMatrixType>(Weights), std::forward<BiasMatrixType>(Bias), std::forward<Lambda>(Act),
                                                                                    std::forward<ActivationMatrixInformation>(ActivationParameters)...);
}

} // namespace layers