#pragma once

#include <semaphore>
#include <tuple>
#include <type_traits>
#include <utility>

#include "../Matrix.hpp"
#include "../functions/linear.hpp"
#include "../types/Complex.hpp"

#include "BaseLayer.hpp"

#include "../helpers/print.hpp"

namespace layers {

template <typename OutputType                                            = float,
          typename StateType                                             = Complex<float>,
          std::size_t SuggestedSubBatchSizeComplex                       = 1,
          std::size_t SuggestedSubBatchSizeReal                          = 1,
          bool        IgnoreSkipConnectionValue                          = false,
          template <typename, typename, typename> class BMACOperator_    = DefaultMACOperation,
          template <typename, typename, typename> class CMACOperator_    = RealResultMACOperation,
          template <typename, typename, typename> class DMACOperator_    = DefaultMACOperation,
          template <typename, typename, typename> class SkipMACOperator_ = DefaultMACOperation,
          IsMatrixType AMatrixType                                       = Matrix<Complex<float>, "C", 1>,
          typename BMatrixType                                           = Matrix<Complex<float>, "IO", 1, 1>,
          IsMatrixType BBiasMatrixType                                   = Matrix<Complex<float>, "C", 1>,
          typename CMatrixType                                           = Matrix<Complex<float>, "IO", 1, 1>,
          IsMatrixType CBiasMatrixType                                   = Matrix<float, "C", 1>,
          IsMatrixType DMatrixType                                       = Matrix<float, "IO", 0, 0>, // if the matrix is empty, it will not be used
          typename SkipMatrixType                                        = Matrix<float, "IO", 1, 1>, // Trainable skip connection, either a matrix "IO" a vector "C" or a scalar "E"
          typename Lambda                                                = decltype([]() {}),
          IsMatrixType... ActivationMatrixInformation>
class DSSMLayer {
  public:
    // template <typename InputType, typename WeightType, typename BiasType>
    // using BMACOperator_ = DefaultMACOperation<InputType, WeightType, BiasType>;
    // template <typename InputType, typename WeightType, typename BiasType>
    // using CMACOperator_ = RealResultMACOperation<InputType, WeightType, BiasType>;
    // template<typename InputType, typename WeightType, typename BiasType>
    // using DMACOperator_ = DefaultMACOperation<InputType, WeightType, BiasType>;
    // template<typename InputType, typename WeightType, typename BiasType>
    // using SkipMACOperator_ = DefaultMACOperation<InputType, WeightType, BiasType>;

    using AMatrixType_     = std::remove_cvref_t<AMatrixType>;
    using BMatrixType_     = std::remove_cvref_t<BMatrixType>;
    using BBiasMatrixType_ = std::remove_cvref_t<BBiasMatrixType>;
    using CMatrixType_     = std::remove_cvref_t<CMatrixType>;
    using CBiasMatrixType_ = std::remove_cvref_t<CBiasMatrixType>;
    using DMatrixType_     = std::remove_cvref_t<DMatrixType>;
    using SkipMatrixType_  = std::remove_cvref_t<SkipMatrixType>;

    const AMatrixType_                                                          amatrix_;
    const BMatrixType_                                                          bmatrix_;
    const BBiasMatrixType_                                                      bbias_;
    const CMatrixType_                                                          cmatrix_;
    const CBiasMatrixType_                                                      cbias_;
    const DMatrixType_                                                          dmatrix_;
    const SkipMatrixType_                                                       skip_matrix_;
    const std::tuple<const std::remove_cvref_t<ActivationMatrixInformation>...> activation_parameters_;
    const Lambda                                                                act_;

    using BMatrixCollapsed    = typename functions::linear::InverseWeightSubBioMatrixType<BMatrixType_>;
    using CMatrixCollapsed    = typename functions::linear::InverseWeightSubBioMatrixType<CMatrixType_>;
    using DMatrixCollapsed    = typename functions::linear::InverseWeightSubBioMatrixType<DMatrixType_>;
    using SkipMatrixCollapsed = typename functions::linear::InverseWeightSubBioMatrixType<SkipMatrixType_>;

    static_assert(BMatrixCollapsed::order.remove("IOE").length() == 0, "Collapsed B Matrix may only contain the orders IOE");
    static_assert(CMatrixCollapsed::order.remove("IO").length() == 0, "Collapsed C Matrix may only contain the orders IO");
    static_assert(DMatrixCollapsed::order.remove("IO").length() == 0, "Collapsed D Matrix may only contain the orders IO");
    static_assert(SkipMatrixCollapsed::order.remove("IOCE").length() == 0, "Collapsed Skip Matrix may only contain the orders IO, C or E");
    static_assert(((int)SkipMatrixCollapsed::order.containsAny("IO") + (int)SkipMatrixCollapsed::order.containsAny("C") + (int)SkipMatrixCollapsed::order.containsAny("E")) == 1,
                  "Collapsed Skip Matrix may only contain the orders IO, C or E");
    static_assert(BBiasMatrixType_::order.remove("EC").length() == 0, "BBiasMatrixType must be in the order of 'C' (Channel) or 'E' (Element/Empty)");
    static_assert(CBiasMatrixType_::order.remove("EC").length() == 0, "CBiasMatrixType must be in the order of 'C' (Channel) or 'E' (Element)");
    static_assert(AMatrixType_::order.remove("C").length() == 0, "AMatrixType must be in the order of 'C' (Channel) only");

    constexpr static Dim_size_t input_channels        = (BMatrixCollapsed::order.contains('I') ? BMatrixCollapsed::dimensions[BMatrixCollapsed::order.indexOf('I')] : 1);
    constexpr static Dim_size_t hidden_channels       = AMatrixType_::dimensions[AMatrixType_::order.indexOf('C')];
    constexpr static Dim_size_t hidden_channels_cmp   = (BMatrixCollapsed::order.contains('O') ? BMatrixCollapsed::dimensions[BMatrixCollapsed::order.indexOf('O')] : 1);
    constexpr static Dim_size_t hidden_channels_cmp_2 = CMatrixCollapsed::dimensions[CMatrixCollapsed::order.indexOf('I')];
    constexpr static Dim_size_t output_channels       = CMatrixCollapsed::dimensions[CMatrixCollapsed::order.indexOf('O')];

    static_assert(BMatrixCollapsed::order.containsOnly("E") || hidden_channels == hidden_channels_cmp, "Input channels must match hidden channels in AMatrixType");
    static_assert(hidden_channels == hidden_channels_cmp_2, "Hidden channels must match hidden channels in CMatrixType");

    template <IsMatrixType InputMatrix>
    using StateMatrixType = Matrix<StateType, "BC", (InputMatrix::order.contains('B') ? InputMatrix::dimensions[InputMatrix::order.indexOf('B')] : 1), hidden_channels>;

    // Can it reuse the input Memory region?
    static constexpr bool memory_inlined = false;
    // Required Buffer size
    // the amount of Memory required for temporary storage
    template <IsMatrixType InputMatrix>
    static constexpr std::size_t memory_buffer =
            (InputMatrix::order.contains('S') && (InputMatrix::dimensions[InputMatrix::order.indexOf('S')] > 1) ? InputMatrix::dimensions[InputMatrix::order.indexOf('S')] : 0) *
            sizeof(StateMatrixType<InputMatrix>);
    // Permanent Memory which is required for some layers, if used in a time series model
    template <IsMatrixType InputMatrix>
    static constexpr std::size_t memory_permanent = sizeof(StateMatrixType<InputMatrix>);

    using ExampleInputMatrix = Matrix<float, "C", 1>;
    template <typename InputMatrix>
    using OutputMatrix = OverrideDimensionMatrix<InputMatrix, "C", output_channels>;

    // Memory Requirements of forward pass
    template <IsMatrixType InputMatrix>
    static constexpr std::size_t memory_minimal = sizeof(MaterializedMatrix<InputMatrix>) + sizeof(MaterializedMatrix<OutputMatrix<InputMatrix>>) + memory_buffer<InputMatrix>;

    // Constructor
    constexpr DSSMLayer(AMatrixType     &&AMatrix,
                        BMatrixType     &&BMatrix,
                        BBiasMatrixType &&BBias,
                        CMatrixType     &&CMatrix,
                        CBiasMatrixType &&CBias,
                        DMatrixType     &&DMatrix,
                        SkipMatrixType  &&SkipMatrix,
                        Lambda          &&Act,
                        ActivationMatrixInformation &...ActivationParameters) noexcept
            : amatrix_(std::forward<AMatrixType>(AMatrix)), bmatrix_(std::forward<BMatrixType>(BMatrix)), bbias_(std::forward<BBiasMatrixType>(BBias)), cmatrix_(std::forward<CMatrixType>(CMatrix)),
              cbias_(std::forward<CBiasMatrixType>(CBias)), dmatrix_(std::forward<DMatrixType>(DMatrix)), skip_matrix_(std::forward<SkipMatrixType>(SkipMatrix)),
              activation_parameters_(std::forward<ActivationMatrixInformation>(ActivationParameters)...), act_(std::forward<Lambda>(Act)) {
    }

    template <bool         ContinueAfter       = true,
              IsMatrixType InputMatrixType     = Matrix<float, "BCS", 1, input_channels>,
              IsMatrixType OutputMatrixType    = OverrideDimensionMatrix<InputMatrixType, "C", output_channels>,
              IsMatrixType BufferMatrixType    = Matrix<char, "E", 0>,
              IsMatrixType PermanentMatrixType = Matrix<char, "E", 0>,
              std::size_t... I>
    __attribute__((always_inline))
    // __attribute__((noinline))
    inline void
    operator()(const InputMatrixType &Input,
               OutputMatrixType      &Out,
               BufferMatrixType      &buffer,
               PermanentMatrixType   &permanent,
               const std::index_sequence<I...> = std::make_index_sequence<sizeof...(ActivationMatrixInformation)>()) const noexcept {
        static_assert(InputMatrixType::order.remove("BCS").length() == 0, "Input may only use the dimensions 'BCS' (Batch, Channel, Sequence), rest not implemented");
        static_assert(OutputMatrixType::order.remove("BCS").length() == 0, "Output may only use the dimensions 'BCS' (Batch, Channel, Sequence), rest not implemented");
        static_assert(BufferMatrixType::order.remove("E").length() == 0, "Buffer may only use the dimensions 'E' (Error)");
        static_assert(PermanentMatrixType::order.remove("E").length() == 0, "Permanent may only use the dimensions 'E' (Error)");
        static_assert(PermanentMatrixType::dimensions[0] == memory_permanent<InputMatrixType>, "Permanent Memory Size does not match the required size, must be memory_permanent<InputMatrixType>");
        static_assert(BufferMatrixType::dimensions[0] == memory_buffer<InputMatrixType>, "Buffer Memory Size does not match the required size, must be memory_buffer<InputMatrixType>");

        using StateMatrixType = StateMatrixType<InputMatrixType>;
        static_assert(StateMatrixType::order.containsOnly("BC"), "State Matrix must use 'BC' (Batch, Channel)");

        // flags
        constexpr bool is_single_sequence                  = !InputMatrixType::order.contains('S') || (InputMatrixType::dimensions[InputMatrixType::order.indexOf('S')] == 1);
        constexpr bool full_skip_connection_enabled        = SkipMatrixCollapsed::order.containsAny("IO");
        constexpr bool elementwise_skip_connection_enabled = SkipMatrixCollapsed::order.containsAny("C") || SkipMatrixCollapsed::order.containsAny("E");
        constexpr bool skip_connection_used                = !SkipMatrixCollapsed::k_has_zero_dimension; // first dimension can never be 0
        constexpr bool d_matrix_used                       = !DMatrixCollapsed::k_has_zero_dimension;    // first dimension can never be 0
        // static_assert(!elementwise_skip_connection_enabled, "Not implemented yet");
        static_assert(!d_matrix_used, "Not implemented yet");

        constexpr Dim_size_t batch_size          = InputMatrixType::order.contains('B') ? InputMatrixType::dimensions[InputMatrixType::order.indexOf('B')] : 1;
        constexpr Dim_size_t batch_size_cmp      = OutputMatrixType::order.contains('B') ? OutputMatrixType::dimensions[OutputMatrixType::order.indexOf('B')] : 1;
        constexpr Dim_size_t sequence_length     = InputMatrixType::order.contains('S') ? InputMatrixType::dimensions[InputMatrixType::order.indexOf('S')] : 1;
        constexpr Dim_size_t sequence_length_cmp = OutputMatrixType::order.contains('S') ? OutputMatrixType::dimensions[OutputMatrixType::order.indexOf('S')] : 1;

        static_assert(batch_size == batch_size_cmp, "Batch size of Input and Output must match");
        static_assert(sequence_length == sequence_length_cmp, "Sequence length of Input and Output must match");

        const auto input_expanded   = permute<"BSC">(conditionalBroadcast<"B", {batch_size}>(conditionalBroadcast<"S", {sequence_length}>(Input)));
        auto       output_expanded  = permute<"BSC">(conditionalBroadcast<"B", {batch_size}>(conditionalBroadcast<"S", {sequence_length}>(Out)));
        const auto input_collapsed  = collapse<"BS", "B">(input_expanded);
        auto       output_collapsed = collapse<"BS", "B">(output_expanded);

        // interpret the permanent memory as the state matrix
        auto &state = *reinterpret_cast<StateMatrixType *>(&permanent.data[0]);

        // switch between multi Sequence and single sequence
        if constexpr (!is_single_sequence) {
            // Multi Sequence
            //    cant use fused activatin function, as the order of the sequence might be inconsistent
            //    we need to loop manually over the sequence dimension
            //    Buffer BU required and x over all t ( we just reuse BU as x[t] )
            //       BU = B@u + BBias
            //       x[t] = A*x[t-1] + BU[t]
            //       y = Act(R(C@x + CBias (+ D@u) ))
            //       out = y + SkipConnection@u
            using BUType = MaterializedMatrix<PermutedMatrix<"BSC", BroadcastedMatrix<StateMatrixType, "S", {sequence_length}>>>;
            static_assert(sizeof(buffer.data) >= sizeof(BUType), "Buffer Memory Size does not match the required size for BUType");
            auto &bu = *reinterpret_cast<BUType *>(&buffer.data[0]);

            // // Required Data views (just for interpretation of the data, no copying)
            auto       bu_collapsed   = collapse<"BS", "B">(bu);
            const auto a_broadcasted  = permute<"BSC">(conditionalBroadcast<"B", {batch_size}>(conditionalBroadcast<"S", {1}>(amatrix_))); // A is broadcasted over the sequence dimension
            auto       bu_permuted    = permute<"BSC">(bu);                                                                                // BU must be in "BSC" order
            auto       state_permuted = permute<"BSC">(broadcast<"S", {1}>(state));                                                        // state must be in "BSC" order
            // auto       xt_m1          = concatenate<1>(state_permuted, slice<"S", sequence_length - 1>(bu_permuted, {0})); // x[t-1] // needs a number

            // BU = B@u + BBias
            functions::linear::Linear<SuggestedSubBatchSizeComplex, BMACOperator_>(input_collapsed, bu_collapsed, bmatrix_, bbias_, [](const auto &x) { return x; });

            // x[t] = A*x[t-1] + BU[t]
            loopUnrolled([](auto &x_t, const auto a, const auto x_t_m1) { x_t += a * x_t_m1; }, slice<"S", 1>(bu_permuted, {0}), a_broadcasted, state_permuted); // copy last state to state
            for (Dim_size_t t = 1; t < sequence_length; t++) {
                auto xt_m1_slice = slice<"S", 1>(bu_permuted, {t - 1}); // dimensions of xt_slice are "BSC"
                auto xt_slice    = slice<"S", 1>(bu_permuted, {t});     // dimensions of xt_slice are "BSC"
                loopUnrolled([](auto &x_t, const auto a, const auto x_t_m1) { x_t += a * x_t_m1; }, xt_slice, a_broadcasted, xt_m1_slice);
            }
            // copy last state to state
            loopUnrolled([](auto &ret, const auto x) { ret = x; }, state_permuted, slice<"S", 1>(bu_permuted, {sequence_length - 1})); // copy last state to state

            if constexpr (!ContinueAfter) {
                // If we do not continue after this layer, we can just return here
                return;
            }
            // y = Act(R(C@x + CBias)) (+ D@x)
            functions::linear::Linear<SuggestedSubBatchSizeComplex, CMACOperator_>(
                    bu_collapsed, output_collapsed, cmatrix_, cbias_, [&](const auto &x, const auto... vals) { return act_(x, vals...); }, std::get<I>(activation_parameters_)...);
        } else {
            // Single Sequence
            const auto a_broadcasted = permute<"BC">(broadcast<"B", {batch_size}>(amatrix_)); // A is broadcasted over the sequence dimension
            if constexpr (input_channels == 1 && BMatrixCollapsed::order.containsOnly("E") && BBiasMatrixType_::order.containsOnly("E")) {
                // Special case optimization, when input channels is 1 and B matrix is E for empty and BBias is empty
                // x[t] = A*x[t-1] + B*U + BBias -> x[t] = A*x[t-1] + u
                const auto input_expanded = replicate<"C", {hidden_channels}>(input_collapsed); // replicate input channels to match hidden channels
                // Datatypes should be Complex, Complex, Real,
                loop([](auto &x_t, const auto a, const auto u) { x_t += a * x_t + u; }, state, a_broadcasted, input_expanded);
            } else 
            {
                // May use fused activation function, as the order of the sequence is consistent as it is a single sequence
                // x[t] = A*x[t-1] + BU + BBias
                functions::linear::Linear<SuggestedSubBatchSizeComplex, BMACOperator_>(
                        input_collapsed, state, bmatrix_, bbias_, [](const auto bu, const auto x, const auto a) { return a * x + bu; }, state, a_broadcasted);
            }
            if constexpr (!ContinueAfter) {
                // If we do not continue after this layer, we can just return here
                return;
            }
            // y = Act(R(C@x + CBias))
            functions::linear::Linear<SuggestedSubBatchSizeComplex, CMACOperator_>(
                    state, output_collapsed, cmatrix_, cbias_, [&](const auto &x, const auto... vals) { return act_(x, vals...); }, std::get<I>(activation_parameters_)...);
        }
        // Handle the skip connection
        if constexpr (skip_connection_used && full_skip_connection_enabled) {
            // out = SkipConnection@u + y
            functions::linear::Linear<SuggestedSubBatchSizeReal, SkipMACOperator_>(input_collapsed, output_collapsed, skip_matrix_, output_collapsed, [](const auto &x) { return x; });

        } else if constexpr (skip_connection_used && elementwise_skip_connection_enabled) {
            static_assert(input_channels == 1 || input_channels == output_channels, "Elementwise skip connection requires input channels to be 1 or equal to output channels, This is cursed");
            static_assert(std::is_same_v<SkipMACOperator_<float, float, float>, DefaultMACOperation<float, float, float>>,
                          "Elementwise skip connection requires DefaultMACOperation, if you've specialized it, please implement elementwise multiplication");
            // out = y + SkipConnection*u
            const auto skip_broadcast = conditionalBroadcast<"B", {batch_size * sequence_length}>(conditionalReplace<"E", "C">(conditionalReplicate<"E", {output_channels}>(skip_matrix_)));
            if constexpr (input_channels == 1) {
                // out = y + SkipConnection[0]*u
                const auto input_collapsed_replicated = conditionalReplicate<"C", {output_channels}>(input_collapsed); // replicate input channels to match output channels
                if constexpr (IgnoreSkipConnectionValue) {
                    loopUnrolled([](auto &out, const auto u) { out += u; }, output_collapsed, input_collapsed_replicated);
                } else {
                    loopUnrolled([](auto &out, const auto skip, const auto u) { out += skip * u; }, output_collapsed, skip_broadcast, input_collapsed_replicated);
                }
            } else if constexpr (input_channels == output_channels) {
                // out = y + SkipConnection*u
                if constexpr (IgnoreSkipConnectionValue) {
                    loopUnrolled([](auto &out, const auto u) { out += u; }, output_collapsed, input_collapsed);
                } else {
                    loopUnrolled([](auto &out, const auto skip, const auto u) { out += skip * u; }, output_collapsed, skip_broadcast, input_collapsed);
                }
            }
        } else if constexpr (!skip_connection_used) {
            // out = y
            // nothing to do, as Output_collapsed already contains the result
        } else {
            static_assert(skip_connection_used, "Skip connection used but unknown how to handle it");
        }
    }
};

static_assert(IsValidLayer<DSSMLayer<>>, "SSM must be a valid layer type");

template <typename OutputType                                            = float,
          typename StateType                                             = Complex<float>,
          std::size_t SuggestedSubBatchSizeComplex                       = 1,
          std::size_t SuggestedSubBatchSizeReal                          = 1,
          template <typename, typename, typename> class BMACOperator_    = DefaultMACOperation,
          template <typename, typename, typename> class CMACOperator_    = RealResultMACOperation,
          template <typename, typename, typename> class DMACOperator_    = DefaultMACOperation,
          template <typename, typename, typename> class SkipMACOperator_ = DefaultMACOperation,
          IsMatrixType AMatrixType                                       = Matrix<Complex<float>, "C", 1>,
          typename BMatrixType                                           = Matrix<Complex<float>, "IO", 1, 1>,
          IsMatrixType BBiasMatrixType                                   = Matrix<Complex<float>, "C", 1>,
          typename CMatrixType                                           = Matrix<Complex<float>, "IO", 1, 1>,
          IsMatrixType CBiasMatrixType                                   = Matrix<float, "C", 1>,
          IsMatrixType DMatrixType                                       = Matrix<float, "IO", 0, 0>, // if the matrix is empty, it will not be used
          typename SkipMatrixType                                        = Matrix<float, "IO", 1, 1>, // Trainable skip connection, either a matrix "IO" a vector "C" or a scalar "E"
          typename Lambda                                                = decltype([]() {}),
          IsMatrixType... ActivationMatrixInformation>
__attribute__((always_inline)) inline constexpr auto DSSM(AMatrixType     &&AMatrix,
                                                          BMatrixType     &&BMatrix,
                                                          BBiasMatrixType &&BBias,
                                                          CMatrixType     &&CMatrix,
                                                          CBiasMatrixType &&CBias,
                                                          DMatrixType     &&DMatrix,
                                                          SkipMatrixType  &&SkipMatrix,
                                                          Lambda          &&Act,
                                                          ActivationMatrixInformation &...ActivationParameters) noexcept {
    return DSSMLayer<OutputType, StateType, SuggestedSubBatchSizeComplex, SuggestedSubBatchSizeReal, false, BMACOperator_, CMACOperator_, DMACOperator_, SkipMACOperator_, AMatrixType, BMatrixType,
                     BBiasMatrixType, CMatrixType, CBiasMatrixType, DMatrixType, SkipMatrixType, Lambda, ActivationMatrixInformation...>(
            std::forward<AMatrixType>(AMatrix), std::forward<BMatrixType>(BMatrix), std::forward<BBiasMatrixType>(BBias), std::forward<CMatrixType>(CMatrix), std::forward<CBiasMatrixType>(CBias),
            std::forward<DMatrixType>(DMatrix), std::forward<SkipMatrixType>(SkipMatrix), std::forward<Lambda>(Act), std::forward<ActivationMatrixInformation>(ActivationParameters)...);
}

template <typename OutputType                                            = float,
          typename StateType                                             = Complex<float>,
          std::size_t SuggestedSubBatchSizeComplex                       = 1,
          std::size_t SuggestedSubBatchSizeReal                          = 1,
          template <typename, typename, typename> class BMACOperator_    = DefaultMACOperation,
          template <typename, typename, typename> class CMACOperator_    = RealResultMACOperation,
          template <typename, typename, typename> class DMACOperator_    = NonMACOperation,
          template <typename, typename, typename> class SkipMACOperator_ = DefaultMACOperation,
          IsMatrixType AMatrixType                                       = Matrix<Complex<float>, "C", 1>,
          typename BMatrixType                                           = Matrix<Complex<float>, "IO", 1, 1>,
          IsMatrixType BBiasMatrixType                                   = Matrix<Complex<float>, "C", 1>,
          typename CMatrixType                                           = Matrix<Complex<float>, "IO", 1, 1>,
          IsMatrixType CBiasMatrixType                                   = Matrix<float, "C", 1>,
          typename SkipMatrixType                                        = Matrix<float, "IO", 1, 1>, // Trainable skip connection, either a matrix "IO" a vector "C" or a scalar "E"
          typename Lambda                                                = decltype([]() {}),
          IsMatrixType... ActivationMatrixInformation>
__attribute__((always_inline)) inline constexpr auto Sedge(AMatrixType     &&AMatrix,
                                                           BMatrixType     &&BMatrix,
                                                           BBiasMatrixType &&BBias,
                                                           CMatrixType     &&CMatrix,
                                                           CBiasMatrixType &&CBias,
                                                           SkipMatrixType  &&SkipMatrix,
                                                           Lambda          &&Act,
                                                           ActivationMatrixInformation &...ActivationParameters) noexcept {
    return DSSMLayer<OutputType, StateType, SuggestedSubBatchSizeComplex, SuggestedSubBatchSizeReal, true, BMACOperator_, CMACOperator_, DMACOperator_, SkipMACOperator_, AMatrixType, BMatrixType,
                     BBiasMatrixType, CMatrixType, CBiasMatrixType, Matrix<OutputType, "IO", 0, 0>, SkipMatrixType, Lambda, ActivationMatrixInformation...>(
            std::forward<AMatrixType>(AMatrix), std::forward<BMatrixType>(BMatrix), std::forward<BBiasMatrixType>(BBias), std::forward<CMatrixType>(CMatrix), std::forward<CBiasMatrixType>(CBias),
            Matrix<OutputType, "IO", 0, 0>(), std::forward<SkipMatrixType>(SkipMatrix), std::forward<Lambda>(Act), std::forward<ActivationMatrixInformation>(ActivationParameters)...);
}

template <typename OutputType                                            = float,
          typename StateType                                             = Complex<float>,
          std::size_t SuggestedSubBatchSizeComplex                       = 1,
          std::size_t SuggestedSubBatchSizeReal                          = 1,
          template <typename, typename, typename> class BMACOperator_    = NonMACOperation,
          template <typename, typename, typename> class CMACOperator_    = RealResultMACOperation,
          template <typename, typename, typename> class DMACOperator_    = NonMACOperation,
          template <typename, typename, typename> class SkipMACOperator_ = DefaultMACOperation,
          IsMatrixType AMatrixType                                       = Matrix<Complex<float>, "C", 1>,
          typename BMatrixType                                           = Matrix<float, "E", 1>,
          IsMatrixType BBiasMatrixType                                   = Matrix<float, "E", 1>,
          typename CMatrixType                                           = Matrix<Complex<float>, "IO", 1, 1>,
          IsMatrixType CBiasMatrixType                                   = Matrix<float, "C", 1>,
          typename SkipMatrixType                                        = Matrix<float, "IO", 1, 1>, // Trainable skip connection, either a matrix "IO" a vector "C" or a scalar "E"
          typename Lambda                                                = decltype([]() {}),
          IsMatrixType... ActivationMatrixInformation>
__attribute__((always_inline)) inline constexpr auto SedgeFirstLayerOp(AMatrixType &&AMatrix,
                                                                       //    BMatrixType     &&BMatrix,
                                                                       //    BBiasMatrixType &&BBias,
                                                                       CMatrixType     &&CMatrix,
                                                                       CBiasMatrixType &&CBias,
                                                                       SkipMatrixType  &&SkipMatrix,
                                                                       Lambda          &&Act,
                                                                       ActivationMatrixInformation &...ActivationParameters) noexcept {
    return DSSMLayer<OutputType, StateType, SuggestedSubBatchSizeComplex, SuggestedSubBatchSizeReal, true, BMACOperator_, CMACOperator_, DMACOperator_, SkipMACOperator_, AMatrixType, BMatrixType,
                     BBiasMatrixType, CMatrixType, CBiasMatrixType, Matrix<OutputType, "IO", 0, 0>, SkipMatrixType, Lambda, ActivationMatrixInformation...>(
            std::forward<AMatrixType>(AMatrix), Matrix<float, "E", 1>{}, Matrix<float, "E", 1>{}, std::forward<CMatrixType>(CMatrix), std::forward<CBiasMatrixType>(CBias),
            Matrix<OutputType, "IO", 0, 0>(), std::forward<SkipMatrixType>(SkipMatrix), std::forward<Lambda>(Act), std::forward<ActivationMatrixInformation>(ActivationParameters)...);
}

template <typename OutputType                                            = float,
          typename StateType                                             = Complex<float>,
          std::size_t SuggestedSubBatchSizeComplex                       = 1,
          std::size_t SuggestedSubBatchSizeReal                          = 1,
          template <typename, typename, typename> class BMACOperator_    = DefaultMACOperation,
          template <typename, typename, typename> class CMACOperator_    = RealResultMACOperation,
          template <typename, typename, typename> class DMACOperator_    = NonMACOperation,
          template <typename, typename, typename> class SkipMACOperator_ = NonMACOperation,
          IsMatrixType AMatrixType                                       = Matrix<Complex<float>, "C", 1>,
          typename BMatrixType                                           = Matrix<Complex<float>, "IO", 1, 1>,
          IsMatrixType BBiasMatrixType                                   = Matrix<Complex<float>, "C", 1>,
          typename CMatrixType                                           = Matrix<Complex<float>, "IO", 1, 1>,
          IsMatrixType CBiasMatrixType                                   = Matrix<float, "C", 1>,
          typename Lambda                                                = decltype([]() {}),
          IsMatrixType... ActivationMatrixInformation>
__attribute__((always_inline)) inline constexpr auto SSMPiano(AMatrixType     &&AMatrix,
                                                              BMatrixType     &&BMatrix,
                                                              BBiasMatrixType &&BBias,
                                                              CMatrixType     &&CMatrix,
                                                              CBiasMatrixType &&CBias,
                                                              Lambda          &&Act,
                                                              ActivationMatrixInformation &...ActivationParameters) noexcept {
    return DSSMLayer<OutputType, StateType, SuggestedSubBatchSizeComplex, SuggestedSubBatchSizeReal, false, BMACOperator_, CMACOperator_, DMACOperator_, SkipMACOperator_, AMatrixType, BMatrixType,
                     BBiasMatrixType, CMatrixType, CBiasMatrixType, Matrix<OutputType, "IO", 0, 0>, Matrix<OutputType, "IO", 0, 0>, Lambda, ActivationMatrixInformation...>(
            std::forward<AMatrixType>(AMatrix), std::forward<BMatrixType>(BMatrix), std::forward<BBiasMatrixType>(BBias), std::forward<CMatrixType>(CMatrix), std::forward<CBiasMatrixType>(CBias),
            Matrix<OutputType, "IO", 0, 0>(), Matrix<OutputType, "IO", 0, 0>(), std::forward<Lambda>(Act), std::forward<ActivationMatrixInformation>(ActivationParameters)...);
}

} // namespace layers
