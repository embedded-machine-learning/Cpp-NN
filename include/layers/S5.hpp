#pragma once

#include <stddef.h>
#include <type_traits>

#include "../Matrix.hpp"
#include "../helpers/Complex.hpp"

#include "../functions/inference/Linear.hpp"
#include "./Linear.hpp"

/* Activation Function to mimic x=Ax+Bu*/
struct Elementwise_Mult_Add {
    template <typename input, typename output, typename AType, typename xType>
    __attribute__((always_inline)) static inline output Act(const input val, const AType A, const xType x) noexcept {
        return val + A * x;
    }
};

// template <>
// __attribute__((always_inline)) inline Complex<float> Elementwise_Mult_Add::Act<Complex<float>, Complex<float>, Complex<float>, Complex<float>>(Complex<float> Bu,
//                                                                                                                                                Complex<float> A,
//                                                                                                                                                Complex<float> x) noexcept {
//     return Bu + A * x;
// }

// template <>
// __attribute__((always_inline)) inline Complex<double> Elementwise_Mult_Add::Act<Complex<double>, Complex<double>, Complex<double>, Complex<double>>(Complex<double> Bu,
//                                                                                                                                                     Complex<double> A,
//                                                                                                                                                     Complex<double> x) noexcept {
//     return Bu + A * x;
// }

template <typename OtherAct>
struct Real_Wrapper {
    template <typename input, typename output, typename... ActivationParameters>
    __attribute__((always_inline)) static inline output Act(const input val, ActivationParameters... params) noexcept {
        return OtherAct::template Act<typename input::value_type, output, ActivationParameters...>(val.real(), params...);
    }
};

namespace layers {

template <
        // typename OutputType,
        //   typename AccumulationType,
        //   size_t NumberOfStates,
        typename WeightMatrixAType,         // Type... Matrix<Complex<Type>, DimensionOrder::D1_Channel, NumberOfStates>
        typename WeightMatrixBType,         // Type... Matrix<Complex<Type>, DimensionOrder::D2_OutChannel_InChannel, NumberOfStates, InputDim>
        typename WeightMatrixCType,         // Type... Matrix<Complex<Type>, DimensionOrder::D2_OutChannel_InChannel, NumberOfStates, OutputDim>
        typename BiasRNNMatrixType,         // Type... Matrix<Complex<Type>, DimensionOrder::D1_Channel, NumberOfStates>
        typename BiasOutMatrixType,         // Type... Matrix<Type, DimensionOrder::D1_Channel, OutputDim>
        typename WeightMatrixSkipLayerType, // Type... Matrix<Type, DimensionOrder::D2_OutChannel_InChannel, OutputDim, InputDim>
        typename Lambda,
        typename... ActivationInformation>
class S5_class_hidden : public BaseLayer {


    // static_assert(WeightMatrixAType::dims == 1, "Matrix A must be 1D");
    // static_assert(std::is_same<typename WeightMatrixAType::type, Complex<AccumulationType>>::value, "Matrix A must be of type Complex<AccumulationType>");
    // static_assert(WeightMatrixBType::dims == 2, "Matrix B must be 2D");
    // static_assert(std::is_same<typename WeightMatrixBType::type, Complex<AccumulationType>>::value, "Matrix B must be of type Complex<AccumulationType>");
    // static_assert(WeightMatrixCType::dims == 2, "Matrix C must be 2D");
    // static_assert(std::is_same<typename WeightMatrixCType::type, Complex<AccumulationType>>::value, "Matrix C must be of type Complex<AccumulationType>");
    // static_assert(WeightMatrixCType::order == D2_OutChannel_InChannel, "Matrix C must be in the order D2_OutChannel_InChannel, rest not implemented yet");

  private:
    /* data */
    const WeightMatrixAType                            MatrixA;
    const WeightMatrixBType                            MatrixB;
    const WeightMatrixCType                            MatrixC;
    const BiasRNNMatrixType                            BiasRNN;
    const BiasOutMatrixType                            BiasOut;
    const WeightMatrixSkipLayerType                    SkipLayer;
    const Lambda                                       Act;
    const std::tuple<const ActivationInformation &...> ActivationParameters;

    using WeightMatrixBEquivalentType = functions::linear::get_un_unrolled_Matrix<WeightMatrixBType>;
    using WeightMatrixCEquivalentType = functions::linear::get_un_unrolled_Matrix<WeightMatrixCType>;
    using WeightMatrixSkipLayerEquivalentType = functions::linear::get_un_unrolled_Matrix<WeightMatrixSkipLayerType>;


    constexpr static size_t InputChannels  = WeightMatrixBEquivalentType::dim2;
    constexpr static size_t OutputChannels = WeightMatrixCEquivalentType::dim1;
    
    using AccumulationType                 = typename WeightMatrixAType::type::value_type;
    using OutputType                       = AccumulationType;
    constexpr static size_t NumberOfStates = WeightMatrixAType::dim1;
    
    // Matrix<Complex<AccumulationType>, DimensionOrder::D1_Channel, NumberOfStates> States{};

  public:
    // TODO: Add accurate memory stuff
    template <typename InputMatrix>
    using OutputMatrix = Matrix<OutputType,
                                DimensionOrder::D3_Batch_Sequence_Channel,
                                InputMatrix::template Permutation<DimensionOrder::D3_Batch_Sequence_Channel>::dim1,
                                InputMatrix::template Permutation<DimensionOrder::D3_Batch_Sequence_Channel>::dim2,
                                OutputChannels>;

    using BufferMatrix = Matrix<char, DimensionOrder::D1_Channel, 0>;

    // Memory Requirements of forward pass
    template <typename InputMatrix>
    static constexpr size_t MemoryMinimal = sizeof(InputMatrix) + sizeof(OutputMatrix<InputMatrix>);
    // Can it reuse the input Memory region?
    static constexpr bool MemoryInlined = false;
    // Required Buffer size
    template <typename InputMatrix>
    static constexpr size_t MemoryBuffer = 0;
    // Dynamic increase if enough memory is available
    static constexpr size_t MemoryDynamic = 0;
    // Permanent Memory, S5 Layers require permanent memory of the size of the states
    template <typename InputMatrix>
    using PermanentMatrix = Matrix<Complex<AccumulationType>, DimensionOrder::D2_Batch_Channel, InputMatrix::template Permutation<DimensionOrder::D3_Batch_Sequence_Channel>::dim1, NumberOfStates>;
    template <typename InputMatrix>
    static constexpr size_t MemoryPermanent = sizeof(PermanentMatrix<InputMatrix>);

    // Constructor
    // constexpr S5_class_hidden(const WeightMatrixAType &MatrixA,
    //                           const WeightMatrixBType &MatrixB,
    //                           const WeightMatrixCType &MatrixC,
    //                           const BiasRNNMatrixType &BiasRNN,
    //                           const BiasOutMatrixType &BiasOut,
    //                           const Lambda            &Act,
    //                           const ActivationInformation &...ActivationParameters) noexcept
    //         : MatrixA(MatrixA), MatrixB(MatrixB), MatrixC(MatrixC), BiasRNN(BiasRNN), BiasOut(BiasOut), SkipLayer(Matrix<char, DimensionOrder::ERROR, 0>()), Act(Act),
    //           ActivationParameters(ActivationParameters...) {};
    // Constructor
    constexpr S5_class_hidden(const WeightMatrixAType         &MatrixA,
                              const WeightMatrixBType         &MatrixB,
                              const WeightMatrixCType         &MatrixC,
                              const BiasRNNMatrixType         &BiasRNN,
                              const BiasOutMatrixType         &BiasOut,
                              const WeightMatrixSkipLayerType &SkipLayer,
                              const Lambda                    &Act,
                              const ActivationInformation &...ActivationParameters) noexcept
            : MatrixA(MatrixA), MatrixB(MatrixB), MatrixC(MatrixC), BiasRNN(BiasRNN), BiasOut(BiasOut), SkipLayer(SkipLayer), Act(Act), ActivationParameters(ActivationParameters...) {};

    // Forward pass
    template <typename InputMatrixType>
    __attribute__((always_inline)) inline OutputMatrix<InputMatrixType> operator()(const InputMatrixType &Input // line break
    ) const noexcept {
        OutputMatrix<InputMatrixType>                                         Out;
        Matrix<char, DimensionOrder::ERROR, MemoryPermanent<InputMatrixType>> permanentMemory{};
        Matrix<char, DimensionOrder::ERROR, MemoryBuffer<InputMatrixType>>    buffer{};
        operator()(Input, Out, std::integer_sequence<bool, true>(), permanentMemory, buffer);
        return Out;
    }

    template <typename InputMatrixType>
    __attribute__((always_inline)) inline void operator()(const InputMatrixType         &Input, // line break
                                                          OutputMatrix<InputMatrixType> &Out) const noexcept {
        Matrix<char, DimensionOrder::ERROR, MemoryPermanent<InputMatrixType>> permanentMemory{};
        Matrix<char, DimensionOrder::ERROR, MemoryBuffer<InputMatrixType>>    buffer{};
        operator()(Input, Out, std::integer_sequence<bool, true>(), buffer, permanentMemory);
    }

    template <typename InputMatrixType, typename OutputMatrixType, size_t... I>
    __attribute__((always_inline)) inline void operator()(const InputMatrixType                                              &Input, // line break
                                                          OutputMatrixType                                                   &Out,
                                                          Matrix<char, DimensionOrder::ERROR, MemoryBuffer<InputMatrixType>> &buffer) const noexcept {
        Matrix<char, DimensionOrder::ERROR, MemoryPermanent<InputMatrixType>> permanentMemory{};
        operator()(Input, Out, std::integer_sequence<bool, true>(), buffer, permanentMemory);
    }

    template <typename InputMatrixType, typename OutputMatrixType, size_t... I>
    __attribute__((always_inline)) inline void operator()(const InputMatrixType                                                 &Input,
                                                          OutputMatrixType                                                      &Out,
                                                          Matrix<char, DimensionOrder::ERROR, MemoryBuffer<InputMatrixType>>    &buffer,
                                                          Matrix<char, DimensionOrder::ERROR, MemoryPermanent<InputMatrixType>> &permanentMemory) const noexcept {
        operator()(Input, Out, std::integer_sequence<bool, true>(), buffer, permanentMemory, std::make_index_sequence<sizeof...(ActivationInformation)>());
    }

    template <bool ProduceOutput, typename InputMatrixType, typename OutputMatrixType, size_t... I>
    __attribute__((always_inline)) inline void operator()(const InputMatrixType                                                 &Input,
                                                          OutputMatrixType                                                      &Out,
                                                          const std::integer_sequence<bool, ProduceOutput>                      &ProduceOutputPlaceholder /* ProduceOutput */,
                                                          Matrix<char, DimensionOrder::ERROR, MemoryBuffer<InputMatrixType>>    &buffer,
                                                          Matrix<char, DimensionOrder::ERROR, MemoryPermanent<InputMatrixType>> &permanentMemory) const noexcept {
        operator()(Input, Out, ProduceOutputPlaceholder, buffer, permanentMemory, std::make_index_sequence<sizeof...(ActivationInformation)>());
    }

    template < // line break
            bool ProduceOutput,
            typename InputMatrixType,
            typename OutputMatrixType,
            std::enable_if_t<WeightMatrixSkipLayerEquivalentType::order == DimensionOrder::ERROR && InputMatrixType::order != DimensionOrder::ERROR, int> = 0,
            size_t... I>
    __attribute__((always_inline)) inline void operator()(const InputMatrixType                                                 &Input, // Input
                                                          OutputMatrixType                                                      &Out,   // Output
                                                          const std::integer_sequence<bool, ProduceOutput>                      &_ /* ProduceOutput */,
                                                          Matrix<char, DimensionOrder::ERROR, MemoryBuffer<InputMatrixType>>    &buffer,
                                                          Matrix<char, DimensionOrder::ERROR, MemoryPermanent<InputMatrixType>> &permanentMemory,
                                                          const std::index_sequence<I...> &) const noexcept {
        static_assert(InputMatrixType::order == DimensionOrder::D3_Batch_Sequence_Channel, "Input must be in the order D3_Batch_Sequence_Channel");
        static_assert(OutputMatrixType::order == DimensionOrder::D3_Batch_Sequence_Channel, "Output must be in the order D3_Batch_Sequence_Channel");
        static_assert(InputMatrixType::template Permutation<DimensionOrder::D3_Batch_Sequence_Channel>::dim1 == 1, "Batch size must be 1, rest not implemented yet");

        constexpr size_t SequenceLength = InputMatrixType::template Permutation<DimensionOrder::D3_Batch_Sequence_Channel>::dim2;
        using InputSliceType            = const Matrix<typename InputMatrixType::type, DimensionOrder::D2_Batch_Channel, 1, InputChannels>;
        using OutputSliceType           = Matrix<typename OutputMatrixType::type, DimensionOrder::D2_Batch_Channel, 1, OutputChannels>;
        using StateBiasType             = Matrix<Complex<AccumulationType>, DimensionOrder::D1_Channel, NumberOfStates>;

        PermanentMatrix<InputMatrixType> *states     = reinterpret_cast<PermanentMatrix<InputMatrixType> *>(&permanentMemory);
        StateBiasType                    *state_bias = reinterpret_cast<StateBiasType *>(states);

        // Go through the Sequence One by One
        for (size_t seq_pos = 0; seq_pos < SequenceLength; seq_pos++) {
            InputSliceType  *InputSeq = reinterpret_cast<InputSliceType *>(&Input.at(0, seq_pos, 0));
            OutputSliceType *OutSeq   = reinterpret_cast<OutputSliceType *>(&Out.at(0, seq_pos, 0));

            // Linear passed arguments ( input, Output, Weight, Bias, Act, Actparams)
            // x = Ax + Bu
            functions::linear::Linear(*InputSeq, *states, MatrixB, BiasRNN, Elementwise_Mult_Add(), MatrixA, *state_bias);

            if constexpr (ProduceOutput) {
                // y = Cx
                // functions::linear::Linear(*states, *OutSeq, MatrixC, BiasOut, Real_Wrapper<Lambda>(), std::get<I>(ActivationParameters)...);
                functions::linear::Linear(*states, *OutSeq, MatrixC, BiasOut, Lambda(), std::get<I>(ActivationParameters)...);
                if constexpr (InputChannels == 1) {
                    for (size_t i = 0; i < OutputChannels; i++) {
                        OutSeq->at(0, i) += InputSeq->at(0, 0);
                    }
                } else {
                    for (size_t i = 0; i < OutputChannels; i++) {
                        OutSeq->at(0, i) += InputSeq->at(0, i);
                    }
                }
            }
        }
    }

    template < // line break
            bool ProduceOutput,
            typename InputMatrixType,
            typename OutputMatrixType,
            std::enable_if_t<WeightMatrixSkipLayerEquivalentType::order != DimensionOrder::ERROR && InputMatrixType::order != DimensionOrder::ERROR, int> = 0,
            size_t... I>
    __attribute__((always_inline)) inline void operator()(const InputMatrixType                                                 &Input, // Input
                                                          OutputMatrixType                                                      &Out,   // Output
                                                          const std::integer_sequence<bool, ProduceOutput>                      &_ /* ProduceOutput */,
                                                          Matrix<char, DimensionOrder::ERROR, MemoryBuffer<InputMatrixType>>    &buffer,
                                                          Matrix<char, DimensionOrder::ERROR, MemoryPermanent<InputMatrixType>> &permanentMemory,
                                                          const std::index_sequence<I...> &) const noexcept {
        static_assert(InputMatrixType::order == DimensionOrder::D3_Batch_Sequence_Channel, "Input must be in the order D3_Batch_Sequence_Channel");
        static_assert(OutputMatrixType::order == DimensionOrder::D3_Batch_Sequence_Channel, "Output must be in the order D3_Batch_Sequence_Channel");
        static_assert(InputMatrixType::template Permutation<DimensionOrder::D3_Batch_Sequence_Channel>::dim1 == 1, "Batch size must be 1, rest not implemented yet");

        constexpr size_t SequenceLength = InputMatrixType::template Permutation<DimensionOrder::D3_Batch_Sequence_Channel>::dim2;
        using InputSliceType            = const Matrix<typename InputMatrixType::type, DimensionOrder::D2_Batch_Channel, 1, InputChannels>;
        using OutputSliceType           = Matrix<typename OutputMatrixType::type, DimensionOrder::D2_Batch_Channel, 1, OutputChannels>;
        using OutputBiasType            = Matrix<typename OutputMatrixType::type, DimensionOrder::D1_Channel, OutputChannels>;
        using StateBiasType             = Matrix<Complex<AccumulationType>, DimensionOrder::D1_Channel, NumberOfStates>;

        PermanentMatrix<InputMatrixType> *states     = reinterpret_cast<PermanentMatrix<InputMatrixType> *>(&permanentMemory);
        StateBiasType                    *state_bias = reinterpret_cast<StateBiasType *>(states);

        // Go through the Sequence One by One
        for (size_t seq_pos = 0; seq_pos < SequenceLength; seq_pos++) {
            InputSliceType  *InputSeq                 = reinterpret_cast<InputSliceType *>(&Input.at(0, seq_pos, 0));
            OutputSliceType *OutSeq                   = reinterpret_cast<OutputSliceType *>(&Out.at(0, seq_pos, 0));
            OutputBiasType  *OutputBiasRepresentation = reinterpret_cast<OutputBiasType *>(&Out.at(0, seq_pos, 0));

            // Linear passed arguments ( input, Output, Weight, Bias, Act, Actparams)
            // x = Ax + Bu
            functions::linear::Linear(*InputSeq, *states, MatrixB, BiasRNN, Elementwise_Mult_Add(), MatrixA, *state_bias);

            if constexpr (ProduceOutput) {
                // y = Cx
                // functions::linear::Linear(*states, *OutSeq, MatrixC, BiasOut, Real_Wrapper<Lambda>(), std::get<I>(ActivationParameters)...);
                functions::linear::Linear(*states, *OutSeq, MatrixC, BiasOut, Lambda(), std::get<I>(ActivationParameters)...);

                // Skip Layer
                functions::linear::Linear(*InputSeq, *OutSeq, SkipLayer, *OutputBiasRepresentation, Passthrough);
            }
        }
    }
};

template <typename WeightMatrixAType,         // Type... Matrix<Complex<Type>, DimensionOrder::D1_Channel, NumberOfStates>
          typename WeightMatrixBType,         // Type... Matrix<Complex<Type>, DimensionOrder::D2_OutChannel_InChannel, NumberOfStates, InputDim>
          typename WeightMatrixCType,         // Type... Matrix<Complex<Type>, DimensionOrder::D2_OutChannel_InChannel, NumberOfStates, OutputDim>
          typename BiasRNNMatrixType,         // Type... Matrix<Complex<Type>, DimensionOrder::D1_Channel, NumberOfStates>
          typename BiasOutMatrixType,         // Type... Matrix<Type, DimensionOrder::D1_Channel, OutputDim>
          typename WeightMatrixSkipLayerType, // Type... Matrix<Type, DimensionOrder::D2_OutChannel_InChannel, OutputDim, InputDim>
          typename Lambda,
          typename... ActivationInformation>
constexpr auto S5(const WeightMatrixAType         &MatrixA,
                  const WeightMatrixBType         &MatrixB,
                  const WeightMatrixCType         &MatrixC,
                  const BiasRNNMatrixType         &BiasRNN,
                  const BiasOutMatrixType         &BiasOut,
                  const WeightMatrixSkipLayerType &SkipLayer,
                  const Lambda                    &Act,
                  const ActivationInformation &...ActivationParameters) noexcept {
    return S5_class_hidden<WeightMatrixAType, WeightMatrixBType, WeightMatrixCType, BiasRNNMatrixType, BiasOutMatrixType, WeightMatrixSkipLayerType, Lambda, ActivationInformation...>(
            MatrixA, MatrixB, MatrixC, BiasRNN, BiasOut, SkipLayer, Act, ActivationParameters...);
}

template <typename WeightMatrixAType, // Type... Matrix<Complex<Type>, DimensionOrder::D1_Channel, NumberOfStates>
          typename WeightMatrixBType, // Type... Matrix<Complex<Type>, DimensionOrder::D2_OutChannel_InChannel, NumberOfStates, InputDim>
          typename WeightMatrixCType, // Type... Matrix<Complex<Type>, DimensionOrder::D2_OutChannel_InChannel, NumberOfStates, OutputDim>
          typename BiasRNNMatrixType, // Type... Matrix<Complex<Type>, DimensionOrder::D1_Channel, NumberOfStates>
          typename BiasOutMatrixType, // Type... Matrix<Type, DimensionOrder::D1_Channel, OutputDim>
          typename Lambda,
          typename... ActivationInformation>
constexpr auto S5(const WeightMatrixAType &MatrixA,
                  const WeightMatrixBType &MatrixB,
                  const WeightMatrixCType &MatrixC,
                  const BiasRNNMatrixType &BiasRNN,
                  const BiasOutMatrixType &BiasOut,
                  const Lambda            &Act,
                  const ActivationInformation &...ActivationParameters) noexcept {
    return S5_class_hidden<WeightMatrixAType, WeightMatrixBType, WeightMatrixCType, BiasRNNMatrixType, BiasOutMatrixType, Matrix<char, DimensionOrder::ERROR, 0>, Lambda, ActivationInformation...>(
            MatrixA, MatrixB, MatrixC, BiasRNN, BiasOut, Matrix<char, DimensionOrder::ERROR, 0>(), Act, ActivationParameters...);
}

} // namespace layers