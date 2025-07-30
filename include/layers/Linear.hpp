#pragma once

#include <tuple>
#include <utility>

#include "../Matrix.hpp"
#include "../functions/inference/Linear.hpp"
#include "../helpers/AccumulationTypes.hpp"
#include "../helpers/c++17_helpers.hpp"
#include "BaseLayer.hpp"

namespace layers {
/*
Type information for the Linear Layer
*/
template <typename Input, typename Weights, typename OutputType>
struct Linear_Generate_out_type_helper;

template <typename Input, typename Weights, typename OutputType>
using Linear_Generate_out_type = typename Linear_Generate_out_type_helper<remove_cvref_t<Input>, remove_cvref_t<Weights>, remove_cvref_t<OutputType>>::type;

template <typename WeightMatrixType>
struct Get_Type_t {
    using type = typename WeightMatrixType::type;
};

template <typename WeightMatrixType, typename WeightMatrixSpillType>
struct Get_Type_t<std::tuple<WeightMatrixType, WeightMatrixSpillType>> {
    using type = typename WeightMatrixType::type;
};

template <typename WeightMatrixType>
using Get_Type = typename Get_Type_t<WeightMatrixType>::type;

/*
Class for the Linear Layer
*/
template <typename OutputType,
          typename WeightMatrixType,
          typename AccumulationType,
          Dim_size_t InputChannels,  // Input Channels
          Dim_size_t OutputChannels, // Output Channels
          typename Lambda,
          typename... ActivationInformation>
class Linear_class_hidden : public BaseLayer {
  private:
    /* data */
    const WeightMatrixType                                                     Weights;
    const Matrix<AccumulationType, DimensionOrder::D1_Channel, OutputChannels> Bias;
    const Lambda                                                               Act;
    const std::tuple<const ActivationInformation &...>                         ActivationParameters;

  public:
    // Type information
    using WeightMatrix = Matrix<Get_Type<WeightMatrixType>, DimensionOrder::D2_OutChannel_InChannel, OutputChannels, InputChannels>;
    using BiasMatrix   = Matrix<AccumulationType, DimensionOrder::D1_Channel, OutputChannels>;

    template <typename InputMatrix>
    using OutputMatrix = Linear_Generate_out_type<InputMatrix, WeightMatrix, OutputType>;

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
    // Permanent Memory, Linear layers do not require permanent memory
    template <typename InputMatrix>
    static constexpr size_t MemoryPermanent = 0;

    // Constructor
    constexpr Linear_class_hidden(const WeightMatrixType                                                     &Weights,
                                  const Matrix<AccumulationType, DimensionOrder::D1_Channel, OutputChannels> &Bias,
                                  const Lambda                                                               &Act,
                                  const ActivationInformation &...ActivationParameters) noexcept
            : Weights(Weights), Bias(Bias), Act(Act), ActivationParameters(ActivationParameters...) {};

    // Forward pass
    template <Dim_size_t Batch, // Batch
              typename InputType>
    __attribute__((always_inline)) inline OutputMatrix<Matrix<InputType, DimensionOrder::D2_Batch_Channel, Batch, InputChannels>> operator()(
            const Matrix<InputType, DimensionOrder::D2_Batch_Channel, Batch, InputChannels> &Input) const noexcept {
        OutputMatrix<Matrix<InputType, DimensionOrder::D2_Batch_Channel, Batch, InputChannels>> out;
        operator()(Input, out, std::make_index_sequence<sizeof...(ActivationInformation)>());
        return out;
    }

    template <typename InputMatrixType, typename OutputMatrixType>
    __attribute__((always_inline)) inline void operator()(const InputMatrixType &Input, OutputMatrixType &Out) const noexcept {
        operator()(Input, Out, std::make_index_sequence<sizeof...(ActivationInformation)>());
    }

    template <typename InputMatrixType, typename OutputMatrixType>
    __attribute__((always_inline)) inline void operator()(
            const InputMatrixType &Input, OutputMatrixType &Out, Matrix < char, DimensionOrder::ERROR, MemoryBuffer<InputMatrixType>> &buffer) const noexcept {
        operator()(Input, Out, std::make_index_sequence<sizeof...(ActivationInformation)>());
    }

    template <typename InputMatrixType, typename OutputMatrixType, size_t... I>
    __attribute__((always_inline)) inline void operator()(const InputMatrixType &Input, OutputMatrixType &Out, const std::index_sequence<I...> &) const noexcept {
        functions::linear::Linear(Input, Out, Weights, Bias, Act, std::get<I>(ActivationParameters)...);
    }
};

template <typename OutputType = float,
          typename WeightMatrixType,
          typename AccumulationType,
          Dim_size_t InputChannels                           = WeightMatrixType::template Permutation<DimensionOrder::D2_OutChannel_InChannel>::dim2, // Input Channels
          Dim_size_t OutputChannels                          = WeightMatrixType::template Permutation<DimensionOrder::D2_OutChannel_InChannel>::dim1, // Weight Output Channels
          std::enable_if_t<WeightMatrixType::dims == 2, int> = 0,
          class Lambda,
          typename... ActivationInformation>
constexpr auto Linear(const WeightMatrixType                                                     &Weights,
                      const Matrix<AccumulationType, DimensionOrder::D1_Channel, OutputChannels> &Bias,
                      const Lambda                                                               &Act,
                      const ActivationInformation &...ActivationParameters) {
    return Linear_class_hidden<OutputType, WeightMatrixType, AccumulationType, InputChannels, OutputChannels, Lambda, ActivationInformation...>(Weights, Bias, Act, ActivationParameters...);
}

template <typename OutputType = float,
          typename WeightMatrixType,
          typename AccumulationType,
          Dim_size_t InputChannels                           = WeightMatrixType::dim2 * WeightMatrixType::dim4, // Input Channels
          Dim_size_t OutputChannels                          = WeightMatrixType::dim1 * WeightMatrixType::dim3, // Weight Output Channels
          std::enable_if_t<WeightMatrixType::dims == 4, int> = 0,                                               // Weight Matrix is 4D so D4_OutChannel_InChannel_KernelParallel_Unrolled
          class Lambda,
          typename... ActivationInformation>
constexpr auto Linear(const WeightMatrixType                                                     &Weights,
                      const Matrix<AccumulationType, DimensionOrder::D1_Channel, OutputChannels> &Bias,
                      const Lambda                                                               &Act,
                      const ActivationInformation &...ActivationParameters) {
    return Linear_class_hidden<OutputType, WeightMatrixType, AccumulationType, InputChannels, OutputChannels, Lambda, ActivationInformation...>(Weights, Bias, Act, ActivationParameters...);
}

template <typename OutputType = float,
          typename WeightMatrixType,      // Unrolled Parallelized Matrix
          typename WeightMatrixTypeSpill, // Rest of the Matrix that could not be parallelized
          typename AccumulationType,
          Dim_size_t InputChannels                           = WeightMatrixType::dim2 * WeightMatrixType::dim4, // Input Channels
          Dim_size_t OutputChannels                          = WeightMatrixType::dim1 * WeightMatrixType::dim3, // Weight Output Channels
          std::enable_if_t<WeightMatrixType::dims == 4, int> = 0,                                               // Weight Matrix is 4D so D4_OutChannel_InChannel_KernelParallel_Unrolled
          class Lambda,
          typename... ActivationInformation>
constexpr auto Linear(const std::tuple<WeightMatrixType, WeightMatrixTypeSpill>                  &Weights,
                      const Matrix<AccumulationType, DimensionOrder::D1_Channel, OutputChannels> &Bias,
                      const Lambda                                                               &Act,
                      const ActivationInformation &...ActivationParameters) {
    return Linear_class_hidden<OutputType, std::tuple<WeightMatrixType, WeightMatrixTypeSpill>, AccumulationType, InputChannels, OutputChannels, Lambda, ActivationInformation...>(
            Weights, Bias, Act, ActivationParameters...);
}

template <Dim_size_t Batch,          // Input batch
          Dim_size_t InputChannels,  // Input Channels
          Dim_size_t OutputChannels, // Weight Output Channels
          typename InputType,
          typename WeightType,
          typename OutputType>
struct Linear_Generate_out_type_helper<Matrix<InputType, DimensionOrder::D2_Batch_Channel, Batch, InputChannels>,
                                       Matrix<WeightType, DimensionOrder::D2_OutChannel_InChannel, OutputChannels, InputChannels>,
                                       OutputType> {
    using type = Matrix<OutputType, DimensionOrder::D2_Batch_Channel, Batch, OutputChannels>;
};

} // namespace layers
