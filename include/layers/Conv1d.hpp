#pragma once

#include <tuple>
#include <utility>

#include "../Matrix.hpp"
#include "../functions/inference/Conv1d.hpp"
#include "../helpers/c++17_helpers.hpp"
#include "BaseLayer.hpp"

namespace layers {
/*
Type information for the Conv1d Layer
*/
template <Dim_size_t stride, Dim_size_t padding, typename Input, typename Weights, typename OutputType>
struct Conv1d_Generate_out_type_helper;

template <Dim_size_t stride, Dim_size_t padding, typename Input, typename Weights, typename OutputType>
using Conv1d_Generate_out_type = typename Conv1d_Generate_out_type_helper<stride, padding, remove_cvref_t<Input>, remove_cvref_t<Weights>, remove_cvref_t<OutputType>>::type;

/*
Class for the Conv1d Layer
*/
template <Dim_size_t stride,
          Dim_size_t padding,
          typename OutputType,
          typename WeightType,
          typename AccumulationType,
          Dim_size_t InputChannels,  // Input Channels
          Dim_size_t OutputChannels, // Weight Output Channels
          Dim_size_t Kernel,         // Weight Kernel shape
          class Lambda,
          typename... ActivationInformation>
class Conv1d_class_hidden : public BaseLayer {
  private:
    /* data */
    const Matrix<WeightType, DimensionOrder::D3_OutChannel_InChannel_Kernel, OutputChannels, InputChannels, Kernel> Weights;
    const Matrix<AccumulationType, DimensionOrder::D1_Channel, OutputChannels>                                      Bias;
    const Lambda                                                                                                    Act;
    const std::tuple<const ActivationInformation &...>                                                              ActivationParameters;

  public:
    // Type information
    using WeightMatrix = Matrix<WeightType, DimensionOrder::D3_OutChannel_InChannel_Kernel, OutputChannels, InputChannels, Kernel>;
    using BiasMatrix   = Matrix<AccumulationType, DimensionOrder::D1_Channel, OutputChannels>;

    template <typename InputMatrix>
    using OutputMatrix = Conv1d_Generate_out_type<stride, padding, InputMatrix, WeightMatrix, OutputType>;

    using BufferMatrix = WeightMatrix;

    // Memory Requirements of forward pass
    template <typename InputMatrix>
    static constexpr size_t MemoryMinimal = sizeof(InputMatrix) + sizeof(OutputMatrix<InputMatrix>);
    // Can it reuse the input Memory region?
    static constexpr bool MemoryInlined = false;
    // Dynamic increase if enough memory is available
    static constexpr size_t MemoryDynamic = sizeof(WeightMatrix);
    // Permanent Memory, Conv1d layers have a time series model, which requires permanent memory of the size of the weights - the "new" inputs
    template <typename InputMatrix>
    using PermanentMatrix = Matrix<void, DimensionOrder::ERROR, 0>; // TODO: Implement this !!!, should be the size of the weights - the "new" inputs, aka the old inputs shifted by one

    template <typename InputMatrix>
    static constexpr size_t MemoryPermanent = sizeof(PermanentMatrix<InputMatrix>);

    // Constructor
    constexpr Conv1d_class_hidden(const Matrix<WeightType, DimensionOrder::D3_OutChannel_InChannel_Kernel, OutputChannels, InputChannels, Kernel> &Weights,
                                  const Matrix<AccumulationType, DimensionOrder::D1_Channel, OutputChannels>                                      &Bias,
                                  const Lambda                                                                                                    &Act,
                                  const ActivationInformation &...ActivationParameters) noexcept
            : Weights(Weights), Bias(Bias), Act(Act), ActivationParameters(ActivationParameters...) {};

    // Forward pass
    template <typename InputMatrixType>
    __attribute__((always_inline)) inline OutputMatrix<InputMatrixType> operator()(const InputMatrixType &Input) const noexcept {
        OutputMatrix<InputMatrixType> Out;
        forward(Input, Out, std::make_index_sequence<sizeof...(ActivationInformation)>());
        return Out;
    }

    template <typename InputMatrixType>
    __attribute__((always_inline)) inline void operator()(const InputMatrixType &Input, OutputMatrix<InputMatrixType> &Out) const noexcept {
        forward(Input, Out, std::make_index_sequence<sizeof...(ActivationInformation)>());
    }

    template <typename InputMatrixType, size_t... I>
    __attribute__((always_inline)) inline void forward(const InputMatrixType &Input, OutputMatrix<InputMatrixType> &Out, const std::index_sequence<I...> &) const noexcept {
        functions::conv1d::Conv1d<stride, padding>(Input, Out, Weights, Bias, Act, std::get<I>(ActivationParameters)...);
    }
};

template <Dim_size_t stride,
          Dim_size_t padding,
          typename OutputType       = float,
          typename WeightType       = float,
          typename AccumulationType = float,
          Dim_size_t InputChannels,  // Input Channels
          Dim_size_t OutputChannels, // Weight Output Channels
          Dim_size_t Kernel,         // Weight Kernel shape
          class Lambda,
          typename... ActivationInformation>
constexpr auto Conv1d(const Matrix<WeightType, DimensionOrder::D3_OutChannel_InChannel_Kernel, OutputChannels, InputChannels, Kernel> &Weights,
                      const Matrix<AccumulationType, DimensionOrder::D1_Channel, OutputChannels>                                      &Bias,
                      const Lambda                                                                                                    &Act,
                      const ActivationInformation &...ActivationParameters) {
    return Conv1d_class_hidden<stride, padding, OutputType, WeightType, AccumulationType, InputChannels, OutputChannels, Kernel, Lambda, ActivationInformation...>(Weights, Bias, Act,
                                                                                                                                                                   ActivationParameters...);
}

template <Dim_size_t stride,
          Dim_size_t padding,
          Dim_size_t Batch,          // Input batch
          Dim_size_t InputChannels,  // Input Channels
          Dim_size_t InputWidth,     // Input Width
          Dim_size_t OutputChannels, // Weight Output Channels
          // Dim_size_t M2_2,	// Weight Input Channels has to be the same as InputChannels
          Dim_size_t Kernel, // Weight Kernel shape
          typename InputType,
          typename WeightType,
          typename OutputType>
struct Conv1d_Generate_out_type_helper<stride,
                                       padding,
                                       Matrix<InputType, DimensionOrder::D3_Batch_Channel_Width, Batch, InputChannels, InputWidth>,
                                       Matrix<WeightType, DimensionOrder::D3_OutChannel_InChannel_Kernel, OutputChannels, InputChannels, Kernel>,
                                       OutputType> {
    using type = Matrix<OutputType, DimensionOrder::D3_Batch_Channel_Width, Batch, OutputChannels, ((InputWidth - Kernel + 2 * padding) / stride + 1)>;
};

} // namespace layers