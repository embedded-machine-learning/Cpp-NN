#pragma once

#include <tuple>
#include <utility>

#include "../Matrix.hpp"
#include "../functions/inference/Conv2d.hpp"
#include "../helpers/c++17_helpers.hpp"
#include "BaseLayer.hpp"

namespace layers {
/*
Type information for the Conv2d Layer
*/
template <Dim_size_t stride, Dim_size_t padding, typename Input, typename Weights, typename OutputType>
struct Conv2d_Generate_out_type_helper;

template <Dim_size_t stride, Dim_size_t padding, typename Input, typename Weights, typename OutputType>
using Conv2d_Generate_out_type = typename Conv2d_Generate_out_type_helper<stride, padding, remove_cvref_t<Input>, remove_cvref_t<Weights>, remove_cvref_t<OutputType>>::type;

/*
Class for the Conv2d Layer
*/
template <Dim_size_t stride,
          Dim_size_t padding,
          typename OutputType,
          typename WeightType,
          typename AccumulationType,
          Dim_size_t InputChannels,  // Input Channels
          Dim_size_t OutputChannels, // Weight Output Channels
          Dim_size_t KernelWidth,    // Weight Kernel shape
          Dim_size_t KernelHeight,   // Weight Kernel shape
          class Lambda,
          typename... ActivationInformation>
class Conv2d_class_hidden : public BaseLayer {
  private:
    /* data */
    const Matrix<WeightType, DimensionOrder::D4_OutChannel_InChannel_KernelWidth_KernelHeight, OutputChannels, InputChannels, KernelWidth, KernelHeight> Weights;
    const Matrix<AccumulationType, DimensionOrder::D1_Channel, OutputChannels>                                                                           Bias;
    const Lambda                                                                                                                                         Act;
    const std::tuple<const ActivationInformation &...>                                                                                                   ActivationParameters;

  public:
    // Type information
    using WeightMatrix = Matrix<WeightType, DimensionOrder::D4_OutChannel_InChannel_KernelWidth_KernelHeight, OutputChannels, InputChannels, KernelWidth, KernelHeight>;
    using BiasMatrix   = Matrix<AccumulationType, DimensionOrder::D1_Channel, OutputChannels>;

    template <typename InputMatrix>
    using OutputMatrix = Conv2d_Generate_out_type<stride, padding, InputMatrix, WeightMatrix, OutputType>;

    using BufferMatrix = WeightMatrix;

    // Memory Requirements of forward pass
    template <typename InputMatrix>
    static constexpr size_t MemoryMinimal = sizeof(InputMatrix) + sizeof(OutputMatrix<InputMatrix>);
    // Can it reuse the input Memory region?
    static constexpr bool MemoryInlined = false;
    // Dynamic increase if enough memory is available
    static constexpr size_t MemoryDynamic = sizeof(WeightMatrix);
    // Required Buffer size
    template <typename InputMatrix>
    static constexpr size_t MemoryBuffer = 0;
    // Permanent Memory, Conv2d layers, i have no idea how they could be used in a time series model
    template <typename InputMatrix>
    static constexpr size_t MemoryPermanent = 0;

    // Constructor
    constexpr Conv2d_class_hidden(const Matrix<WeightType, DimensionOrder::D4_OutChannel_InChannel_KernelWidth_KernelHeight, OutputChannels, InputChannels, KernelWidth, KernelHeight> &Weights,
                                  const Matrix<AccumulationType, DimensionOrder::D1_Channel, OutputChannels>                                                                           &Bias,
                                  const Lambda                                                                                                                                         &Act,
                                  const ActivationInformation &...ActivationParameters) noexcept
            : Weights(Weights), Bias(Bias), Act(Act), ActivationParameters(ActivationParameters...) {};

    // Forward pass
    template <typename InputMatrixType>
    __attribute__((always_inline)) inline OutputMatrix<InputMatrixType> operator()(const InputMatrixType &Input) const noexcept {
        OutputMatrix<InputMatrixType> out;
        forward(Input, out, std::make_index_sequence<sizeof...(ActivationInformation)>());
        return out;
    }

    template <typename InputMatrixType>
    __attribute__((always_inline)) inline void operator()(const InputMatrixType &Input, OutputMatrix<InputMatrixType> &Out) const noexcept {
        forward(Input, Out, std::make_index_sequence<sizeof...(ActivationInformation)>());
    }

    template <typename InputMatrixType, size_t... I>
    __attribute__((always_inline)) inline void forward(const InputMatrixType &Input, OutputMatrix<InputMatrixType> &Out, const std::index_sequence<I...> &) const noexcept {
        functions::conv2d::Conv2d<stride, padding>(Input, Out, Weights, Bias, Act, std::get<I>(ActivationParameters)...);
    }
};

template <Dim_size_t stride,
          Dim_size_t padding,
          typename OutputType       = float,
          typename WeightType       = float,
          typename AccumulationType = float,
          Dim_size_t InputChannels,  // Input Channels
          Dim_size_t OutputChannels, // Weight Output Channels
          Dim_size_t KernelWidth,    // Weight Kernel shape
          Dim_size_t KernelHeight,   // Weight Kernel shape
          class Lambda,
          typename... ActivationInformation>
constexpr auto Conv2d(const Matrix<WeightType, DimensionOrder::D4_OutChannel_InChannel_KernelWidth_KernelHeight, OutputChannels, InputChannels, KernelWidth, KernelHeight> &Weights,
                      const Matrix<AccumulationType, DimensionOrder::D1_Channel, OutputChannels>                                                                           &Bias,
                      const Lambda                                                                                                                                         &Act,
                      const ActivationInformation &...ActivationParameters) {
    return Conv2d_class_hidden<stride, padding, OutputType, WeightType, AccumulationType, InputChannels, OutputChannels, KernelWidth, KernelHeight, Lambda, ActivationInformation...>(
            Weights, Bias, Act, ActivationParameters...);
}

template <Dim_size_t stride,
          Dim_size_t padding,
          Dim_size_t Batch,          // Input batch
          Dim_size_t InputChannels,  // Input Channels
          Dim_size_t InputWidth,     // Input Width
          Dim_size_t InputHeight,    // Input Width
          Dim_size_t OutputChannels, // Weight Output Channels
          // Dim_size_t M2_2,	// Weight Input Channels has to be the same as InputChannels
          Dim_size_t KernelWidth,  // Weight Kernel shape
          Dim_size_t KernelHeight, // Weight Kernel shape
          typename InputType,
          typename WeightType,
          typename OutputType>
struct Conv2d_Generate_out_type_helper<stride,
                                       padding,
                                       Matrix<InputType, DimensionOrder::D4_Batch_Channel_Width_Height, Batch, InputChannels, InputWidth, InputHeight>,
                                       Matrix<WeightType, DimensionOrder::D4_OutChannel_InChannel_KernelWidth_KernelHeight, OutputChannels, InputChannels, KernelWidth, KernelHeight>,
                                       OutputType> {
    using type = Matrix<OutputType,
                        DimensionOrder::D4_Batch_Channel_Width_Height,
                        Batch,
                        OutputChannels,
                        ((InputWidth - KernelWidth + 2 * padding) / stride + 1),
                        ((InputHeight - KernelHeight + 2 * padding) / stride + 1)>;
};

} // namespace layers