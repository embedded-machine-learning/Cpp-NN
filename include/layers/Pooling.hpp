#pragma once

#include <tuple>
#include <utility>

#include "../Matrix.hpp"
#include "../functions/inference/Pooling.hpp"
#include "../helpers/c++17_helpers.hpp"
#include "BaseLayer.hpp"

namespace layers {

/*
================================================================================================================================================
                                                        Flatten
================================================================================================================================================
*/

template <typename Input>
struct Flatten_Generate_out_type_helper;

template <typename Input>
using Flatten_Generate_out_type = typename Flatten_Generate_out_type_helper<remove_cvref_t<Input>>::type;

template <bool is_inlined = false>
class Flatten_class_hidden : public BaseLayer {
  public:
    // Type information
    template <typename InputMatrix>
    using OutputMatrix = Flatten_Generate_out_type<InputMatrix>;

    using BufferMatrix = Matrix<char, DimensionOrder::D1_Channel, 0>;

    // Constructor
    constexpr Flatten_class_hidden() noexcept = default;

    // Memory Requirements of forward pass
    template <typename InputMatrix>
    static constexpr size_t MemoryMinimal = sizeof(InputMatrix) + ((is_inlined) ? 0 : sizeof(OutputMatrix<InputMatrix>));
    // Can it reuse the input Memory region?
    static constexpr bool MemoryInlined = is_inlined;
    // Required Buffer size
    template <typename InputMatrix>
    static constexpr size_t MemoryBuffer = 0;
    // Dynamic increase if enough memory is available
    static constexpr size_t MemoryDynamic = 0;
    // Permanent Memory, Flatten layers do not require permanent memory
    template <typename InputMatrix>
    using PermanentMatrix = Matrix<void, DimensionOrder::ERROR, 0>;
    template <typename InputMatrix>
    static constexpr size_t MemoryPermanent = sizeof(PermanentMatrix<InputMatrix>);

    // Forward pass
    template <typename InputMatrix>
    __attribute__((always_inline)) inline OutputMatrix<InputMatrix> operator()(const InputMatrix &Input) const noexcept {
        return functions::Flatten(Input);
    }

    template <typename InputMatrix>
    __attribute__((always_inline)) inline void operator()(const InputMatrix &Input, OutputMatrix<InputMatrix> &Out) const noexcept {
        functions::Flatten(Input, Out);
    }
};

template <bool is_inlined = false>
constexpr auto Flatten() {
    return Flatten_class_hidden<is_inlined>();
}

template <typename Type, Dim_size_t M1_1, Dim_size_t M1_2, Dim_size_t M1_3, Dim_size_t M1_4>
struct Flatten_Generate_out_type_helper<Matrix<Type, DimensionOrder::D4_Batch_Channel_Width_Height, M1_1, M1_2, M1_3, M1_4>> {
    using type = Matrix<Type, DimensionOrder::D2_Batch_Channel, M1_1, M1_2 * M1_3 * M1_4>;
};

template <typename Type, Dim_size_t M1_1, Dim_size_t M1_2, Dim_size_t M1_3>
struct Flatten_Generate_out_type_helper<Matrix<Type, DimensionOrder::D3_Batch_Channel_Width, M1_1, M1_2, M1_3>> {
    using type = Matrix<Type, DimensionOrder::D2_Batch_Channel, M1_1, M1_2 * M1_3>;
};

/*
================================================================================================================================================
                                                        AdaptiveAveragePool
================================================================================================================================================
*/

template <typename Input>
struct AdaptiveAveragePool_Generate_out_type_helper;

template <typename Input>
using AdaptiveAveragePool_Generate_out_type = typename AdaptiveAveragePool_Generate_out_type_helper<remove_cvref_t<Input>>::type;

template <bool is_inlined = false>
class AdaptiveAveragePool_class_hidden : public BaseLayer {
  public:
    // Type information
    template <typename InputMatrix>
    using OutputMatrix = AdaptiveAveragePool_Generate_out_type<InputMatrix>;

    using BufferMatrix = Matrix<char, DimensionOrder::ERROR, 0>; // TODO: possible change to char[]

    // Memory Requirements of forward pass
    template <typename InputMatrix>
    static constexpr size_t MemoryMinimal = sizeof(InputMatrix) + ((is_inlined) ? 0 : sizeof(OutputMatrix<InputMatrix>));
    // Can it reuse the input Memory region?
    static constexpr bool MemoryInlined = is_inlined;
    // Dynamic increase if enough memory is available
    static constexpr size_t MemoryDynamic = 0;
    // Permanent Memory, AdaptiveAveragePool might require permanent memory if used in a time series model, like a running average
    template <typename InputMatrix>
    using PermanentMatrix = Matrix<void, DimensionOrder::ERROR, 0>; // TODO: Implement this, maybe
    template <typename InputMatrix>
    static constexpr size_t MemoryPermanent = sizeof(PermanentMatrix<InputMatrix>);

    // Constructor
    constexpr AdaptiveAveragePool_class_hidden() noexcept = default;

    // Forward pass
    template <typename InputMatrix>
    __attribute__((always_inline)) inline OutputMatrix<InputMatrix> operator()(const InputMatrix &Input) const noexcept {
        return functions::AdaptiveAveragePool(Input);
    }

    template <typename InputMatrix>
    __attribute__((always_inline)) inline void operator()(const InputMatrix &Input, OutputMatrix<InputMatrix> &Out) const noexcept {
        functions::AdaptiveAveragePool(Input, Out);
    }
};

template <bool is_inlined = false>
constexpr auto AdaptiveAveragePool() {
    return AdaptiveAveragePool_class_hidden<is_inlined>();
}

template <typename Type, DimensionOrder Order, Dim_size_t M1_1, Dim_size_t M1_2, Dim_size_t... dims>
struct AdaptiveAveragePool_Generate_out_type_helper<Matrix<Type, Order, M1_1, M1_2, dims...>> {
    static_assert(sizeof...(dims) > 1, "AdaptiveAveragePool is only for 3D or higher");
    static_assert(Order == DimensionOrder::D3_Batch_Channel_Width || Order == DimensionOrder::D4_Batch_Channel_Width_Height, "Unsupported Order");
    using type = Matrix<Type, DimensionOrder::D2_Batch_Channel, M1_1, M1_2>;
};

/*
================================================================================================================================================
                                                        MaxPool
================================================================================================================================================
*/

template <Dim_size_t pool_size, typename Input>
struct MaxPool_Generate_out_type_helper;

template <Dim_size_t pool_size, typename Input>
using MaxPool_Generate_out_type = typename MaxPool_Generate_out_type_helper<pool_size, remove_cvref_t<Input>>::type;

template <Dim_size_t pool_size, bool is_inlined = false>
class MaxPool_class_hidden : public BaseLayer {
  public:
    // Type information
    template <typename InputMatrix>
    using OutputMatrix = MaxPool_Generate_out_type<pool_size, InputMatrix>;

    using BufferMatrix = Matrix<char, DimensionOrder::ERROR, 0>; // TODO: Possible change to char[]

    // Memory Requirements of forward pass
    template <typename InputMatrix>
    static constexpr size_t MemoryMinimal = sizeof(InputMatrix) + ((is_inlined) ? 0 : sizeof(OutputMatrix<InputMatrix>));
    // Can it reuse the input Memory region?
    static constexpr bool MemoryInlined = is_inlined;
    // Required Buffer size
    template <typename InputMatrix>
    static constexpr size_t MemoryBuffer = 0;
    // Dynamic increase if enough memory is available
    static constexpr size_t MemoryDynamic = 0;
    // Permanent Memory, MaxPool layers might require permanent memory if used in a time series model, like a running maxpooling
    template <typename InputMatrix>
    using PermanentMatrix = Matrix<void, DimensionOrder::ERROR, 0>; // TODO: Implement this, maybe

    // Constructor
    constexpr MaxPool_class_hidden() noexcept = default;

    // Forward pass
    template <typename InputMatrix>
    __attribute__((always_inline)) inline OutputMatrix<InputMatrix> operator()(const InputMatrix &Input) const noexcept {
        return functions::MaxPool<pool_size>(Input);
    }

    template <typename InputMatrix>
    __attribute__((always_inline)) inline void operator()(const InputMatrix &Input, OutputMatrix<InputMatrix> &Out) const noexcept {
        functions::MaxPool<pool_size>(Input, Out);
    }
};

template <Dim_size_t pool_size, bool is_inlined = false>
constexpr auto MaxPool() {
    return MaxPool_class_hidden<pool_size, is_inlined>();
}

template <Dim_size_t pool_size, typename Type, Dim_size_t M1_1, Dim_size_t M1_2, Dim_size_t M1_3, Dim_size_t M1_4>
struct MaxPool_Generate_out_type_helper<pool_size, Matrix<Type, DimensionOrder::D4_Batch_Channel_Width_Height, M1_1, M1_2, M1_3, M1_4>> {
    using type = Matrix<Type, DimensionOrder::D4_Batch_Channel_Width_Height, M1_1, M1_2, M1_3 / pool_size, M1_4 / pool_size>;
};

template <Dim_size_t pool_size, typename Type, Dim_size_t M1_1, Dim_size_t M1_2, Dim_size_t M1_3>
struct MaxPool_Generate_out_type_helper<pool_size, Matrix<Type, DimensionOrder::D3_Batch_Channel_Width, M1_1, M1_2, M1_3>> {
    using type = Matrix<Type, DimensionOrder::D3_Batch_Channel_Width, M1_1, M1_2, M1_3 / pool_size>;
};

} // namespace layers
