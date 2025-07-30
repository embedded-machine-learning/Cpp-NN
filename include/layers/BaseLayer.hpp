#pragma once

#include <stddef.h>
#include <type_traits>

#include "../Matrix.hpp"

namespace layers {

class BaseLayer {
  public:
    // Memory Requirements of forward pass
    template <typename InputMatrix>
    static constexpr size_t MemoryMinimal = 0;
    // Can it reuse the input Memory region?
    static constexpr bool MemoryInlined = false;
    // Required Buffer size
    // the amount of Memory required for temporary storage
    static constexpr size_t MemoryBuffer = 0;
    // Dynamic increase if enough memory is available, might change behavior of the layer, like buffering reused weights in fast memory instead of slow storage
    static constexpr size_t MemoryDynamic = 0;
    // Permanent Memory which is required for some layers, if used in a time series model
    template <typename InputMatrix>
    static constexpr size_t MemoryPermanent = 0;

    template <typename InputMatrix>
    using OutputMatrix = Matrix<void, DimensionOrder::ERROR, 0>;

    // Constructor
    constexpr BaseLayer() = default;

    // To catch all the cases where the operator is not implemented, at the same time define the parameters
    template <typename InputType, typename OutputType>
    __attribute__((always_inline)) inline void operator()(const InputType &Input, OutputType &Out) const noexcept {
        static_assert(!std::is_same<typename InputType::type, void>::value, "Not Implemented");
    }

    template <typename InputType>
    __attribute__((always_inline)) inline OutputMatrix<InputType> operator()(const InputType &Input) const noexcept {
        static_assert(!std::is_same<typename InputType::type, void>::value, "Not Implemented");
    }
};

} // namespace layers