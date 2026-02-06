#pragma once

#include <concepts>
#include <stddef.h>
#include <type_traits>

#include "../Matrix.hpp"

namespace layers {

class BaseLayer {
  public:
    // Memory Requirements of forward pass
    template <IsMatrixType InputMatrix>
    static constexpr std::size_t memory_minimal = 0;
    // Can it reuse the input Memory region?
    static constexpr bool memory_inlined = false;
    // Required Buffer size
    // the amount of Memory required for temporary storage
    template <IsMatrixType InputMatrix>
    static constexpr std::size_t memory_buffer = 0;
    // Permanent Memory which is required for some layers, if used in a time series model
    template <IsMatrixType InputMatrix>
    static constexpr std::size_t memory_permanent = 0;

    using ExampleInputMatrix = Matrix<char, "E", 0>;
    template <typename InputMatrix>
    using OutputMatrix = Matrix<char, "E", 0>;

    // Constructor
    constexpr BaseLayer() = default;

// To catch all the cases where the operator is not implemented, at the same time define the parameters
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"

    template <IsMatrixType InputMatrixType, IsMatrixType OutputMatrixType, IsBaseMatrixType BufferMatrixType, IsBaseMatrixType PermanentMatrixType>
    __attribute__((always_inline)) inline void operator()(const InputMatrixType &Input, OutputMatrixType &Out, BufferMatrixType &buffer, PermanentMatrixType &permanent) const noexcept {
        static_assert(!std::is_same<typename InputMatrixType::type, void>::value, "Not Implemented");
    }

#pragma clang diagnostic pop
};

template <typename LayerType>
constexpr std::size_t get_memory_buffer_size = LayerType::template memory_buffer<typename LayerType::ExampleInputMatrix>;
template <typename LayerType>
constexpr std::size_t get_memory_permanent_size = LayerType::template memory_permanent<typename LayerType::ExampleInputMatrix>;
template <typename LayerType>
using get_ExpectedInputMatrix = typename LayerType::ExampleInputMatrix;
template <typename LayerType>
using get_ExpectedOutputMatrix = typename LayerType::template OutputMatrix<typename LayerType::ExampleInputMatrix>;

template <typename LayerType>
concept IsValidLayer = requires(LayerType layer) {
    typename LayerType::ExampleInputMatrix;
    typename LayerType::template OutputMatrix<typename LayerType::ExampleInputMatrix>;
    { LayerType::memory_inlined } -> std::convertible_to<bool>;
    { LayerType::template memory_buffer<typename LayerType::ExampleInputMatrix> } -> std::convertible_to<std::size_t>;
    { LayerType::template memory_minimal<typename LayerType::ExampleInputMatrix> } -> std::convertible_to<std::size_t>;
    { LayerType::template memory_permanent<typename LayerType::ExampleInputMatrix> } -> std::convertible_to<std::size_t>;
    {
        layer(std::declval<get_ExpectedInputMatrix<LayerType>>(), std::declval<get_ExpectedOutputMatrix<LayerType> &>(), std::declval<Matrix<char, "E", get_memory_buffer_size<LayerType>> &>(),
              std::declval<Matrix<char, "E", get_memory_permanent_size<LayerType>> &>())
    } -> std::same_as<void>;
};

static_assert(IsValidLayer<BaseLayer>, "BaseLayer does not meet the requirements of a valid layer");

} // namespace layers