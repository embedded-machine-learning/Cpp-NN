#pragma once

#include <concepts>
#include <stddef.h>
#include <type_traits>

#include "../Matrix.hpp"
#include "./BaseLayer.hpp"

namespace layers {

template<DimensionOrder ReductionOrder = "S">
class SumReduction {
  public:
    static_assert(ReductionOrder.length() == 1, "ReductionOrder must have a length of 1");

    using ExampleInputMatrix = Matrix<float, ReductionOrder+"BC", 1,1,1>;
    template <IsMatrixType InputMatrix>
    using OutputMatrix = OverrideRemoveDimensionMatrix<MaterializedMatrix<InputMatrix>, ReductionOrder>;

    template <IsMatrixType InputMatrix>
    using StateMatrixType = MaterializedMatrix<OutputMatrix<InputMatrix>>;

    // Memory Requirements of forward pass
    template <IsMatrixType InputMatrix>
    static constexpr std::size_t memory_minimal = std::max(sizeof(MaterializedMatrix<InputMatrix>), sizeof(MaterializedMatrix<OutputMatrix<InputMatrix>>));
    // Can it reuse the input Memory region?
    static constexpr bool memory_inlined = true;
    // Required Buffer size
    // the amount of Memory required for temporary storage
    template <IsMatrixType InputMatrix>
    static constexpr std::size_t memory_buffer = 0 ;
    // Permanent Memory which is required for some layers, if used in a time series model
    template <IsMatrixType InputMatrix>
    static constexpr std::size_t memory_permanent = sizeof(OutputMatrix<InputMatrix>);


    template <bool ContinueAfter=true, IsMatrixType InputMatrixType, IsMatrixType OutputMatrixType, IsMatrixType BufferMatrixType, IsMatrixType PermanentMatrixType>
    __attribute__((always_inline)) inline void operator()(const InputMatrixType &Input, OutputMatrixType &Out, [[maybe_unused]] BufferMatrixType &buffer, PermanentMatrixType &permanent) const noexcept {
        static_assert(InputMatrixType::order.containsAll(ReductionOrder), "Input must the ReductionOrder in its order");
        static_assert(!OutputMatrixType::order.containsAny(ReductionOrder), "Output must not contain the ReductionOrder in its order");

        static_assert(sizeof(permanent.data) >= sizeof(OutputMatrixType), "Permanent Memory Size does not match the required size for OutputMatrixType");

        auto& state= *reinterpret_cast<StateMatrixType<InputMatrixType>*>(&permanent.data[0]);

        auto state_expanded = permute<InputMatrixType::order>(conditionalBroadcast<ReductionOrder, {InputMatrixType::dimensions[InputMatrixType::order.indexOf(ReductionOrder.order[0])]}>(state));

        loopUnrolled([](auto& a, const auto& b){a+=b;}, state_expanded, Input);
        if constexpr (ContinueAfter) {
            loopUnrolled([](auto& a, const auto& b){a=b;}, Out, state);
        }
        
    }

};

static_assert(IsValidLayer<SumReduction<>>, "BaseLayer does not meet the requirements of a valid layer");

} // namespace layers