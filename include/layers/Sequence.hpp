#pragma once
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "../helpers/cpp_helpers.hpp"
#include "./BaseLayer.hpp"

namespace layers {

struct MemoryLocation {
    std::size_t Input_index;
    std::size_t Input_size;
    std::size_t Output_index;
    std::size_t Output_size;
    std::size_t buffer_index;
    std::size_t buffer_size;
    std::size_t permanent_index; // carefull peramemt memory is in a different buffer than the dynamic memory
    std::size_t permanent_size;
};

template <IsMatrixType Input, typename LayerTuple, typename IndexSequence>
struct MemoryPlaning;

#warning "TODO: Add the ability to skip the first and last layer"
#warning "TODO: add aligning"

template <IsMatrixType Input, layers::IsValidLayer... Layers, std::size_t Current, std::size_t... Indices>
struct MemoryPlaning<Input, std::tuple<Layers...>, std::index_sequence<Current, Indices...>> {
    using LayerTuple       = std::tuple<Layers...>;
    using CurrentLayer     = std::tuple_element_t<Current, LayerTuple>;
    using RemainingIndexes = std::index_sequence<Indices...>;
    using Input_           = MaterializedMatrix<Input>;
    using Output_          = MaterializedMatrix<typename CurrentLayer::template OutputMatrix<Input_>>;
    using Next             = MemoryPlaning<Output_, LayerTuple, RemainingIndexes>;

    constexpr static std::size_t memory_permanent     = CurrentLayer::template memory_permanent<Input_>;
    constexpr static std::size_t memory_minimal       = CurrentLayer::template memory_minimal<Input_>;
    constexpr static std::size_t memory_buffer        = CurrentLayer::template memory_buffer<Input_>;
    constexpr static bool        memory_inlined       = CurrentLayer::memory_inlined;
    constexpr static std::size_t memory_inline_offset = memory_inlined ? 1 : 0;

    using Permanent_            = Matrix<char, "E", memory_permanent>;
    using Buffer_               = Matrix<char, "E", memory_buffer>;
    using TypeTuple             = std::tuple<Input_, Output_, Buffer_, Permanent_>;
    using LocalMatrixTypesTuple = std::tuple<TypeTuple, std::tuple_element_t<Indices - Current - 1, typename Next::LocalMatrixTypesTuple>...>;

    constexpr static auto        memory_minimal_tuple   = std::make_tuple(memory_minimal, std::get<Indices - Current - 1>(Next::memory_minimal_tuple)...);
    constexpr static auto        memory_buffer_tuple    = std::make_tuple(memory_buffer, std::get<Indices - Current - 1>(Next::memory_buffer_tuple)...);
    constexpr static auto        memory_permanent_tuple = std::make_tuple(memory_permanent, std::get<Indices - Current - 1>(Next::memory_permanent_tuple)...);
    constexpr static std::size_t total_memory_minimal   = vmax(std::get<0>(memory_minimal_tuple), std::get<Indices - Current>(memory_minimal_tuple)...);
    constexpr static std::size_t total_memory_buffer    = total_memory_minimal;
    constexpr static std::size_t total_memory_dynamic   = vmax(std::get<0>(memory_buffer_tuple), std::get<Indices - Current>(memory_buffer_tuple)...);
    constexpr static std::size_t total_memory_permanent = (std::get<0>(memory_permanent_tuple) + ... + std::get<Indices - Current>(memory_permanent_tuple));

    // Uses a ping pong memory scheme, where the output of the current layer is used as input for the next layer
    template <std::size_t MemorySize = 0, std::size_t Offset = 0, std::size_t PermanentOffset = 0>
    constexpr static MemoryLocation memory_index_current_locations = {.Input_index     = ((Current + Offset) % 2 == 0) ? 0 : MemorySize - sizeof(Input_),
                                                                      .Input_size      = sizeof(Input_),
                                                                      .Output_index    = ((Current + Offset + memory_inline_offset) % 2 == 0) ? MemorySize - sizeof(Output_) : 0,
                                                                      .Output_size     = sizeof(Output_),
                                                                      .buffer_index    = (memory_inlined)                ? std::max(sizeof(Input_), sizeof(Output_))
                                                                                         : ((Current + Offset) % 2 == 0) ? sizeof(Input_)
                                                                                                                         : sizeof(Output_),
                                                                      .buffer_size     = (memory_inlined) ? (signed)(MemorySize) - (signed)std::max(sizeof(Input_), sizeof(Output_))
                                                                                                          : std::max((signed)(MemorySize) - (signed)sizeof(Input_) - (signed)sizeof(Output_), 0),
                                                                      .permanent_index = PermanentOffset,
                                                                      .permanent_size  = memory_permanent};

    template <std::size_t MemorySize = 0, std::size_t Offset = 0, std::size_t PermanentOffset = 0>
    constexpr static std::array<MemoryLocation, sizeof...(Indices) + 1> memory_index_locations{
            memory_index_current_locations<MemorySize, Offset, PermanentOffset>,
            std::get<Indices - Current - 1>(Next::template memory_index_locations<MemorySize, Offset + memory_inline_offset, PermanentOffset + memory_permanent>)...};
};

template <IsValidLayer... Layers>
struct Sequence {
  public:
    static constexpr auto index_sequence = std::make_index_sequence<sizeof...(Layers)>();
    using LayerTypes                     = std::tuple<const std::remove_cvref_t<Layers>...>;

    const LayerTypes layers;

    constexpr static std::size_t layer_count = sizeof...(Layers);

    template <IsMatrixType InputMatrix>
    using CurrentMemoryPlaning = MemoryPlaning<InputMatrix, LayerTypes, std::make_index_sequence<sizeof...(Layers)>>;

    // Memory Requirements of forward pass
    template <IsMatrixType InputMatrix>
    static constexpr std::size_t memory_minimal = CurrentMemoryPlaning<InputMatrix>::total_memory_minimal;
    // Can it reuse the input Memory region?
    static constexpr bool memory_inlined = true; // Sequence layers can always reuse the input memory region, as they are designed to be used in a sequence and it is assumed that the input buffer is
                                                 // already used as input for the first layer
    // Required Buffer size
    // the amount of Memory required for temporary storage
    template <IsMatrixType InputMatrix>
    static constexpr std::size_t memory_buffer = CurrentMemoryPlaning<InputMatrix>::total_memory_buffer;
    // Dynamic increase if enough memory is available, might change behavior of the layer, like buffering reused weights in fast memory instead of slow storage
    template <IsMatrixType InputMatrix>
    static constexpr std::size_t memory_dynamic = CurrentMemoryPlaning<InputMatrix>::total_memory_dynamic;
    // Permanent Memory which is required for some layers, if used in a time series model
    template <IsMatrixType InputMatrix>
    static constexpr std::size_t memory_permanent = CurrentMemoryPlaning<InputMatrix>::total_memory_permanent;

    using ExampleInputMatrix = Matrix<char, "E", 0>;
    template <typename InputMatrix>
    using OutputMatrix = std::tuple_element_t<1, std::tuple_element_t<sizeof...(Layers) - 1, typename CurrentMemoryPlaning<InputMatrix>::LocalMatrixTypesTuple>>;

    // Constructor
    constexpr Sequence(Layers &&...layers) : layers(std::forward<Layers>(layers)...) {};

    template <IsMatrixType InputMatrixType, IsMatrixType BufferMatrixType = Matrix<char, "E", 0>>
    __attribute__((always_inline)) inline InputMatrixType *getInputMatrix(BufferMatrixType &buffer) const noexcept {
        static_assert(IsBaseMatrixType<BufferMatrixType>, "Buffer Matrix Type must be a Base Matrix Type");
        static_assert(sizeof(buffer.data) >= CurrentMemoryPlaning<InputMatrixType>::total_memory_buffer, "Buffer Memory Size does not match the required size");
        constexpr auto memory_index_location = std::get<0>(CurrentMemoryPlaning<InputMatrixType>::template memory_index_locations<sizeof(buffer.data), 0>); // Get the first Input
        constexpr auto input_index           = memory_index_location.Input_index;
        return reinterpret_cast<InputMatrixType *>(&buffer.data[input_index]);
    }

    template <IsMatrixType InputMatrixType, IsMatrixType BufferMatrixType = Matrix<char, "E", 0>>
    __attribute__((always_inline)) inline OutputMatrix<InputMatrixType> *getOutputMatrix(BufferMatrixType &buffer) const noexcept {
        static_assert(IsBaseMatrixType<BufferMatrixType>, "Buffer Matrix Type must be a Base Matrix Type");
        static_assert(sizeof(buffer.data) >= CurrentMemoryPlaning<InputMatrixType>::total_memory_buffer, "Buffer Memory Size does not match the required size");
        constexpr auto memory_index_location = std::get<sizeof...(Layers) - 1>(CurrentMemoryPlaning<InputMatrixType>::template memory_index_locations<sizeof(buffer.data), 0>); // Get the last Output
        constexpr auto output_index          = memory_index_location.Output_index;
        return reinterpret_cast<OutputMatrix<InputMatrixType> *>(&buffer.data[output_index]);
    }

    template <std::array<bool, sizeof...(Layers)> ContinueCalculation = makeFilledArray<bool, sizeof...(Layers)>(true),
              std::size_t                         At                  = 0,
              IsMatrixType                        InputMatrixType,
              IsMatrixType                        OutputMatrixType,
              IsMatrixType                        BufferMatrixType              = Matrix<char, "E", 0>,
              IsMatrixType                        PermanentMatrixType           = Matrix<char, "E", 0>,
              typename CurrentMemoryPlaning                                     = CurrentMemoryPlaning<InputMatrixType>,
              std::array<MemoryLocation, sizeof...(Layers)> MemoryIndexSchedule = CurrentMemoryPlaning::template memory_index_locations<sizeof(BufferMatrixType::data), 0, 0>,
              typename... ProfilingFunctional>
        requires(At < sizeof...(Layers) - 1) // Ensure at is within bounds
    __attribute__((always_inline)) inline void operator()(
            const InputMatrixType &Input, OutputMatrixType &Out, BufferMatrixType &buffer, PermanentMatrixType &permanent, ProfilingFunctional... profilingFnc) const noexcept {
        static_assert(sizeof(permanent.data) >= CurrentMemoryPlaning::total_memory_permanent, "Permanent Memory Size does not match the required size");
        static_assert(sizeof(buffer.data) >= CurrentMemoryPlaning::total_memory_buffer, "Buffer Memory Size does not match the required size");
        static_assert(IsBaseMatrixType<BufferMatrixType>, "Buffer Matrix Type must be a Base Matrix Type");
        static_assert(IsBaseMatrixType<PermanentMatrixType>, "Permanent Matrix Type must be a Base Matrix Type");

        using LocalMatrixTypes       = std::tuple_element_t<At, typename CurrentMemoryPlaning::LocalMatrixTypesTuple>;
        using InputMatrix_comparison = std::tuple_element_t<0, LocalMatrixTypes>;
        using NextInput              = std::tuple_element_t<1, LocalMatrixTypes>;
        using LayerBuffer            = std::tuple_element_t<2, LocalMatrixTypes>;
        using LayerPermanent         = std::tuple_element_t<3, LocalMatrixTypes>;

        static_assert(IsPermutationalSame<std::remove_cvref_t<InputMatrixType>, InputMatrix_comparison>, "Input Matrix Type does not match the expected type for this layer");

        constexpr bool        continue_after          = ContinueCalculation[At];
        constexpr auto        current_memory_schedule = std::get<At>(MemoryIndexSchedule);
        constexpr std::size_t output_index            = current_memory_schedule.Output_index;
        constexpr std::size_t output_size             = current_memory_schedule.Output_size;
        constexpr std::size_t buffer_index            = current_memory_schedule.buffer_index;
        constexpr std::size_t buffer_size             = current_memory_schedule.buffer_size;
        constexpr std::size_t permanent_index         = current_memory_schedule.permanent_index;
        constexpr std::size_t permanent_size          = current_memory_schedule.permanent_size;

        static_assert(output_index + output_size <= sizeof(buffer.data), "Output index and size exceed buffer size");
        static_assert(buffer_size + output_size <= sizeof(buffer.data), "Buffer and output size exceed buffer size");
        static_assert(permanent_index + permanent_size <= sizeof(permanent.data), "Permanent index and size exceed permanent size");

        char *const next_input_pointer = &(buffer.data[output_index]);       // Next Input Location
        char *const buffer_pointer     = &(buffer.data[buffer_index]);       // Buffer Location
        char *const permanent_pointer  = &(permanent.data[permanent_index]); // Permanent Memory Location

        auto *intermediate_output = reinterpret_cast<NextInput *>(next_input_pointer);     // Use the buffer data as the output matrix
        auto *layer_buffer        = reinterpret_cast<LayerBuffer *>(buffer_pointer);       // Use the buffer data as the layer buffer
        auto *layer_permanent     = reinterpret_cast<LayerPermanent *>(permanent_pointer); // Use the permanent data as the layer permanent memory

        // Call the current layer's operator
        std::get<At>(layers).template operator()<continue_after>(Input, *intermediate_output, *layer_buffer, *layer_permanent);

        if constexpr (continue_after) {
            operator()<ContinueCalculation, At + 1, NextInput, OutputMatrixType, BufferMatrixType, PermanentMatrixType, CurrentMemoryPlaning, MemoryIndexSchedule>(*intermediate_output, Out, buffer,
                                                                                                                                                                   permanent, profilingFnc...);
        } else {
            (profilingFnc(), ...);
        }
    }

    template <std::array<bool, sizeof...(Layers)> ContinueCalculation = makeFilledArray<bool, sizeof...(Layers)>(true),
              std::size_t                         At                  = 0,
              IsMatrixType                        InputMatrixType,
              IsMatrixType                        OutputMatrixType,
              IsMatrixType                        BufferMatrixType,
              IsMatrixType                        PermanentMatrixType,
              typename CurrentMemoryPlaning                                     = CurrentMemoryPlaning<InputMatrixType>,
              std::array<MemoryLocation, sizeof...(Layers)> MemoryIndexSchedule = CurrentMemoryPlaning::template memory_index_locations<sizeof(BufferMatrixType::data), 0, 0>,
              typename... ProfilingFunctional>
        requires(At >= sizeof...(Layers) - 1) // Ensure at is within bounds
    __attribute__((always_inline)) inline void operator()(
            const InputMatrixType &Input, OutputMatrixType &Out, BufferMatrixType &buffer, PermanentMatrixType &permanent, ProfilingFunctional... profilingFnc) const noexcept {
        static_assert(sizeof(permanent.data) >= CurrentMemoryPlaning::total_memory_permanent, "Permanent Memory Size does not match the required size");
        static_assert(sizeof(buffer.data) >= CurrentMemoryPlaning::total_memory_buffer, "Buffer Memory Size does not match the required size");
        static_assert(IsBaseMatrixType<BufferMatrixType>, "Buffer Matrix Type must be a Base Matrix Type");
        static_assert(IsBaseMatrixType<PermanentMatrixType>, "Permanent Matrix Type must be a Base Matrix Type");

        using LocalMatrixTypes        = std::tuple_element_t<At, typename CurrentMemoryPlaning::LocalMatrixTypesTuple>;
        using InputMatrix_comparison  = std::tuple_element_t<0, LocalMatrixTypes>;
        using OutputMatrix_comparison = std::tuple_element_t<1, LocalMatrixTypes>;
        using LayerBuffer             = std::tuple_element_t<2, LocalMatrixTypes>;
        using LayerPermanent          = std::tuple_element_t<3, LocalMatrixTypes>;

        static_assert(IsPermutationalSame<std::remove_cvref_t<InputMatrixType>, InputMatrix_comparison>, "Input Matrix Type does not match the expected type for this layer");
        static_assert(IsPermutationalSame<std::remove_cvref_t<OutputMatrixType>, OutputMatrix_comparison>, "Output Matrix Type does not match the expected type for this layer");

        constexpr bool        continue_after          = ContinueCalculation[At]; // Will be used to partially execute the current layer
        constexpr auto        current_memory_schedule = std::get<At>(MemoryIndexSchedule);
        constexpr std::size_t buffer_index            = current_memory_schedule.buffer_index;
        constexpr std::size_t permanent_index         = current_memory_schedule.permanent_index;
        constexpr std::size_t permanent_size          = current_memory_schedule.permanent_size;

        static_assert(permanent_index + permanent_size <= sizeof(permanent.data), "Permanent index and size exceed permanent size");

        char *const buffer_pointer    = &(buffer.data[buffer_index]);       // Buffer Location
        char *const permanent_pointer = &(permanent.data[permanent_index]); // Permanent Memory Location

        auto *layer_buffer    = reinterpret_cast<LayerBuffer *>(buffer_pointer);       // Use the buffer data as the layer buffer
        auto *layer_permanent = reinterpret_cast<LayerPermanent *>(permanent_pointer); // Use the permanent data as the layer permanent memory

        // Call the current layer's operator
        std::get<At>(layers).template operator()<continue_after>(Input, Out, *layer_buffer, *layer_permanent);

        // Call the profiling functions if provided
        (profilingFnc(), ...);
    }
};

static_assert(IsValidLayer<Sequence<BaseLayer>>, "Sequence must be a valid layer type");

}; // namespace layers