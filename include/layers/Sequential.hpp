#pragma once

#include <tuple>
#include <utility>

#include "../Matrix.hpp"
#include "../helpers/Algorithm.hpp"
#include "BaseLayer.hpp"

#include "Sequential_helpers/Memory.hpp"

#include "../helpers/c++17_helpers.hpp"

// For debuggingh
#if false
#define __enamble_memory_info true
#include "../helpers/PrintTuple.hpp"
#include "../helpers/TestHelpers.hpp"
#else
#define __enamble_memory_info false
#endif

namespace layers {
/*
===============================================================================================================================================
                                                        Sequential Helpers Declarations
===============================================================================================================================================
*/
template <typename Input, typename... Layers>
struct Sequential_Generate_out_type_helper;

template <typename Input, typename... Layers>
using Sequential_Generate_out_type = typename Sequential_Generate_out_type_helper<remove_cvref_t<Input>, remove_cvref_t<Layers>...>::type;

template <typename Indices, typename... Layers>
struct Sequential_class_forward_helper;

template <typename Input, typename... Layers>
__attribute__((always_inline)) inline auto Sequential_forward(const Input &input, std::tuple<Layers...> &layers) {
    return Sequential_class_forward_helper<std::index_sequence_for<Layers...>, Layers...>::template forward<Input>(input, layers);
}

template <typename... IOTuples, typename... Layers>
__attribute__((always_inline)) inline void Sequential_forward(const std::tuple<IOTuples...> &iob, std::tuple<Layers...> &layers) {
    Sequential_class_forward_helper<std::index_sequence_for<Layers...>, Layers...>::forward(iob, layers);
}

template <typename IOTypeTuple, typename... IOPositionTuples, typename InputType, typename OutputType, typename... Layers>
__attribute__((always_inline)) inline void Sequential_forward_index(
        const std::tuple<IOPositionTuples...> &IOPositionTupleTuple, char *Buffer, const InputType &Input, OutputType &Output, std::tuple<Layers...> &layers) {
    Sequential_class_forward_helper<std::index_sequence_for<Layers...>, Layers...>::template forward_index<IOTypeTuple>(IOPositionTupleTuple, Buffer, Input, Output, layers);
}

template <typename IOTypeTuple, typename MemoryScheduleSequence, typename InputType, typename OutputType, typename... Layers, typename... ProfilingType>
__attribute__((always_inline)) inline void Sequential_forward_index_sequence(
        char *const Buffer, const InputType &Input, OutputType &Output, const std::tuple<Layers...> &layers, ProfilingType... profiling) {
    Sequential_class_forward_helper<std::index_sequence_for<Layers...>, Layers...>::template forward_index_sequence<IOTypeTuple, MemoryScheduleSequence>(Buffer, Input, Output, layers, profiling...);
}

template <typename IOTypeTuple,
          typename MemoryScheduleSequence,
          typename PermanentMemoryScheduleSequence,
          typename ContinueCalculation,
          typename InputType,
          typename OutputType,
          typename... Layers,
          typename... ProfilingType>
__attribute__((always_inline)) inline void Sequential_forward_index_sequence(
        char *const Buffer, char *const PermanentMemory, const InputType &Input, OutputType &Output, const std::tuple<Layers...> &layers, ProfilingType... profiling) {
    Sequential_class_forward_helper<std::index_sequence_for<Layers...>, Layers...>::template forward_index_sequence<IOTypeTuple, MemoryScheduleSequence, PermanentMemoryScheduleSequence,
                                                                                                                    ContinueCalculation>(Buffer, PermanentMemory, Input, Output, layers, profiling...);
}

/*
===============================================================================================================================================
                                                        Sequential Class
===============================================================================================================================================
*/

template <typename... Layers>
class Sequential_class_hidden : public BaseLayer {
  private:
  public:
    const std::tuple<Layers...> layers;

  public:
    // Type information
    template <typename InputMatrix>
    using OutputMatrix = Sequential_Generate_out_type<InputMatrix, Layers...>;

    using BufferMatrix = Matrix<char, DimensionOrder::ERROR, 0>; // TODO: Actually define this, possible change to char[]

    // Internal Memory Information
    template <typename Input>
    using MemoryInputOutputPairsInformation = Sequential_class_MemoryInputOutputPairs<Input, Layers...>;
    template <typename Input, bool ignore_Input = false, bool ignore_output = false>
    static constexpr auto MemoryMinimalInformation = Sequential_class_MemoryMinimal_Resolver<Input, ignore_Input, ignore_output, Layers...>;
    template <typename Input>
    static constexpr auto MemoryOutTypeInformation = Sequential_class_MemoryOutType_Resolver<Input, Layers...>;
    static constexpr auto MemoryInlinedInformation = Sequential_class_MemoryInlined_Resolver<Layers...>;
    static constexpr auto MemoryDynamicInformation = Sequential_class_MemoryDynamic_Resolver<Layers...>;
    template <typename Input>
    static constexpr auto MemoryPermanentInformation = Sequential_class_MemoryPermanent_Resolver<Input, Layers...>;

    // Memory Requirements of forward pass
    template <typename InputMatrix>
    static constexpr size_t MemoryMinimal = helpers::max(MemoryMinimalInformation<InputMatrix, false, false>);
    // Can it reuse the input Memory region?
    static constexpr bool MemoryInlined = ((sizeof...(Layers) - helpers::sum(MemoryInlinedInformation)) % 2) == 0;
    // Dynamic increase if enough memory is available
    static constexpr size_t MemoryDynamic = helpers::max(MemoryDynamicInformation);
    // Memory Buffer
    template <typename InputMatrix>
    static constexpr size_t MemoryBuffer = MemoryMinimal<InputMatrix>;
    // Permanent Memory, Sequential layers require permanent memory if used in a time series model
    template <typename InputMatrix>
    static constexpr size_t MemoryPermanent = helpers::sum(MemoryPermanentInformation<InputMatrix>);

    // Constructor
    constexpr Sequential_class_hidden(const Layers &...layers) noexcept : layers(layers...) {};

    constexpr size_t number_of_layers() const noexcept { return sizeof...(Layers); }

    // Forward pass
    template <typename Input>
    __attribute__((always_inline)) inline auto operator()(const Input &input) const noexcept {
        // return Sequential_forward(input, layers);
        constexpr size_t BufferSize = MemoryMinimal<Input>;
        char             buffer[BufferSize];
        // const auto       schedule = Sequential_class_MemoryScheduler_Resolver<BufferSize, Input, false, Layers...>(buffer, &input);
        // Sequential_forward(schedule, layers);
        // return *std::get<1>(std::get<sizeof...(Layers) - 1>(schedule));

        using IOTypeInformation = MemoryInputOutputPairsInformation<Input>;
        // constexpr auto schedule_index = Sequential_class_MemoryScheduler_Resolver_only_Index<BufferSize, Input, Layers...>(-1);
        // constexpr size_t last_index = sizeof...(Layers) - 1;
        // using OutputType           = std::tuple_element_t<1, std::tuple_element_t<last_index, IOTypeInformation>>;
        // OutputType* out = reinterpret_cast<OutputType *>(&buffer[std::get<1>(std::get<last_index>(schedule_index))]);

        // Sequential_forward_index<IOTypeInformation>(schedule_index, buffer, input, *out, layers);

        using schedule_sequence     = Sequential_class_MemoryScheduler_Resolver_only_Index_sequence<BufferSize, Input, -1, 0, Layers...>;
        constexpr size_t last_index = sizeof...(Layers) - 1;
        using OutputType            = std::tuple_element_t<1, std::tuple_element_t<last_index, IOTypeInformation>>;
        OutputType *out             = reinterpret_cast<OutputType *>(&buffer[integer_at(std::tuple_element_t<last_index, schedule_sequence>{}, 1)]);
        // std::cout << "schedule_sequence: " << TypeName<schedule_sequence>  << "\n";
        Sequential_forward_index_sequence<IOTypeInformation, schedule_sequence>(buffer, input, *out, layers);

        return *out;
    }

    // // Forward pass
    // template <typename Input, size_t BufferSize>
    // __attribute__((always_inline)) inline auto operator()(const Input &input, Matrix<char, DimensionOrder::ERROR, BufferSize> &buffer) const noexcept {
    //     // return Sequential_forward(input, layers);
    //     // const auto schedule = Sequential_class_MemoryScheduler_Resolver<BufferSize, Input, false, Layers...>(&buffer.data[0], &input);
    //     // Sequential_forward(schedule, layers);
    //     // return *std::get<1>(std::get<sizeof...(Layers) - 1>(schedule));

    //     const char    *buffer_ptr     = &buffer.data[0];
    //     constexpr auto schedule_index = Sequential_class_MemoryScheduler_Resolver_only_Index<BufferSize, Input, Layers...>(0);
    //     using IOTypeInformation       = MemoryInputOutputPairsInformation<Input>;
    //     constexpr size_t last_index   = sizeof...(Layers) - 1;
    //     using OutputType              = std::tuple_element_t<1, std::tuple_element_t<last_index, IOTypeInformation>>;
    //     OutputType *out               = reinterpret_cast<OutputType *>(&buffer_ptr[std::get<1>(std::get<last_index>(schedule_index))]);

    //     Sequential_forward_index<IOTypeInformation>(schedule_index, buffer_ptr, input, *out, layers);
    //     return *out;
    // }

    template <typename Input, size_t BufferSize, typename... ProfilingType>
    __attribute__((always_inline)) inline void operator()(const Input                                     &input,
                                                          OutputMatrix<Input>                             &out,
                                                          Matrix<char, DimensionOrder::ERROR, BufferSize> &buffer,
                                                          ProfilingType... Profiler) const noexcept {
        // const auto schedule = Sequential_class_MemoryScheduler_Resolver<BufferSize, Input, OutputMatrix<Input>, false, Layers...>(&buffer.data[0], &input, &out);
        // Sequential_forward(schedule, layers);
        using IOTypeInformation = MemoryInputOutputPairsInformation<Input>;

        // constexpr auto schedule_index = Sequential_class_MemoryScheduler_Resolver_only_Index<BufferSize, Input, Layers...>(-1, -1);
        // Sequential_forward_index<IOTypeInformation>(schedule_index, &buffer.data[0], input, out, layers);
        // std::cout << "schedule_index: " << schedule_index  << "\n";

        using schedule_sequence = Sequential_class_MemoryScheduler_Resolver_only_Index_sequence<BufferSize, Input, -1, -1, Layers...>;
        // std::cout << "schedule_sequence: " << TypeName<schedule_sequence>  << "\n";
        Sequential_forward_index_sequence<IOTypeInformation, schedule_sequence>(&buffer.data[0], input, out, layers, Profiler...);
    }

    template <typename Input, size_t BufferSize, size_t PermanentMemorySize, typename... ProfilingType>
    __attribute__((always_inline)) inline void operator()(const Input                                              &input,
                                                          OutputMatrix<Input>                                      &out,
                                                          Matrix<char, DimensionOrder::ERROR, BufferSize>          &buffer,
                                                          Matrix<char, DimensionOrder::ERROR, PermanentMemorySize> &PermanentMemory,
                                                          ProfilingType... Profiler) const noexcept {
        static_assert(PermanentMemorySize == MemoryPermanent<Input>, "Permanent Memory Size does not match the required size");
        // const auto schedule = Sequential_class_MemoryScheduler_Resolver<BufferSize, Input, OutputMatrix<Input>, false, Layers...>(&buffer.data[0], &input, &out);
        // Sequential_forward(schedule, layers);
        using IOTypeInformation = MemoryInputOutputPairsInformation<Input>;

        // constexpr auto schedule_index = Sequential_class_MemoryScheduler_Resolver_only_Index<BufferSize, Input, Layers...>(-1, -1);
        // Sequential_forward_index<IOTypeInformation>(schedule_index, &buffer.data[0], input, out, layers);
        // std::cout << "schedule_index: " << schedule_index  << "\n";

        using MemoryScheduleSequence  = Sequential_class_MemoryScheduler_Resolver_only_Index_sequence<BufferSize, Input, -1, -1, Layers...>;
        using PermanentMemorySequence = Sequential_class_PermanentMemoryScheduler_Resolver_only_Index_sequence<PermanentMemorySize, Input, 0, Layers...>;
        // std::cout << "schedule_sequence: " << TypeName<schedule_sequence>  << "\n";
        Sequential_forward_index_sequence<IOTypeInformation, MemoryScheduleSequence, PermanentMemorySequence, SequenceRepeater<sizeof...(Layers),bool,true>>(&buffer.data[0], &PermanentMemory.data[0], input, out, layers, Profiler...);
    }

    template <typename ContinueCalculation, typename Input, size_t BufferSize, size_t PermanentMemorySize, typename... ProfilingType>
    __attribute__((always_inline)) inline void partialExecution(const Input                                              &input,
                                                          OutputMatrix<Input>                                      &out,
                                                          Matrix<char, DimensionOrder::ERROR, BufferSize>          &buffer,
                                                          Matrix<char, DimensionOrder::ERROR, PermanentMemorySize> &PermanentMemory,
                                                          ProfilingType... Profiler) const noexcept {
        static_assert(PermanentMemorySize == MemoryPermanent<Input>, "Permanent Memory Size does not match the required size");
        // const auto schedule = Sequential_class_MemoryScheduler_Resolver<BufferSize, Input, OutputMatrix<Input>, false, Layers...>(&buffer.data[0], &input, &out);
        // Sequential_forward(schedule, layers);
        using IOTypeInformation = MemoryInputOutputPairsInformation<Input>;

        // constexpr auto schedule_index = Sequential_class_MemoryScheduler_Resolver_only_Index<BufferSize, Input, Layers...>(-1, -1);
        // Sequential_forward_index<IOTypeInformation>(schedule_index, &buffer.data[0], input, out, layers);
        // std::cout << "schedule_index: " << schedule_index  << "\n";

        using MemoryScheduleSequence  = Sequential_class_MemoryScheduler_Resolver_only_Index_sequence<BufferSize, Input, -1, -1, Layers...>;
        using PermanentMemorySequence = Sequential_class_PermanentMemoryScheduler_Resolver_only_Index_sequence<PermanentMemorySize, Input, 0, Layers...>;
        // std::cout << "schedule_sequence: " << TypeName<schedule_sequence>  << "\n";
        Sequential_forward_index_sequence<IOTypeInformation, MemoryScheduleSequence, PermanentMemorySequence, ContinueCalculation>(&buffer.data[0], &PermanentMemory.data[0], input, out, layers, Profiler...);
    }

#if __enamble_memory_info
    template <typename Input>
    void memory_information(const Input &input) const noexcept {
        std::cout << "Memory Minimal Information: " << MemoryMinimalInformation<Input, false, false> << "\n";
        std::cout << "Memory OutType Information: " << MemoryOutTypeInformation<Input> << "\n";
        std::cout << "Memory Inlined Information: " << MemoryInlinedInformation << "\n";
        std::cout << "Memory Dynamic Information: " << MemoryDynamicInformation << "\n";

        std::cout << "Memory Minimal: " << MemoryMinimal<Input> << "\n";
        std::cout << "Memory Inlined: " << MemoryInlined << "\n";
        std::cout << "Memory Dynamic: " << MemoryDynamic << "\n";
        std::cout << TypeName<MemoryInputOutputPairsInformation<Input>> << "\n";

        const size_t Buffersize = MemoryMinimal<Input>; // + MemoryDynamic; //+1000;
        char         buffer[Buffersize];
        const auto   schedule = Sequential_class_MemoryScheduler_Resolver<Buffersize, Input, false, Layers...>(buffer, &input);
        std::cout << "Memory Scheduler: " << schedule << "\n";
        std::cout << "Memory Scheduler: " << TypeName<decltype(schedule)> << "\n";
        print_Memory_Location_loop(schedule, buffer, Buffersize, Buffersize / 160, std::make_index_sequence<sizeof...(Layers)>());
    }

    template <typename Input, size_t BufferSize>
    __attribute__((always_inline)) inline void memory_information(const Input &input, OutputMatrix<Input> &out, Matrix<char, DimensionOrder::ERROR, BufferSize> &buffer) const noexcept {
        std::cout << "Memory Minimal Information: " << MemoryMinimalInformation<Input, false, false> << "\n";
        std::cout << "Memory OutType Information: " << MemoryOutTypeInformation<Input> << "\n";
        std::cout << "Memory Inlined Information: " << MemoryInlinedInformation << "\n";
        std::cout << "Memory Dynamic Information: " << MemoryDynamicInformation << "\n";

        std::cout << "Memory Minimal: " << MemoryMinimal<Input> << "\n";
        std::cout << "Memory Inlined: " << MemoryInlined << "\n";
        std::cout << "Memory Dynamic: " << MemoryDynamic << "\n";
        std::cout << TypeName<MemoryInputOutputPairsInformation<Input>> << "\n";

        const auto schedule = Sequential_class_MemoryScheduler_Resolver<BufferSize, Input, OutputMatrix<Input>, false, Layers...>(&buffer.data[0], &input, &out);

        std::cout << "Memory Scheduler: " << schedule << "\n";
        std::cout << "Memory Scheduler: " << TypeName<decltype(schedule)> << "\n";
        print_Memory_Location_loop(schedule, buffer.data, BufferSize, BufferSize / 160, std::make_index_sequence<sizeof...(Layers)>());
    }
#endif
};

template <typename... Layers>
constexpr auto Sequential(const Layers &...layers) noexcept {
    return Sequential_class_hidden<Layers...>(layers...);
}

/*
===============================================================================================================================================
                                                        Sequential Helpers Definitions
===============================================================================================================================================
*/

template <std::size_t Index, std::size_t... Indices, typename... Layers>
struct Sequential_class_forward_helper<std::index_sequence<Index, Indices...>, Layers...> {
    // template <typename Input>
    // __attribute__((always_inline)) static inline auto forward(const Input &input, std::tuple<Layers...> &layers) {
    //     const auto output = std::get<Index>(layers)(input);
    //     return Sequential_class_forward_helper<std::index_sequence<Indices...>, Layers...>::template forward<remove_cvref_t<decltype(output)>>(output, layers);
    // }

    // template <typename... IOTuples>
    // __attribute__((always_inline)) static inline void forward(const std::tuple<IOTuples...> &iob, std::tuple<Layers...> &layers) {
    //     const auto tup = std::get<Index>(iob);
    //     std::get<Index>(layers)(*std::get<0>(tup), *std::get<1>(tup));
    //     // std::get<Index>(layers)(std::get<0>(tup), std::get<1>(tup), std::get<2>(tup);    // once buffer is implemented
    //     Sequential_class_forward_helper<std::index_sequence<Indices...>, Layers...>::forward(iob, layers);
    // }

    // template <typename IOTypeTuples, typename... IOPositionTuples, typename InputType, typename FinalOutputType>
    // __attribute__((always_inline)) static inline void forward_index(
    //         const std::tuple<IOPositionTuples...> &IOPositionTupleTuple, char *Buffer, const InputType &Input, FinalOutputType &FinalOutput, std::tuple<Layers...> &layers) {
    //     const auto tup          = std::get<Index>(IOPositionTupleTuple);
    //     using IOTypeInformation = std::tuple_element_t<Index, IOTypeTuples>;
    //     // using InputType              = typename std::tuple_element_t<0, IOTypeInformation>;
    //     // const int InputPosition  = std::get<0>(tup);
    //     using OutputType           = typename std::tuple_element_t<1, IOTypeInformation>;
    //     const int   OutputPosition = std::get<1>(tup);
    //     OutputType *Output         = reinterpret_cast<OutputType *>(&Buffer[OutputPosition]);

    //     std::get<Index>(layers)(Input, *Output);

    //     // std::get<Index>(layers)(std::get<0>(tup), std::get<1>(tup), std::get<2>(tup);    // once buffer is implemented
    //     Sequential_class_forward_helper<std::index_sequence<Indices...>, Layers...>::template forward_index<IOTypeTuples>(IOPositionTupleTuple, Buffer, *Output, FinalOutput, layers);
    // }

    template <typename IOTypeTuples, typename MemoryScheduleSequence, typename InputType, typename FinalOutputType, typename... ProfilingType>
    __attribute__((always_inline)) static inline void forward_index_sequence(
            char const *Buffer, const InputType &Input, FinalOutputType &FinalOutput, const std::tuple<Layers...> &layers, ProfilingType... profiling) {
        using sequ                                            = std::tuple_element_t<Index, MemoryScheduleSequence>;
        using IOTypeInformation                               = std::tuple_element_t<Index, IOTypeTuples>;
        using OutputType                                      = typename std::tuple_element_t<1, IOTypeInformation>;
        constexpr int                          OutputPosition = integer_at(sequ{}, 1);
        OutputType                            *Output         = reinterpret_cast<OutputType *>(&Buffer[OutputPosition]);
        Matrix<char, DimensionOrder::ERROR, 0> buffer{};

        ((profiling()), ...);

        std::get<Index>(layers)(Input, *Output, buffer);

        // std::get<Index>(layers)(std::get<0>(tup), std::get<1>(tup), std::get<2>(tup);    // once buffer is implemented
        Sequential_class_forward_helper<std::index_sequence<Indices...>, Layers...>::template forward_index_sequence<IOTypeTuples, MemoryScheduleSequence>(Buffer, *Output, FinalOutput, layers,
                                                                                                                                                           profiling...);
    }

    template <typename IOTypeTuples,
              typename MemoryScheduleSequence,
              typename PermanentMemoryScheduleSequence,
              typename ContinueCalculation,
              typename InputType,
              typename FinalOutputType,
              typename... ProfilingType>
    __attribute__((always_inline)) static inline void forward_index_sequence(
            char *const Buffer, char *const PermanentMemory, const InputType &Input, FinalOutputType &FinalOutput, const std::tuple<Layers...> &layers, ProfilingType... profiling) {
        using sequ                                            = std::tuple_element_t<Index, MemoryScheduleSequence>;
        using IOTypeInformation                               = std::tuple_element_t<Index, IOTypeTuples>;
        using OutputType                                      = typename std::tuple_element_t<1, IOTypeInformation>;
        constexpr size_t                       OutputPosition = integer_at(sequ{}, 1);
        OutputType *const                      Output         = reinterpret_cast<OutputType *const>(&Buffer[OutputPosition]);
        Matrix<char, DimensionOrder::ERROR, 0> buffer{};

        using permanentMemory_start_size          = std::tuple_element_t<Index, PermanentMemoryScheduleSequence>;
        constexpr size_t permanentMemory_start    = integer_at(permanentMemory_start_size{}, 0);
        constexpr size_t permanentMemory_size     = integer_at(permanentMemory_start_size{}, 1);
        constexpr bool   ContinueCalculation_here = integer_at(ContinueCalculation{}, Index);

        Matrix<char, DimensionOrder::ERROR, permanentMemory_size> *const permanentMemory =
                reinterpret_cast<Matrix<char, DimensionOrder::ERROR, permanentMemory_size> *const>(&PermanentMemory[permanentMemory_start]);

        ((profiling()), ...);

        std::get<Index>(layers)(Input, *Output, std::integer_sequence<bool, ContinueCalculation_here>(), buffer, *permanentMemory);
        // std::get<Index>(layers)(Input, *Output, std::integer_sequence<bool, true>(), buffer, *permanentMemory);

        // std::get<Index>(layers)(std::get<0>(tup), std::get<1>(tup), std::get<2>(tup);    // once buffer is implemented
        if constexpr (ContinueCalculation_here)
            Sequential_class_forward_helper<std::index_sequence<Indices...>, Layers...>::template forward_index_sequence<IOTypeTuples, MemoryScheduleSequence, PermanentMemoryScheduleSequence,
                                                                                                                         ContinueCalculation>(Buffer, PermanentMemory, *Output, FinalOutput, layers,
                                                                                                                                              profiling...);
    }
};

template <std::size_t Index, typename... Layers>
struct Sequential_class_forward_helper<std::index_sequence<Index>, Layers...> {
    // template <typename Input>
    // __attribute__((always_inline)) static inline auto forward(const Input &input, std::tuple<Layers...> &layers) {
    //     const auto output = std::get<Index>(layers)(input);
    //     return output;
    // }

    // template <typename... IOTuples>
    // __attribute__((always_inline)) static inline void forward(const std::tuple<IOTuples...> &iob, std::tuple<Layers...> &layers) {
    //     const auto tup = std::get<Index>(iob);
    //     std::get<Index>(layers)(*std::get<0>(tup), *std::get<1>(tup));
    //     // const auto output = std::get<Index>(layers)(std::get<0>(tup), std::get<1>(tup), std::get<2>(tup);    // once buffer is implemented
    // }

    // template <typename IOTypeTuples, typename... IOPositionTuples, typename InputType, typename FinalOutputType>
    // __attribute__((always_inline)) static inline void forward_index(
    //         const std::tuple<IOPositionTuples...> &IOPositionTupleTuple, char *Buffer, const InputType &Input, FinalOutputType &FinalOutput, std::tuple<Layers...> &layers) {
    //     std::get<Index>(layers)(Input, FinalOutput);
    // }

    template <typename IOTypeTuples, typename MemoryScheduleSequence, typename InputType, typename FinalOutputType, typename... ProfilingType>
    __attribute__((always_inline)) static inline void forward_index_sequence(
            char *Buffer, const InputType &Input, FinalOutputType &FinalOutput, const std::tuple<Layers...> &layers, ProfilingType... profiling) {
        Matrix<char, DimensionOrder::ERROR, 0> buffer{};

        ((profiling()), ...);
        std::get<Index>(layers)(Input, FinalOutput, buffer);
    }

    template <typename IOTypeTuples,
              typename MemoryScheduleSequence,
              typename PermanentMemoryScheduleSequence,
              typename ContinueCalculation,
              typename InputType,
              typename FinalOutputType,
              typename... ProfilingType>
    __attribute__((always_inline)) static inline void forward_index_sequence(
            char *const Buffer, char *const PermanentMemory, const InputType &Input, FinalOutputType &FinalOutput, const std::tuple<Layers...> &layers, ProfilingType... profiling) {
        Matrix<char, DimensionOrder::ERROR, 0> buffer{};

        using permanentMemory_start_size          = std::tuple_element_t<Index, PermanentMemoryScheduleSequence>;
        constexpr size_t permanentMemory_start    = integer_at(permanentMemory_start_size{}, 0);
        constexpr size_t permanentMemory_size     = integer_at(permanentMemory_start_size{}, 1);
        constexpr bool   ContinueCalculation_here = integer_at(ContinueCalculation{}, Index);

        Matrix<char, DimensionOrder::ERROR, permanentMemory_size> *const permanentMemory =
                reinterpret_cast<Matrix<char, DimensionOrder::ERROR, permanentMemory_size> *const>(&PermanentMemory[permanentMemory_start]);

        ((profiling()), ...);

        std::get<Index>(layers)(Input, FinalOutput, std::integer_sequence<bool, ContinueCalculation_here>(), buffer, *permanentMemory);

    }
};

template <typename Input, typename Layer, typename... Layers>
struct Sequential_Generate_out_type_helper<Input, Layer, Layers...> {
    using type = typename Sequential_Generate_out_type_helper<typename Layer::template OutputMatrix<Input>, Layers...>::type;
};

template <typename Input, typename Layer>
struct Sequential_Generate_out_type_helper<Input, Layer> {
    using type = typename Layer::template OutputMatrix<Input>;
};
}; // namespace layers
