#pragma once
#include <cstdint>
#include <tuple>
#include <type_traits>

#include "../../Matrix.hpp"
#include "../../helpers/TestHelpers.hpp"

template <typename Indices, typename... Layers>
struct Sequential_class_MemoryResolver_helper;

template <typename Input, bool ignore_Input, bool ignore_output, typename... Layers>
constexpr auto Sequential_class_MemoryMinimal_Resolver =
        Sequential_class_MemoryResolver_helper<std::index_sequence_for<Layers...>, remove_cvref_t<Layers>...>::template Gather_MemoryMinimal<remove_cvref_t<Input>, ignore_Input, ignore_output>(
                std::tuple(), std::make_index_sequence<0>());

template <typename... Layers>
constexpr auto Sequential_class_MemoryInlined_Resolver =
        Sequential_class_MemoryResolver_helper<std::index_sequence_for<Layers...>, remove_cvref_t<Layers>...>::Gather_MemoryInlined(std::tuple(), std::make_index_sequence<0>());

template <typename... Layers>
constexpr auto Sequential_class_MemoryDynamic_Resolver =
        Sequential_class_MemoryResolver_helper<std::index_sequence_for<Layers...>, remove_cvref_t<Layers>...>::Gather_MemoryDynamic(std::tuple(), std::make_index_sequence<0>());

template <typename Input, typename... Layers>
constexpr auto Sequential_class_MemoryOutType_Resolver =
        Sequential_class_MemoryResolver_helper<std::index_sequence_for<Layers...>, remove_cvref_t<Layers>...>::template Gather_MemoryOutType<remove_cvref_t<Input>>(std::tuple(),
                                                                                                                                                                    std::make_index_sequence<0>());
template <typename Input, typename... Layers>
constexpr auto Sequential_class_MemoryPermanent_Resolver =
        Sequential_class_MemoryResolver_helper<std::index_sequence_for<Layers...>, remove_cvref_t<Layers>...>::template Gather_MemoryPermanent<remove_cvref_t<Input>>(std::tuple(),
                                                                                                                                                                      std::make_index_sequence<0>());

template <size_t Buffersize, typename Input, bool front = true, typename... Layers>
constexpr auto Sequential_class_MemoryScheduler_Resolver(char buffer[Buffersize], const Input *const input) noexcept {
    return Sequential_class_MemoryResolver_helper<std::index_sequence_for<Layers...>, remove_cvref_t<Layers>...>::template Gather_MemoryScheduler<Buffersize, front>(buffer, input, std::tuple(),
                                                                                                                                                                     std::make_index_sequence<0>());
}

template <size_t Buffersize, typename Input, typename Output, bool front = true, typename... Layers>
constexpr auto Sequential_class_MemoryScheduler_Resolver(char buffer[Buffersize], const Input *const input, Output *const output) noexcept {
    return Sequential_class_MemoryResolver_helper<std::index_sequence_for<Layers...>, remove_cvref_t<Layers>...>::template Gather_MemoryScheduler<Buffersize, front>(
            buffer, input, output, std::tuple(), std::make_index_sequence<0>());
}

template <size_t Buffersize, typename Input, typename... Layers>
constexpr auto Sequential_class_MemoryScheduler_Resolver_only_Index(int input_position = 0, int output_position = 0) noexcept {
    return Sequential_class_MemoryResolver_helper<std::index_sequence_for<Layers...>, remove_cvref_t<Layers>...>::template Gather_MemoryScheduler_only_index<Buffersize, Input>(
            input_position, output_position, std::tuple(), std::make_index_sequence<0>());
}

template <size_t Buffersize, typename Input, int input_position = 0, int output_position = 0, typename... Layers>
using Sequential_class_MemoryScheduler_Resolver_only_Index_sequence =
        decltype(Sequential_class_MemoryResolver_helper<std::index_sequence_for<Layers...>, remove_cvref_t<Layers>...>::
                         template Gather_MemoryScheduler_only_index_sequence<Buffersize, Input, input_position, output_position>(std::tuple(), std::make_index_sequence<0>()));

template <size_t PermanentMemoryTotalSize, typename Input, int start_position = 0, typename... Layers>
using Sequential_class_PermanentMemoryScheduler_Resolver_only_Index_sequence =
        decltype(Sequential_class_MemoryResolver_helper<std::index_sequence_for<Layers...>, remove_cvref_t<Layers>...>::
                         template Gather_PermanentMemoryScheduler_only_index_sequence<PermanentMemoryTotalSize, Input, start_position>(std::tuple(), std::make_index_sequence<0>()));

template <typename Input, typename... Layers>
using Sequential_class_MemoryInputOutputPairs =
        typename Sequential_class_MemoryResolver_helper<std::index_sequence_for<Layers...>, remove_cvref_t<Layers>...>::template InputOutputPairs<remove_cvref_t<Input>, std::tuple<>>;

template < // Memory Resolver
        std::size_t Index,
        std::size_t... Indices,
        typename... Layers>
struct Sequential_class_MemoryResolver_helper<std::index_sequence<Index, Indices...>, Layers...> {
    using CurrentLayer = std::tuple_element_t<Index, std::tuple<Layers...>>;
    template <typename Input>
    using Output = typename CurrentLayer::template OutputMatrix<Input>;
    template <typename Input>
    static constexpr size_t MemoryBuffer = CurrentLayer::template MemoryBuffer<Input>;
    // using NextLayers   = Sequential_class_MemoryResolver_helper<std::index_sequence<Indices...>, Layers...>::CurrentLayer;
    static constexpr bool   MemoryInlined = CurrentLayer::MemoryInlined;
    static constexpr size_t MemoryDynamic = CurrentLayer::MemoryDynamic;
    template <typename Input>
    static constexpr size_t MemoryPermanent = CurrentLayer::template MemoryPermanent<Input>;

    template <typename Input>
    using InputOutputPair = std::tuple<Input, Output<Input>>;

    template <typename Input, typename Tuple, size_t... indexes>
    using InputOutputPairs = typename Sequential_class_MemoryResolver_helper<std::index_sequence<Indices...>, Layers...>::
            template InputOutputPairs<Output<Input>, std::tuple<std::tuple_element_t<indexes, Tuple>..., InputOutputPair<Input>>, indexes..., sizeof...(indexes)>;

    template <typename Input, bool ignore_Input, bool ignore_output, typename MMinimal, std::size_t... otherindexes>
    constexpr static auto Gather_MemoryMinimal(MMinimal mminimal, std::index_sequence<otherindexes...>) noexcept {
        return Sequential_class_MemoryResolver_helper<std::index_sequence<Indices...>, Layers...>::template Gather_MemoryMinimal<Output<Input>, false, ignore_output>(
                std::tuple(std::get<otherindexes>(mminimal)..., MemoryBuffer<Input> + sizeof(Output<Input>) + ((ignore_Input) ? 0 : sizeof(Input))),
                std::make_index_sequence<sizeof...(otherindexes) + 1>());
    }

    template <typename Input, typename MOutType, std::size_t... otherindexes>
    constexpr static auto Gather_MemoryOutType(MOutType mouttype, std::index_sequence<otherindexes...>) noexcept {
        return Sequential_class_MemoryResolver_helper<std::index_sequence<Indices...>, Layers...>::template Gather_MemoryOutType<Output<Input>>(
                std::tuple(std::get<otherindexes>(mouttype)..., TypeName<typename Output<Input>::type>), std::make_index_sequence<sizeof...(otherindexes) + 1>());
    }

    template <typename Input, typename MPermType, std::size_t... otherindexes>
    constexpr static auto Gather_MemoryPermanent(MPermType mperm, std::index_sequence<otherindexes...>) noexcept {
        return Sequential_class_MemoryResolver_helper<std::index_sequence<Indices...>, Layers...>::template Gather_MemoryPermanent<Output<Input>>(
                std::tuple(std::get<otherindexes>(mperm)..., MemoryPermanent<Input>), std::make_index_sequence<sizeof...(otherindexes) + 1>());
    }

    template <typename MInlined, std::size_t... otherindexes>
    constexpr static auto Gather_MemoryInlined(MInlined minlined, std::index_sequence<otherindexes...>) noexcept {
        return Sequential_class_MemoryResolver_helper<std::index_sequence<Indices...>, Layers...>::Gather_MemoryInlined(std::tuple(std::get<otherindexes>(minlined)..., MemoryInlined),
                                                                                                                        std::make_index_sequence<sizeof...(otherindexes) + 1>());
    }

    template <typename MDynamic, std::size_t... otherindexes>
    constexpr static auto Gather_MemoryDynamic(MDynamic mdynamic, std::index_sequence<otherindexes...>) noexcept {
        return Sequential_class_MemoryResolver_helper<std::index_sequence<Indices...>, Layers...>::Gather_MemoryDynamic(std::tuple(std::get<otherindexes>(mdynamic)..., MemoryDynamic),
                                                                                                                        std::make_index_sequence<sizeof...(otherindexes) + 1>());
    }

    template <size_t Buffersize, typename Input, typename MemoryTupleTuple, std::size_t... otherindexes>
    constexpr static auto Gather_MemoryScheduler_only_index(const int input, const int final_output, const MemoryTupleTuple &mtupletuple, const std::index_sequence<otherindexes...> &) noexcept {
        const int output = (input == 0) ? Buffersize - sizeof(Output<Input>) : 0;

        const int rest_Memory = Buffersize - ((input != -1) ? sizeof(Input) : 0) - sizeof(Output<Input>);

        const int dynamic_start = (input == 0) ? sizeof(Input) : sizeof(Output<Input>);

        const int dynamic_stop = dynamic_start + ((MemoryDynamic > rest_Memory) ? 0 : MemoryDynamic);

        return Sequential_class_MemoryResolver_helper<std::index_sequence<Indices...>, Layers...>::template Gather_MemoryScheduler_only_index< // Template Parameter
                Buffersize,                                                                                                                    // Buffer Size
                Output<Input>                                                                                                                  // Input of the next Layer
                >                                                                                                                              // Template Parameters end
                (output,                                                                                                                       // Final Output
                 final_output,                                                                                                                 // Next Input
                 std::make_tuple(                                                      // Tuple<*Input [Matrix], *Output [Matrix], *Dynamic [Matrix<char,n>]>
                         std::get<otherindexes>(mtupletuple)...,                       // So far
                         std::make_tuple(input, output, dynamic_start, dynamic_stop)), // Current
                 std::make_index_sequence<sizeof...(otherindexes) + 1>());
    }

    template <size_t Buffersize,
              typename Input,
              int input, // input position
              int final_output,
              int output        = (input == 0) ? Buffersize - sizeof(Output<Input>) : 0,
              int rest_Memory   = Buffersize - ((input != -1) ? sizeof(Input) : 0) - sizeof(Output<Input>),
              int dynamic_start = (input == 0) ? sizeof(Input) : sizeof(Output<Input>),
              int dynamic_stop  = dynamic_start + ((MemoryDynamic > rest_Memory) ? 0 : MemoryDynamic),
              typename MemoryTupleTuple,
              std::size_t... otherindexes>
    constexpr static auto Gather_MemoryScheduler_only_index_sequence(const MemoryTupleTuple &mtupletuple, const std::index_sequence<otherindexes...> &) noexcept {
        return Sequential_class_MemoryResolver_helper<std::index_sequence<Indices...>, Layers...>::template Gather_MemoryScheduler_only_index_sequence< // Template Parameter
                Buffersize,                                                                                                                             // Buffer Size
                Output<Input>,                                                                                                                          // Input of the next Layer
                output,                                                                                                                                 // next Input Position
                final_output>                                                                                                                           // Template Parameters end
                (std::make_tuple(                                                                   // Tuple<*Input [Matrix], *Output [Matrix], *Dynamic [Matrix<char,n>]>
                         std::get<otherindexes>(mtupletuple)...,                                    // So far
                         std::integer_sequence<int, input, output, dynamic_start, dynamic_stop>{}), // Current
                 std::make_index_sequence<sizeof...(otherindexes) + 1>());
    }

    template <size_t Buffersize, bool front, typename Input, typename MemoryTupleTuple, std::size_t... otherindexes>
    constexpr static auto Gather_MemoryScheduler(char buffer[Buffersize], const Input *const input, MemoryTupleTuple mtupletuple, std::index_sequence<otherindexes...>) noexcept {
        Output<Input> *const nextInput = reinterpret_cast<Output<Input> *>(&buffer[(front != MemoryInlined) ? 0 : Buffersize - sizeof(Output<Input>)]);

        const size_t rest_Memory = Buffersize - sizeof(Input) - sizeof(Output<Input>);

        Matrix<char, DimensionOrder::ERROR, (MemoryDynamic > rest_Memory) ? 0 : MemoryDynamic> *const dynamic =
                reinterpret_cast<Matrix<char, DimensionOrder::ERROR, (MemoryDynamic > rest_Memory) ? 0 : MemoryDynamic> *>(&buffer[(front != MemoryInlined) ? sizeof(Output<Input>) : sizeof(Input)]);

        return Sequential_class_MemoryResolver_helper<std::index_sequence<Indices...>, Layers...>::template Gather_MemoryScheduler< // Template Parameter
                Buffersize,                                                                                                         // Buffer Size
                !(front != MemoryInlined)                                                                                           // Flipping Front
                >(buffer,                                                                                                           // Buffer
                  nextInput,                                                                                                        // Next Input
                  std::make_tuple(                                     // Tuple<*Input [Matrix], *Output [Matrix], *Dynamic [Matrix<char,n>]>
                          std::get<otherindexes>(mtupletuple)...,      // So far
                          std::make_tuple(input, nextInput, dynamic)), // Current
                  std::make_index_sequence<sizeof...(otherindexes) + 1>());
    }

    template <size_t Buffersize, bool front, typename Input, typename FinalOutput, typename MemoryTupleTuple, std::size_t... otherindexes>
    constexpr static auto Gather_MemoryScheduler(
            char buffer[Buffersize], const Input *const input, FinalOutput *const output, MemoryTupleTuple mtupletuple, std::index_sequence<otherindexes...>) noexcept {
        Output<Input> *const nextInput = reinterpret_cast<Output<Input> *>(&buffer[(front != MemoryInlined) ? 0 : Buffersize - sizeof(Output<Input>)]);

        const size_t rest_Memory = Buffersize - sizeof(Input) - sizeof(Output<Input>);

        Matrix<char, DimensionOrder::ERROR, (MemoryDynamic > rest_Memory) ? 0 : MemoryDynamic> *const dynamic =
                reinterpret_cast<Matrix<char, DimensionOrder::ERROR, (MemoryDynamic > rest_Memory) ? 0 : MemoryDynamic> *>(&buffer[(front != MemoryInlined) ? sizeof(Output<Input>) : sizeof(Input)]);

        return Sequential_class_MemoryResolver_helper<std::index_sequence<Indices...>, Layers...>::template Gather_MemoryScheduler< // Template Parameter
                Buffersize,                                                                                                         // Buffer Size
                !(front != MemoryInlined)                                                                                           // Flipping Front
                >(buffer,                                                                                                           // Buffer
                  nextInput,                                                                                                        // Next Input
                  output,                                                                                                           // Final Output
                  std::make_tuple(                                     // Tuple<*Input [Matrix], *Output [Matrix], *Dynamic [Matrix<char,n>]>
                          std::get<otherindexes>(mtupletuple)...,      // So far
                          std::make_tuple(input, nextInput, dynamic)), // Current
                  std::make_index_sequence<sizeof...(otherindexes) + 1>());
    }

    template <size_t PermanentMemoryTotalSize,
              typename Input,
              int PermanentMemoryPos, // input position
              typename PermanentMemoryTupleTuple,
              std::size_t... otherindexes>
    constexpr static auto Gather_PermanentMemoryScheduler_only_index_sequence(const PermanentMemoryTupleTuple &pmtupletuple, const std::index_sequence<otherindexes...> &) noexcept {
        static_assert(PermanentMemoryPos + MemoryPermanent<Input> <= PermanentMemoryTotalSize, "Permanent Memory is larger than assigned Memory Size");
        return Sequential_class_MemoryResolver_helper<std::index_sequence<Indices...>, Layers...>::template Gather_PermanentMemoryScheduler_only_index_sequence< // Template Parameter
                PermanentMemoryTotalSize,                                                                                                               // Buffer Size
                Output<Input>,                                                                                                                          // Input of the next Layer
                PermanentMemoryPos + MemoryPermanent<Input>>                                                                                            // Next Permanent Memory Start Position
                (std::make_tuple(                                                                   // Tuple<Permanent Index start, Permanent Memory Size>
                         std::get<otherindexes>(pmtupletuple)...,                                    // So far
                         std::integer_sequence<int, PermanentMemoryPos, MemoryPermanent<Input>>{}), // Current
                 std::make_index_sequence<sizeof...(otherindexes) + 1>());
    }
};

template <std::size_t Index, typename... Layers>
struct Sequential_class_MemoryResolver_helper<std::index_sequence<Index>, Layers...> {
    using CurrentLayer = std::tuple_element_t<Index, std::tuple<Layers...>>;
    template <typename Input>
    using Output = typename CurrentLayer::template OutputMatrix<Input>;
    template <typename Input>
    static constexpr size_t MemoryBuffer = CurrentLayer::template MemoryBuffer<Input>;

    static constexpr bool   MemoryInlined = CurrentLayer::MemoryInlined;
    static constexpr size_t MemoryDynamic = CurrentLayer::MemoryDynamic;
    template <typename Input>
    static constexpr size_t MemoryPermanent = CurrentLayer::template MemoryPermanent<Input>;

    template <typename Input>
    using InputOutputPair = std::tuple<Input, Output<Input>>;

    template <typename Input, typename Tuple, size_t... indexes>
    using InputOutputPairs = std::tuple<std::tuple_element_t<indexes, Tuple>..., InputOutputPair<Input>>;

    template <typename Input, bool ignore_Input, bool ignore_output, typename MMinimal, std::size_t... otherindexes>
    constexpr static auto Gather_MemoryMinimal(MMinimal mminimal, std::index_sequence<otherindexes...>) noexcept {
        return std::tuple(std::get<otherindexes>(mminimal)..., MemoryBuffer<Input> + ((ignore_output) ? 0 : sizeof(Output<Input>)) + ((ignore_Input) ? 0 : sizeof(Input)));
    }

    template <typename Input, typename MOutType, std::size_t... otherindexes>
    constexpr static auto Gather_MemoryOutType(MOutType mouttype, std::index_sequence<otherindexes...>) noexcept {
        return std::tuple(std::get<otherindexes>(mouttype)..., TypeName<typename Output<Input>::type>);
    }

    template <typename MInlined, std::size_t... otherindexes>
    constexpr static auto Gather_MemoryInlined(MInlined minlined, std::index_sequence<otherindexes...>) noexcept {
        return std::tuple(std::get<otherindexes>(minlined)..., MemoryInlined);
    }

    template <typename MDynamic, std::size_t... otherindexes>
    constexpr static auto Gather_MemoryDynamic(MDynamic mdynamic, std::index_sequence<otherindexes...>) noexcept {
        return std::tuple(std::get<otherindexes>(mdynamic)..., MemoryDynamic);
    }

    template <typename Input, typename MPermType, std::size_t... otherindexes>
    constexpr static auto Gather_MemoryPermanent(MPermType mperm, std::index_sequence<otherindexes...>) noexcept {
        return std::tuple(std::get<otherindexes>(mperm)..., MemoryPermanent<Input>);
    }

    template <size_t Buffersize, typename Input, typename MemoryTupleTuple, std::size_t... otherindexes>
    constexpr static auto Gather_MemoryScheduler_only_index(const int input, const int final_output, const MemoryTupleTuple &mtupletuple, const std::index_sequence<otherindexes...> &) noexcept {
        const int output = (final_output != -1) ? ((input == 0) ? Buffersize - sizeof(Output<Input>) : 0) : -1;

        const int rest_Memory = Buffersize - ((input != -1) ? sizeof(Input) : 0) - ((final_output != -1) ? sizeof(Output<Input>) : 0);

        const int dynamic_start = (input == 0) ? sizeof(Input) : sizeof(Output<Input>);

        const int dynamic_stop = dynamic_start + ((MemoryDynamic > rest_Memory) ? 0 : MemoryDynamic);

        return std::tuple(                                                    // Tuple<*Input [Matrix], *Output [Matrix], *Dynamic [Matrix<char,n>]>
                std::get<otherindexes>(mtupletuple)...,                       // So far
                std::make_tuple(input, output, dynamic_start, dynamic_stop)); // Current
    }

    template <size_t Buffersize,
              typename Input,
              int input,
              int final_output,
              int output        = (final_output != -1) ? ((input == 0) ? static_cast<int>(Buffersize) - static_cast<int>(sizeof(Output<Input>)) : 0) : -1,
              int rest_Memory   = Buffersize - ((input != -1) ? static_cast<int>(sizeof(Input)) : 0) - ((final_output != -1) ? static_cast<int>(sizeof(Output<Input>)) : 0),
              int dynamic_start = (input == 0) ? static_cast<int>(sizeof(Input)) : static_cast<int>(sizeof(Output<Input>)),
              int dynamic_stop  = dynamic_start + ((MemoryDynamic > rest_Memory) ? 0 : MemoryDynamic),
              typename MemoryTupleTuple,
              std::size_t... otherindexes>
    constexpr static auto Gather_MemoryScheduler_only_index_sequence(const MemoryTupleTuple &mtupletuple, const std::index_sequence<otherindexes...> &) noexcept {
        return std::tuple(                                                                 // Tuple<*Input [Matrix], *Output [Matrix], *Dynamic [Matrix<char,n>]>
                std::get<otherindexes>(mtupletuple)...,                                    // So far
                std::integer_sequence<int, input, output, dynamic_start, dynamic_stop>{}); // Current
    }

    template <size_t Buffersize, bool front, typename Input, typename MemoryTupleTuple, std::size_t... otherindexes>
    constexpr static auto Gather_MemoryScheduler(char buffer[Buffersize], const Input *input, MemoryTupleTuple mtupletuple, std::index_sequence<otherindexes...>) noexcept {
        Output<Input> *const output = reinterpret_cast<Output<Input> *>(&buffer[(front != MemoryInlined) ? 0 : Buffersize - sizeof(Output<Input>)]);

        const size_t rest_Memory = Buffersize - sizeof(Input) - sizeof(Output<Input>);

        Matrix<char, DimensionOrder::ERROR, (MemoryDynamic > rest_Memory) ? 0 : MemoryDynamic> *const dynamic =
                reinterpret_cast<Matrix<char, DimensionOrder::ERROR, (MemoryDynamic > rest_Memory) ? 0 : MemoryDynamic> *>(&buffer[(front != MemoryInlined) ? sizeof(Output<Input>) : sizeof(Input)]);

        return std::tuple(                                // Tuple<*Input [Matrix], *Output [Matrix], *Dynamic [Matrix<char,n>]>
                std::get<otherindexes>(mtupletuple)...,   // So far
                std::make_tuple(input, output, dynamic)); // Current
    }

    template <size_t Buffersize, bool front, typename Input, typename MemoryTupleTuple, std::size_t... otherindexes>
    constexpr static auto Gather_MemoryScheduler(
            char buffer[Buffersize], const Input *input, Output<Input> *const output, MemoryTupleTuple mtupletuple, std::index_sequence<otherindexes...>) noexcept {
        const size_t rest_Memory = Buffersize - sizeof(Input) - sizeof(Output<Input>);

        Matrix<char, DimensionOrder::ERROR, (MemoryDynamic > rest_Memory) ? 0 : MemoryDynamic> *const dynamic =
                reinterpret_cast<Matrix<char, DimensionOrder::ERROR, (MemoryDynamic > rest_Memory) ? 0 : MemoryDynamic> *>(&buffer[(front != MemoryInlined) ? sizeof(Output<Input>) : sizeof(Input)]);

        return std::tuple(                                // Tuple<*Input [Matrix], *Output [Matrix], *Dynamic [Matrix<char,n>]>
                std::get<otherindexes>(mtupletuple)...,   // So far
                std::make_tuple(input, output, dynamic)); // Current
    }

    template <size_t PermanentMemoryTotalSize,
              typename Input,
              int PermanentMemoryPos, // input position
              typename PermanentMemoryTupleTuple,
              std::size_t... otherindexes>
    constexpr static auto Gather_PermanentMemoryScheduler_only_index_sequence(const PermanentMemoryTupleTuple &pmtupletuple, const std::index_sequence<otherindexes...> &) noexcept {
        static_assert(PermanentMemoryPos + MemoryPermanent<Input> <= PermanentMemoryTotalSize, "Permanent Memory is larger than assigned Memory Size");
        return std::tuple(                                                                 // Tuple<Permanent Index start, Permanent Memory Size>
                std::get<otherindexes>(pmtupletuple)...,                                    // So far
                std::integer_sequence<int, PermanentMemoryPos, MemoryPermanent<Input>>{}); // Current
    }
};
