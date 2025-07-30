#pragma once

#if false
template <typename Type>
constexpr int TypeName = 0;
#else

#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

#include "../Matrix.hpp"
#include "../MatrixOperations.hpp"
#include "../helpers/Complex.hpp"
#include "../helpers/Benchmark.hpp"
#include "c++17_helpers.hpp"

#ifdef __ARM_FP16_FORMAT_IEEE
#define TestHelpers_USE_FP16
#endif

#ifdef __ARM_FP16_FORMAT_ALTERNATIVE
#define TestHelpers_USE_FP16
#endif

template <size_t sa, size_t sb>
constexpr std::array<char, sa + sb> concat(const std::array<char, sa> &a, const std::array<char, sb> &b) {
    std::array<char, sa + sb> ret{};
    for (size_t i = 0; i < sa; i++) {
        ret[i] = a[i];
    }
    for (size_t i = 0; i < sb; i++) {
        ret[i + sa] = b[i];
    }
    return ret;
}

template <typename A, typename... Bs>
constexpr auto concat(const A &a, const Bs &...b) {
    return concat(a, concat(b...));
}

// https://stackoverflow.com/questions/23999573/convert-a-number-to-a-string-literal-with-constexpr
namespace detail {
template <unsigned... digits>
struct to_chars {
    static const char value[];
};

template <unsigned... digits>
constexpr char to_chars<digits...>::value[] = {('0' + digits)..., 0};

template <unsigned rem, unsigned... digits>
struct explode : explode<rem / 10, rem % 10, digits...> {};

template <unsigned... digits>
struct explode<0, digits...> : to_chars<digits...> {};

template <> // modification to add 0 capability
struct explode<0> : to_chars<0> {};

} // namespace detail

template <unsigned num>
struct num_to_string_s : detail::explode<num> {};

template <long num>
constexpr auto num_to_string = concat(to_array((num >= 0) ? " " : "-"), to_array(num_to_string_s<static_cast<unsigned>((num >= 0) ? num : -num)>::value));

// done

template <typename Type>
constexpr auto TypeName = to_array("Unknown");
template <>
constexpr auto TypeName<void> = to_array("void");
template <>
constexpr auto TypeName<float> = to_array("float");
#ifdef TestHelpers_USE_FP16
template <>
constexpr auto TypeName<__fp16> = to_array("__fp16");
#endif
template <>
constexpr auto TypeName<double> = to_array("double");
template <>
constexpr auto TypeName<int8_t> = to_array("int8_t");
template <>
constexpr auto TypeName<int16_t> = to_array("int16_t");
template <>
constexpr auto TypeName<int32_t> = to_array("int32_t");
template <>
constexpr auto TypeName<int64_t> = to_array("int64_t");
template <>
constexpr auto TypeName<uint8_t> = to_array("uint8_t");
template <>
constexpr auto TypeName<uint16_t> = to_array("uint16_t");
template <>
constexpr auto TypeName<uint32_t> = to_array("uint32_t");
template <>
constexpr auto TypeName<uint64_t> = to_array("uint64_t");
template <>
constexpr auto TypeName<char> = to_array("char");
template <>
constexpr auto TypeName<bool> = to_array("bool");

template <typename Type>
constexpr auto TypeName<Complex<Type>> = concat(to_array("Complex<"), TypeName<Type>, to_array(">"));
template <typename Type>
constexpr auto TypeName<helpers::Benchmark::TypeInstance<Type>> = concat(to_array("Benchmark::TypeInstance<"), TypeName<Type>, to_array(">"));
template <typename... Types>
constexpr auto TypeName<std::tuple<Types...>> = concat(to_array("std::tuple<"), concat(TypeName<Types>, to_array(", "))..., to_array("\b\b >"));
template <typename Type, Type... dims>
constexpr auto TypeName<std::integer_sequence<Type, dims...>> =
        concat(to_array("std::integer_sequence<"), TypeName<Type>, to_array(", "), concat(num_to_string<dims>, to_array(", "))..., to_array("\b\b >"));
template <typename Type, DimensionOrder Order, Dim_size_t... dims>
constexpr auto TypeName<Matrix<Type, Order, dims...>> =
        concat(to_array("Matrix<"), TypeName<Type>, to_array(","), TypeName<std::index_sequence<Order>>, concat(to_array(","), num_to_string<dims>)..., to_array(">"));
// template <typename Type, DimensionOrder Order, Dim_size_t dim>
// constexpr auto TypeName<Matrix<Type, Order, dim>> = concat(to_array("Matrix<"), TypeName<Type>, concat(to_array(","), to_array(num_to_string<dim>)), to_array(">"));
template <typename Type>
constexpr auto TypeName<const Type> = concat(to_array("const "), TypeName<Type>);
template <typename Type>
constexpr auto TypeName<Type *> = concat(TypeName<Type>, to_array("*"));

template <DimensionOrder Order>
constexpr auto TypeName<std::index_sequence<Order>> = to_array("Unknown Order");
template <>
constexpr auto TypeName<std::index_sequence<DimensionOrder::ERROR>> = to_array("ERROR");
template <>
constexpr auto TypeName<std::index_sequence<DimensionOrder::D1_Channel>> = to_array("D1_Channel");
template <>
constexpr auto TypeName<std::index_sequence<DimensionOrder::D2_Batch_Channel>> = to_array("D2_Batch_Channel");
template <>
constexpr auto TypeName<std::index_sequence<DimensionOrder::D2_Channel_Batch>> = to_array("D2_Channel_Batch");
template <>
constexpr auto TypeName<std::index_sequence<DimensionOrder::D2_OutChannel_InChannel>> = to_array("D2_OutChannel_InChannel");
template <>
constexpr auto TypeName<std::index_sequence<DimensionOrder::D2_InChannel_OutChannel>> = to_array("D2_InChannel_OutChannel");
template <>
constexpr auto TypeName<std::index_sequence<DimensionOrder::D3_Batch_Channel_Width>> = to_array("D3_Batch_Channel_Width");
template <>
constexpr auto TypeName<std::index_sequence<DimensionOrder::D3_Batch_Width_Channel>> = to_array("D3_Batch_Width_Channel");
template <>
constexpr auto TypeName<std::index_sequence<DimensionOrder::D3_OutChannel_InChannel_Kernel>> = to_array("D3_OutChannel_InChannel_Kernel");
template <>
constexpr auto TypeName<std::index_sequence<DimensionOrder::D3_Batch_Sequence_Channel>> = to_array("D3_Batch_Sequence_Channel");
template <>
constexpr auto TypeName<std::index_sequence<DimensionOrder::D4_Batch_Channel_Width_Height>> = to_array("D4_Batch_Channel_Width_Height");
template <>
constexpr auto TypeName<std::index_sequence<DimensionOrder::D4_Batch_Width_Height_Channel>> = to_array("D4_Batch_Width_Height_Channel");
template <>
constexpr auto TypeName<std::index_sequence<DimensionOrder::D4_OutChannel_InChannel_KernelWidth_KernelHeight>> = to_array("D4_OutChannel_InChannel_KernelWidth_KernelHeight");
template <>
constexpr auto TypeName<std::index_sequence<DimensionOrder::D4_OutChannel_InChannel_KernelParallel_Unrolled>> = to_array("D4_OutChannel_InChannel_KernelParallel_Unrolled");
template <>
constexpr auto TypeName<std::index_sequence<DimensionOrder::D5_OutChannel_Kernel_InChannel_KernelParallel_Unrolled>> = to_array("D5_OutChannel_Kernel_InChannel_KernelParallel_Unrolled");

template <typename Type>
auto Mprint = [](const Type &ret) -> void { std::cout << ret << "\n"; };

template <typename Type>
auto rand_gen = [](Type &ret) -> void { ret = static_cast<Type>(std::numeric_limits<Type>::max() * ((double)rand() / (double)RAND_MAX) - std::numeric_limits<Type>::max() / 2); };

// prints the matrix
template <typename Type, DimensionOrder Order, Dim_size_t... dims>
__attribute__((always_inline)) inline void Matrix_print(Matrix<Type, Order, dims...> &A) {
    Matrix_unroll_helper<Type, Order, std::integer_sequence<Dim_size_t, dims...>, decltype(Mprint<Type>)>::unroll(Mprint<Type>, A);
};

template <typename Type, DimensionOrder Order, Dim_size_t... dims>
__attribute__((always_inline)) inline void Matrix_gen(Matrix<Type, Order, dims...> &A) {
    Matrix_unroll_helper<Type, Order, std::integer_sequence<Dim_size_t, dims...>, decltype(rand_gen<Type>)>::unroll(rand_gen<Type>, A);
};

template <class Ch, class Tr, typename Type, DimensionOrder Order, Dim_size_t... dims>
std::basic_ostream<Ch, Tr> &operator<<(std::basic_ostream<Ch, Tr> &os, const Matrix<Type, Order, dims...> &vals) noexcept {
    os << "Matrix of Type: " << TypeName<Type> << " with dimensions: ";
    ((os << dims << ", "), ...);
    os << "\b\b  ";
    return os;
}

template <class Ch, class Tr, size_t size>
std::basic_ostream<Ch, Tr> &operator<<(std::basic_ostream<Ch, Tr> &os, const std::array<char, size> arr) noexcept {
    for (size_t i = 0; i < size; i++) {
        os << arr[i];
    }
    return os;
}

template <typename Type>
void print_Memory_Location(const Type *Data, const std::string &name, const char *memory, int memory_size, int division, char ch = '-') {
    const char *const initial_position = memory;
    const char *const data_position    = (const char *)(void *)(Data);
    int               offset           = static_cast<int>(data_position - initial_position);
    int               datasize         = std::ceil(static_cast<float>(sizeof(Type)) / division) * division;
    std::cout << name << std::string(10 - 1 - name.length(), ' ') << ":";
    std::cout << std::string(offset / division, ' ') << std::string(datasize / division, ch) << std::string((memory_size - offset - datasize) / division, ' ') << std::endl;
}

template <typename InputType, typename OutputType, typename Dynamic>
void print_Memory_Location(const std::tuple<InputType *, OutputType *, Dynamic *> &Data, const std::string &name, const char *memory, int memory_size, int division) {
    const char *const initial_position      = memory;
    const char *const data_Input_position   = (const char *)(void *)(std::get<0>(Data));
    const char *const data_Output_position  = (const char *)(void *)(std::get<1>(Data));
    const char *const data_Dynamic_position = (const char *)(void *)(std::get<2>(Data));
    const int         offset_Input          = static_cast<int>(data_Input_position - initial_position) / division;
    const int         offset_Output         = static_cast<int>(data_Output_position - initial_position) / division;
    const int         offset_Dynamic        = static_cast<int>(data_Dynamic_position - initial_position) / division;
    const int         datasize_Input        = std::ceil(static_cast<float>(sizeof(InputType)) / division);
    const int         datasize_Output       = std::ceil(static_cast<float>(sizeof(OutputType)) / division);
    const int         datasize_Dynamic      = std::ceil(static_cast<float>(sizeof(Dynamic)) / division);

    std::string printBuffer = std::string(memory_size / division, '-');

    if (offset_Dynamic + datasize_Dynamic <= memory_size / division)
        for (int i = 0; i < datasize_Dynamic; i++)
            printBuffer[offset_Dynamic + i] = 'D';

    if (static_cast<int>(data_Input_position - initial_position) + static_cast<float>(sizeof(InputType)) <= memory_size)
        for (int i = 0; i < datasize_Input; i++)
            printBuffer[offset_Input + i] = (printBuffer[offset_Input + i] == 'D') ? 'U' : 'I';

    if (static_cast<int>(data_Output_position - initial_position) + static_cast<float>(sizeof(OutputType)) <= memory_size)
        for (int i = 0; i < datasize_Output; i++)
            switch (printBuffer[offset_Output + i]) {
            case '-': printBuffer[offset_Output + i] = 'O'; break;
            case 'D': printBuffer[offset_Output + i] = 'T'; break;
            case 'U': printBuffer[offset_Output + i] = 'A'; break;
            case 'I': printBuffer[offset_Output + i] = 'X'; break;
            default: break;
            }
    std::cout << name << std::string(10 - 1 - name.length(), ' ') << ":";
    std::cout << printBuffer << "\tI: " << sizeof(InputType) << " O: " << sizeof(OutputType) << " D: " << sizeof(Dynamic) << " ";

    if (static_cast<int>(data_Dynamic_position - initial_position) + static_cast<float>(sizeof(Dynamic)) > memory_size)
        std::cout << "Out of Memory Dynamic Binding (How the Fuck?)";
    if (static_cast<int>(data_Input_position - initial_position) + static_cast<float>(sizeof(InputType)) > memory_size)
        std::cout << "Out of Memory Input Binding";
    if (static_cast<int>(data_Output_position - initial_position) + static_cast<float>(sizeof(OutputType)) > memory_size)
        std::cout << "Out of Memory Output Binding";
    std::cout << std::endl;
}

template <typename Type, size_t... indexes>
void print_Memory_Location_loop(const Type &Data, const char *memory, int memory_size, int division, std::index_sequence<indexes...>) {
    (print_Memory_Location(std::get<indexes>(Data), std::to_string(indexes), memory, memory_size, division), ...);
    std::cout << "I: input, O: output, D: dynamic, X: input and output, U: input and dynamic, A: input, output and dynamic, T: output and dynamic" << std::endl;
}

std::fstream nullstream("/dev/null", std::ios::out);

auto nullstreamprint = [](const auto &, const auto &ret) -> void { nullstream << ret << "\n"; };
auto coutstreamprint = [](const auto &, const auto &ret) -> void { std::cout << ret << ", " << std::endl; };

// prints the matrix
template <typename Type, DimensionOrder Order, Dim_size_t... dims>
__attribute__((always_inline)) inline void Matrix_print_nullstream(const Matrix<Type, Order, dims...> &A) {
    Matrix<Type, Order, dims...> ignore;
    Matrix_unroll_helper<Type, Order, std::integer_sequence<Dim_size_t, dims...>, decltype(nullstreamprint), Matrix<Type, Order, dims...>>::unroll(nullstreamprint, ignore, A);
};

template <typename Type, DimensionOrder Order, Dim_size_t... dims>
__attribute__((always_inline)) inline void Matrix_print(const Matrix<Type, Order, dims...> &A) {
    Matrix<Type, Order, dims...> ignore;
    Matrix_unroll_helper<Type, Order, std::integer_sequence<Dim_size_t, dims...>, decltype(coutstreamprint), Matrix<Type, Order, dims...>>::unroll(coutstreamprint, ignore, A);
};

template <typename Type, DimensionOrder Order, Dim_size_t... dims>
std::ostream &operator<<(std::ostream &os, const Matrix<Type, Order, dims...> &vals) noexcept {
    os << "Matrix of Type: " << TypeName<Type> << ", with Order: " << TypeName<std::index_sequence<Order>> << " with dimensions: ";
    ((os << dims << ", "), ...);
    os << "\b\b  ";
    return os;
}
#endif
