#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <iostream>
#include <tuple>

#include <cstdio>

#include "../Matrix.hpp"
#include "../MatrixOperations.hpp"
#include "../layers/Sequence.hpp"
#include "human_readable_types.hpp"

#include "../types/Benchmark.hpp"

template <class Ch, class Tr, typename Type, std::size_t length>
std::basic_ostream<Ch, Tr> &operator<<(std::basic_ostream<Ch, Tr> &os, const std::array<Type, length> &vals) noexcept {
    os << "{";
    for (auto a : vals)
        os << a << ", ";
    os << "}";
    return os;
}

template <class Ch, class Tr, std::size_t length>
std::basic_ostream<Ch, Tr> &operator<<(std::basic_ostream<Ch, Tr> &os, const std::array<char, length> &vals) noexcept {
    for (auto a : vals)
        if (a != '\0')
            os << a;
    return os;
}

template <class Ch, class Tr, typename TupleType, std::size_t... Indexes>
std::basic_ostream<Ch, Tr> &print_tuple_impl(std::basic_ostream<Ch, Tr> &os, const TupleType &vals, std::index_sequence<Indexes...>) noexcept {
    os << "{";
    ((os << std::get<Indexes>(vals) << (Indexes + 1 < sizeof...(Indexes) ? ", " : "")), ...);
    os << "}";
    return os;
}

template <class Ch, class Tr, typename TupleType>
    requires requires(TupleType t) {
        { std::tuple_size<TupleType>::value } -> std::convertible_to<std::size_t>;
        typename std::tuple_element<0, TupleType>::type;
        // std::get<0>(t);
    }
std::basic_ostream<Ch, Tr> &operator<<(std::basic_ostream<Ch, Tr> &os, const TupleType &vals) noexcept {
    return print_tuple_impl(os, vals, std::make_index_sequence<std::tuple_size<TupleType>::value>());
}

template <class Ch, class Tr>
std::basic_ostream<Ch, Tr> &operator<<(std::basic_ostream<Ch, Tr> &os, const DimensionOrder order) noexcept {
    os << "DimensionOrder of length: " << order.length() << " with content: \"" << order.order << "\"";
    return os;
}

template <class Ch, class Tr, IsMatrixType MatrixType>
std::basic_ostream<Ch, Tr> &operator<<(std::basic_ostream<Ch, Tr> &os, const MatrixType matrix) noexcept {
    os << "Matrix of type: " << human_readable_type<MatrixType> << "\n";
    os << "Order: " << matrix.order << "\n";
    os << "Dimensions: " << matrix.dimensions;
    // os << "Data: \n";
    // os << matrix.data << "\n";
    return os;
}

template <IsMatrixType MatrixType>
    requires(MatrixType::number_of_dimensions == 2)
void print2DMatrix_(const MatrixType &matrix) {
    constexpr std::size_t value_length = 10; // Length of each value when printed
    // Print the second dimension header, with an arrow horizontally ( j itterator)
    constexpr std::size_t header_length = (MatrixType::dimensions[1] * value_length) / 2 + 2; // +2 for the arrow
    std::cout << std::string((header_length > 0 ? header_length : 0), ' ') << MatrixType::order[1] << " ->\n";
    for (Dim_size_t i = 0; i < matrix.dimensions[0]; ++i) {
        // Print a downwards arrow by line
        switch (i - matrix.dimensions[0] / 2 + 1) {
        case 0: std::cout << MatrixType::order[0] << "  "; break; // One row above the middle
        case 1: std::cout << "|  "; break;                        // Middle row
        case 2: std::cout << "V  "; break;                        // One row below the middle
        default: std::cout << "   "; break;                       // Other rows
        }
        for (Dim_size_t j = 0; j < matrix.dimensions[1]; ++j) {
            if constexpr (std::is_floating_point_v<typename MatrixType::value_type>) {
                printf("%8.1e, ", static_cast<float>(matrix.at(i, j))); // Print each element with 4 decimal places
            } else if constexpr (std::is_integral_v<typename MatrixType::value_type>) {
                printf("%8ld, ", static_cast<long>(matrix.at(i, j))); // Print each element with 4 decimal places
            } else {
                printf("%8.1e, ", static_cast<float>(matrix.at(i, j))); // just try to print it as float
            }
        }
        std::cout << "\b\b\n";
    }
}

template <IsMatrixType MatrixType>
    requires(MatrixType::number_of_dimensions == 2)
void print2DMatrix(const MatrixType &matrix) {
    std::cout << "Matrix of type: " << human_readable_type<MatrixType> << "\n";
    std::cout << "Order: " << matrix.order << "\n";
    std::cout << "Dimensions: " << matrix.dimensions << "\n";
    std::cout << "Data:\n";
    print2DMatrix_(matrix);
}

template <IsMatrixType MatrixType>
    requires(MatrixType::number_of_dimensions > 2)
void printNDMatrix_(const MatrixType &matrix) {
    for (Dim_size_t i = 0; i < matrix.dimensions[0]; ++i) {
        std::cout << MatrixType::order[0] << " = " << i << ":\n";
        printNDMatrix_(collapse<MatrixType::order.range(0, 2), MatrixType::order.range(1, 2)>(slice<MatrixType::order.range(0, 1), 1>(matrix, {i})));
        std::cout << "\n";
    }
}

template <IsMatrixType MatrixType>
    requires(MatrixType::number_of_dimensions == 2)
void printNDMatrix_(const MatrixType &matrix) {
    print2DMatrix_(matrix);
}

template <IsMatrixType MatrixType>
    requires(MatrixType::number_of_dimensions >= 2)
void printNDMatrix(const MatrixType &matrix) {
    std::cout << "Matrix of type: " << human_readable_type<MatrixType> << "\n";
    std::cout << "Order: " << matrix.order << "\n";
    std::cout << "Dimensions: " << matrix.dimensions << "\n";
    std::cout << "Data:\n";
    printNDMatrix_(matrix);
}

template <std::same_as<layers::MemoryLocation> T>
std::ostream &operator<<(std::ostream &os, const T memory_location) {
    os << "MemoryLocation: {"
       << "Input_index: " << memory_location.Input_index << ", Input_size: " << memory_location.Input_size << ", Output_index: " << memory_location.Output_index
       << ", Output_size: " << memory_location.Output_size << ", buffer_index: " << memory_location.buffer_index << ", buffer_size: " << memory_location.buffer_size
       << ", permanent_index: " << memory_location.permanent_index << ", permanent_size: " << memory_location.permanent_size << "}";
    return os;
}

template <std::size_t N, std::same_as<layers::MemoryLocation> T>
std::ostream &operator<<(std::ostream &os, const std::array<T, N> &memory_locations) {
    os << "Array(" << N << ") of memory locations: {\n";
    for (std::size_t i = 0; i < N; ++i) {
        os << "    " << memory_locations[i] << ",\n";
    }
    os << "\b\b}";
    return os;
}

template <typename Type>
void printBenchmark() {
    std::cout << "Benchmark for type: " << human_readable_type<Type> << "\n";
    std::cout << "counted_multiplications : " << helpers::Benchmark::TypeInstance<Type>::counted_multiplications << std::endl;
    std::cout << "counted_additions       : " << helpers::Benchmark::TypeInstance<Type>::counted_additions << std::endl;
    std::cout << "counted_divisions       : " << helpers::Benchmark::TypeInstance<Type>::counted_divisions << std::endl;
    std::cout << "counted_subtractions    : " << helpers::Benchmark::TypeInstance<Type>::counted_subtractions << std::endl;
    std::cout << "counted_comparisons     : " << helpers::Benchmark::TypeInstance<Type>::counted_comparisons << std::endl;
    std::cout << "counted_extractions     : " << helpers::Benchmark::TypeInstance<Type>::counted_extractions << std::endl;
    std::cout << "counted_abs             : " << helpers::Benchmark::TypeInstance<Type>::counted_abs << std::endl;
    std::cout << "--------------------------" << std::endl;
    std::cout << "Total counted operations: "
              << helpers::Benchmark::TypeInstance<Type>::counted_multiplications + helpers::Benchmark::TypeInstance<Type>::counted_additions +
                         helpers::Benchmark::TypeInstance<Type>::counted_divisions + helpers::Benchmark::TypeInstance<Type>::counted_subtractions +
                         helpers::Benchmark::TypeInstance<Type>::counted_comparisons + helpers::Benchmark::TypeInstance<Type>::counted_extractions + helpers::Benchmark::TypeInstance<Type>::counted_abs
              << std::endl;
    std::cout << "--------------------------" << std::endl;
}
