#pragma once

#include <iostream>
#include <tuple>
#include <utility>

#include "./Complex.hpp"

template <class Ch, class Tr, typename... Types>
std::basic_ostream<Ch, Tr> &operator<<(std::basic_ostream<Ch, Tr> &os, const std::tuple<Types...> &vals) noexcept;

template <class Ch, class Tr, typename Indexes, typename... Types>
struct PrintTuple_helper;

template <class Ch, class Tr, size_t Index, size_t... Indexes, typename... Types>
struct PrintTuple_helper<Ch, Tr, std::index_sequence<Index, Indexes...>, Types...> {
    __attribute__((always_inline))
    static inline std::basic_ostream<Ch, Tr> &print(std::basic_ostream<Ch, Tr> &os, const std::tuple<Types...> &vals) noexcept {
        os << std::get<Index>(vals) << ", ";
        return PrintTuple_helper<Ch, Tr, std::index_sequence<Indexes...>, Types...>::print(os, vals);
    }
};

template <class Ch, class Tr, size_t Index, typename... Types>
struct PrintTuple_helper<Ch, Tr, std::index_sequence<Index>, Types...> {
    __attribute__((always_inline))
    static inline std::basic_ostream<Ch, Tr> &print(std::basic_ostream<Ch, Tr> &os, const std::tuple<Types...> &vals) noexcept {
        os << std::get<Index>(vals);
        return os;
    }
};

template <class Ch, class Tr, typename... Types>
std::basic_ostream<Ch, Tr> &operator<<(std::basic_ostream<Ch, Tr> &os, const std::tuple<Types...> &vals) noexcept {
    os << "{";
    PrintTuple_helper<Ch, Tr, std::index_sequence_for<Types...>, Types...>::print(os, vals);
    os << "}";
    return os;
}

template <class Ch, class Tr, size_t... indexes>
std::basic_ostream<Ch, Tr> &operator<<(std::basic_ostream<Ch, Tr> &os, const std::index_sequence<indexes...> &vals) noexcept {
    os << "{";
    ((os << indexes << ", "), ...);
    os << "\b\b}";
    return os;
}

template <class Ch, class Tr, typename Type>
std::basic_ostream<Ch, Tr> &operator<<(std::basic_ostream<Ch, Tr> &os, const Complex<Type> &val) noexcept {
    os << val.real() << " + " << val.imag() << "i";
    return os;
}