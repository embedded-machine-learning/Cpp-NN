#pragma once
#include <array>
#include <stddef.h>

/*
Templates to make the code c++17 compatible
these are some templates introduced above c++17
*/

// c++20 std::remove_cvref
template <typename T>
using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

// c++20 std::to_array
// specialized for char[]
template <size_t N>
constexpr std::array<char, N> to_array(const char (&str)[N]) {
    std::array<char, N> ret{};
    for (size_t i = 0; i < N; i++) {
        ret[i] = str[i];
    }
    return ret;
}

template <size_t OutLength, size_t CharLength, typename = std::enable_if_t<(OutLength > CharLength)>>
constexpr std::array<char, OutLength> to_array(const char (&str)[CharLength]) {
    std::array<char, OutLength> ret{0};
    for (size_t i = 0; i < CharLength; i++) {
        ret[i] = str[i];
    }
    return ret;
}

template< typename T, T... Ints >
constexpr T integer_at( std::integer_sequence<T, Ints...>, size_t index ) {
    return std::array{Ints...}[index];
}

template<size_t repitition, typename T,T value, T... values>
struct SequenceRepeater_helper
{
    using type = typename SequenceRepeater_helper<repitition - 1, T, value, value, values...>::type;
};

template<typename T,T value, T... values>
struct SequenceRepeater_helper<0, T, value, values...>
{
    using type = std::integer_sequence<T, values...>;
};

template<size_t repitition, typename T, T value>
using SequenceRepeater = typename SequenceRepeater_helper<repitition, T, value>::type;

template<typename A, typename B>
struct SequencesConcatenate_helper;

template<typename T, T... A, T... B>
struct SequencesConcatenate_helper<std::integer_sequence<T, A...>, std::integer_sequence<T, B...>>
{
    using type = std::integer_sequence<T, A..., B...>;
};

template<typename A, typename B>
using SequencesConcatenate = typename SequencesConcatenate_helper<A, B>::type;