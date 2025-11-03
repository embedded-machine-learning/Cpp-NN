#pragma once

#include <array>
#include <tuple>
#include <type_traits>
#include <utility>


template <std::size_t OutLength, std::size_t CharLength>
 requires(OutLength >= CharLength)
constexpr std::array<char, OutLength> toArray(const char (&str)[CharLength]) {
    std::array<char, OutLength> ret{0};
    for (std::size_t i = 0; i < CharLength; i++) {
        ret.at(i) = str[i];
    }
    return ret;
}

template <std::size_t Length>
constexpr std::array<char, Length> toArrayAuto(const char (&str)[Length]) {
    return toArray<Length>(str);
}


template <typename T, std::size_t sa, std::size_t sb>
consteval std::array<T, sa + sb> concat(const std::array<T, sa> &a, const std::array<T, sb> &b) {
    std::array<T, sa + sb> ret{};
    for (std::size_t i = 0; i < sa; i++) {
        ret[i] = a[i];
    }
    for (std::size_t i = 0; i < sb; i++) {
        ret[i + sa] = b[i];
    }
    return ret;
}

template <typename A, typename... Bs>
consteval auto concat(const A &a, const Bs &...b) {
    return concat(a, concat(b...));
}

// https://stackoverflow.com/questions/23999573/convert-a-number-to-a-string-literal-with-constexpr
namespace detail {
template <unsigned... Digits>
struct ToChars {
    static const char value[];
};

template <unsigned... Digits>
constexpr char ToChars<Digits...>::value[] = {('0' + Digits)..., 0};

template <unsigned Rem, unsigned... Digits>
struct Explode : Explode<Rem / 10, Rem % 10, Digits...> {};

template <unsigned... Digits>
struct Explode<0, Digits...> : ToChars<Digits...> {};

template <> // modification to add 0 capability
struct Explode<0> : ToChars<0> {};

} // namespace detail

template <unsigned Num>
struct NumToStringS : detail::Explode<Num> {};

template <long Num>
constexpr auto num_to_string = concat(toArrayAuto((Num >= 0) ? " " : "-"), toArrayAuto(NumToStringS<static_cast<unsigned>((Num >= 0) ? Num : -Num)>::value));

template <std::size_t N, std::array<std::size_t,N> Num, typename >
constexpr auto array_to_string_hidden = toArray<1>("");

template <std::size_t N, std::array<std::size_t,N> Num, std::size_t... VariadicIndices>
constexpr auto array_to_string_hidden<N,Num,std::index_sequence<VariadicIndices...>> = concat(
    toArrayAuto("["),
    concat(num_to_string<Num[VariadicIndices]>, toArrayAuto(", "))...,
    toArrayAuto("\b\b]"));

template <std::size_t N, std::array<std::size_t,N> Num>
constexpr auto array_to_string = array_to_string_hidden<N, Num, std::make_index_sequence<N>>;


template <typename T,std::size_t N>
consteval std::array<T,N> makeFilledArray(T t) {
    std::array<T, N> ret{};
    for (std::size_t i = 0; i < N; i++) {
        ret[i] = t;
    }
    return ret;
}



template <typename T, typename... Ts>
constexpr bool all_same = ((std::is_same_v<T, Ts>) && ...);

template <typename T, std::size_t N, typename... ArrayType>
    requires((sizeof...(ArrayType) >= 2) && all_same<std::array<T, N>, ArrayType...>)
consteval std::array<bool, N> variadicArrayCompare(const ArrayType &...arr) {
    std::array<bool, N> result{};
    for (std::size_t i = 0; i < N; i++) {
        result[i] = ((std::get<0>(std::make_tuple(arr...))[i] == arr[i]) && ...);
    }
    return result;
}

template <std::size_t N, typename... ArrayType>
    requires((sizeof...(ArrayType) >= 2) && all_same<std::array<std::size_t, N>, ArrayType...>)
consteval std::array<std::size_t, N> variadicArrayAdd(const ArrayType &...arr) {
    std::array<std::size_t, N> result{};
    for (std::size_t i = 0; i < N; i++) {
        result[i] = (arr[i] + ...);
    }
    return result;
}

template <typename... ValueType>
    requires((sizeof...(ValueType) >= 2) && all_same<std::size_t, ValueType...>)
consteval std::array<std::size_t, sizeof...(ValueType)> variadicCumSum(const ValueType... arr) noexcept {
    std::array<std::size_t, sizeof...(ValueType)> result{arr...};
    for (std::size_t i = 1; i < sizeof...(ValueType); i++) {
        result[i] += result[i - 1];
    }
    return result;
}

template <typename T>
using reference_or_rvalue = std::conditional_t<std::is_rvalue_reference_v<T>, std::add_rvalue_reference_t<std::remove_cvref_t<T>>, std::add_lvalue_reference_t<std::remove_cvref_t<T>>>;

template <typename T>
using const_reference_or_rvalue =
        std::conditional_t<std::is_rvalue_reference_v<T>, std::add_rvalue_reference_t<std::add_const_t<std::remove_cvref_t<T>>>, std::add_lvalue_reference_t<std::add_const_t<std::remove_cvref_t<T>>>>;

template<typename T>
using drop_reference = std::conditional_t<std::is_const_v<T>, std::add_const_t<std::remove_cvref_t<T>>, std::remove_cvref_t<T>>;

template<typename T>
using reference_or_copy = std::conditional_t<std::is_rvalue_reference_v<T>, drop_reference<T>, std::add_lvalue_reference_t<drop_reference<T>>>;

template<bool A, typename B>
// using reference_or_rvalue_const_preserving = std::conditional_t<std::is_rvalue_reference_v<T>, std::add_rvalue_reference_t<drop_reference<T>>, std::add_lvalue_reference_t<drop_reference<T>>>;
using conditional_const = std::conditional_t<A, std::add_const_t<B>, B>;

template<typename T>
using enforce_l_or_r_ref = std::conditional_t<std::is_lvalue_reference_v<T>, std::add_lvalue_reference_t<std::remove_reference_t<T>>, std::add_rvalue_reference_t<std::remove_reference_t<T>>>;