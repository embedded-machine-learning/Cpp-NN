#pragma once

#include <type_traits>
#include <cmath>

template<typename T>
constexpr auto PassThrough = [](const T &x) { return x; };

template<typename T>
constexpr auto Tanh = [](const T &x) { return std::tanh(x); };

// template <typename T>
// constexpr auto FastTanh = [](const T val){
//     static_assert(true, "FastTanh is only defined for float, use FastTanh<float> instead");
// };

template <typename T>
constexpr auto LeakyReLU = [](const T val){
    return (val < static_cast<T>(0)) ? (static_cast<T>(0.01) * val) : val; // Leaky ReLU with a slope of 0.01 for negative values
};

template <typename T>
requires(std::is_convertible_v<float, T>)
constexpr auto FastTanh = [](const T val) -> T{
    const auto x  = val;
    const auto ax = fabsf(x);
    const auto x2 = x * x;

    constexpr auto a = static_cast<T>(2.45550750702956f);
    constexpr auto b = static_cast<T>(0.893229853513558f);
    constexpr auto c = static_cast<T>(0.821226666969744f);
    constexpr auto d = static_cast<T>(2.44506634652299f);
    constexpr auto e = static_cast<T>(0.814642734961073f);

    return (x * (a + a * ax + (b + c * ax) * x2) / (d + (d + x2) * fabsf(x + e * x * ax)));
};


