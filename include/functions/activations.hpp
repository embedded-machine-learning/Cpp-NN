#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <sys/types.h>
#include <type_traits>

#include "../types/Benchmark.hpp"
#include "../types/Complex.hpp"

template <typename T>
constexpr auto PassThrough = [](const T &x) { return x; };

template <typename T>
constexpr auto Tanh = [](const T &x) { return std::tanh(x); };

// template <typename T>
// constexpr auto FastTanh = [](const T val){
//     static_assert(true, "FastTanh is only defined for float, use FastTanh<float> instead");
// };

template <typename T>
constexpr auto LeakyReLU = [](const T val) {
    return (val < static_cast<T>(0)) ? (static_cast<T>(0.01) * val) : val; // Leaky ReLU with a slope of 0.01 for negative values
};

template <>
constexpr auto LeakyReLU<int32_t> = [](const int32_t val, int32_t scale) -> int8_t {
    return (val < static_cast<int32_t>(0)) ? (static_cast<int32_t>(0.01 * scale) * val) : val; // Leaky ReLU with a slope of 0.01 for negative values
};



template <typename T>
constexpr auto ReLU = [](const T val) {
    return (val < static_cast<T>(0)) ? static_cast<T>(0) : val; // ReLU activation function
};



template <typename T>
    requires(std::is_convertible_v<float, T>)
constexpr auto FastTanh = [](const T val) -> T {
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

// https://en.wikipedia.org/wiki/Fast_inverse_square_root
// Quack 3 implementation of Fast Inverse Square Root
// with customizable number of iterations
union floatUnion {
    float    f;
    uint32_t i;
};

template <std::size_t it = 1, typename T = float>
    requires(std::is_convertible_v<float, T>)
constexpr auto FastInvertSQRT = [](const T val) -> T {
    const float    threehalfs   = 1.5F;
    const uint32_t magic_number = 0x5f3759df;
    floatUnion     y            = {.f = static_cast<float>(val)};

    const floatUnion x2 = {.f = val * 0.5F};
    y.i                 = magic_number - (y.i >> 1); // what the fuck?

#pragma GCC unroll(65534)
    for (std::size_t i = 0; i < it; i++) {
        y.f = y.f * (threehalfs - (x2.f * y.f * y.f)); // nst iteration
    }
    return static_cast<T>(y.f);
};

template <typename T = float>
    requires(std::is_convertible_v<float, T>)
constexpr auto InvertSQRT = [](const T val) -> T { return static_cast<T>(1 / sqrtf(val)); };

template <typename T>
constexpr auto Norm = [](const Complex<T> val) -> T { return static_cast<T>(sqrtf(val.real() * val.real() + val.imag() * val.imag())); };

template <std::size_t it = 1, typename T = float>
    requires(std::is_convertible_v<float, T>)
constexpr auto FastNorm = [](const Complex<T> val) -> T {
    const float tmp = val.real() * val.real() + val.imag() * val.imag();
    return tmp * FastInvertSQRT<it, T>(tmp);
};

template <std::size_t it = 1, typename T = float>
    requires(std::is_convertible_v<float, T>)
constexpr auto QuakeTanh = [](const T val) -> T {
    const float tmp = 1 + val * val;
    return tmp * FastInvertSQRT<it, T>(tmp);
};

template <typename T = float>
    requires(std::is_convertible_v<float, T>)
constexpr auto SimpleTanhAprox = [](const T val) -> T {
    const float tmp = 1 + val * val;
    return tmp / sqrtf(tmp);
};

template <typename T = float>
    requires(std::is_convertible_v<float, T>)
constexpr auto HardTanh = [](const T val) -> T { return (val < static_cast<T>(-1)) ? static_cast<T>(-1) : ((val > static_cast<T>(1)) ? static_cast<T>(1) : val); };
