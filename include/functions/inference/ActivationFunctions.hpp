#pragma once

#include <cstdint>
#include <type_traits>

#ifdef __ARM_FP16_FORMAT_IEEE
#define ACTIVATIONS_USE_FP16
#endif

#ifdef __ARM_FP16_FORMAT_ALTERNATIVE
#define ACTIVATIONS_USE_FP16
#endif

#define EXPENSIVE_ACTIVATIONS

/*
=========================================================================
                                ReLU
=========================================================================
*/
struct ReLU_struct {
    template <typename input, typename output = input, typename... ActivationParameters>
    __attribute__((always_inline)) static inline output Act(const input val, ActivationParameters... params) noexcept{
        static_assert(std::is_same_v<input, output>, "Default ReLU only works with same input and output types, PLease Implement the desired conversion");
        static_assert(sizeof...(params) == 0, "ReLU does not take any parameters, check if the correct ReLU is used");
        if (val >= static_cast<const input>(0)) // should be negative flag
            return val;
        return static_cast<output>(0);
    };
};

const auto ReLU = ReLU_struct();

// FLOAT -------------------------------------------------------------------------------------------------------------------------------
template <>
__attribute__((always_inline)) inline float ReLU_struct::Act<float, float>(float val) noexcept {
    const float val2 = (val > static_cast<float>(0)) ? val : static_cast<float>(0);
    return val2;
}

template <>
__attribute__((always_inline)) inline double ReLU_struct::Act<double, double>(double val) noexcept {
    const double val2 = (val > static_cast<double>(0)) ? val : static_cast<double>(0);
    return val2;
}

#ifdef ACTIVATIONS_USE_FP16
template <>
__attribute__((always_inline)) inline __fp16 ReLU_struct::Act<float, __fp16>(const float val) noexcept {
    if (val >= static_cast<float>(0)) // should be negative flag
        return static_cast<__fp16>(val);
    return static_cast<__fp16>(0);
}
#endif

// for testing
template <>
__attribute__((always_inline)) inline float ReLU_struct::Act<float, float, float, float>(float val, float B, float C) noexcept {
    if (val >= static_cast<float>(0)) // should be negative flag
        return val * B + C;
    return static_cast<float>(0);
}

// for testing
template <>
__attribute__((always_inline)) inline int8_t ReLU_struct::Act<float, int8_t>(const float val) noexcept {
    if (val >= static_cast<float>(0)) // should be negative flag
        return static_cast<int8_t>(val);
    return static_cast<int8_t>(0);
}

// for testing
template <>
__attribute__((always_inline)) inline int16_t ReLU_struct::Act<float, int16_t>(const float val) noexcept {
    if (val >= static_cast<float>(0)) // should be negative flag
        return static_cast<int16_t>(val);
    return static_cast<int16_t>(0);
}

// for testing
template <>
__attribute__((always_inline)) inline int16_t ReLU_struct::Act<int32_t, int16_t>(const int32_t val) noexcept {
    const int32_t  val2 = val >> 9;
    const uint32_t val3 = (val2 < 0) ? 0 : val2;
    const int16_t  val4 = (val3 > 0x7FFF) ? static_cast<int16_t>(0x7FFF) : static_cast<int16_t>(val3);
    return val4;
}

template <>
__attribute__((always_inline)) inline int16_t ReLU_struct::Act<int32_t, int16_t, int32_t>(const int32_t val, const int32_t mul) noexcept {
    const int32_t  val2 = (val * mul) >> 16;
    const uint32_t val3 = (val2 < 0) ? 0 : val2;
    const int16_t  val4 = (val3 > 0x7FFF) ? static_cast<int16_t>(0x7FFF) : static_cast<int16_t>(val3);
    return val4;
}

template <>
__attribute__((always_inline)) inline int8_t ReLU_struct::Act<int32_t, int8_t>(const int32_t val) noexcept {
    const int32_t  val2 = val >> 9;
    const uint32_t val3 = (val2 < 0) ? 0 : val2;
    const int8_t   val4 = (val3 > 0x7F) ? static_cast<int8_t>(0x7F) : static_cast<int8_t>(val3);
    return val4;
}

template <>
__attribute__((always_inline)) inline int8_t ReLU_struct::Act<int32_t, int8_t, int32_t>(const int32_t val, const int32_t mul) noexcept {
    const int32_t  val2 = (val * mul) >> 16;
    const uint32_t val3 = (val2 < 0) ? 0 : val2;
    const int8_t   val4 = (val3 > 0x7F) ? static_cast<int8_t>(0x7F) : static_cast<int8_t>(val3);
    return val4;
}

// for testing
template <>
__attribute__((always_inline)) inline int8_t ReLU_struct::Act<float, int8_t, float>(float val, float shift) noexcept {
    if (val >= static_cast<float>(0)) // should be negative flag
        return static_cast<int8_t>(val);
    return static_cast<int8_t>(0);
}

// for testing
template <>
__attribute__((always_inline)) inline int16_t ReLU_struct::Act<float, int16_t, float>(float val, float shift) noexcept {
    if (val >= static_cast<float>(0)) // should be negative flag
        return static_cast<int16_t>(val);
    return static_cast<int16_t>(0);
}

// for testing
template <>
__attribute__((always_inline)) inline uint8_t ReLU_struct::Act<float, uint8_t, float>(float val, float shift) noexcept {
    if (val >= static_cast<float>(0)) // should be negative flag
        return static_cast<uint8_t>(val);
    return static_cast<uint8_t>(0);
}

// for testing
template <>
__attribute__((always_inline)) inline float ReLU_struct::Act<float, float, float>(float val, float B) noexcept {
    if (val >= static_cast<float>(0)) // should be negative flag
        return val * B;
    return static_cast<float>(0);
}

// INT ---------------------------------------------------------------------------------------------------------------------------------

// ReLU for int32 downcasting to uint16 using min and max of uint16
template <>
__attribute__((always_inline)) inline uint16_t ReLU_struct::Act<int32_t, uint16_t, int32_t>(int32_t val, int32_t shift) noexcept {
    int32_t a{val >> shift};
    int32_t b = (a & (~(a >> 31)));
    int32_t c = ((32767 - b) >> 31);
    return static_cast<uint16_t>((b & ~c) | (32767 & c));
}



/*
=========================================================================
                                LeakyReLU
=========================================================================
*/
struct LeakyReLU_struct {
    constexpr static float negative_slope=0.01;
    template <typename input, typename output = input, typename... ActivationParameters>
    __attribute__((always_inline)) static inline output Act(const input val, ActivationParameters... params) noexcept{
        static_assert(std::is_same_v<input, output>, "Default ReLU only works with same input and output types, PLease Implement the desired conversion");
        static_assert(sizeof...(params) == 0, "ReLU does not take any parameters, check if the correct ReLU is used");
        if (val >= static_cast<const input>(0)) // should be negative flag
            return val;
        return static_cast<output>(val*static_cast<const input>(negative_slope));
    };
};

const auto LeakyReLU = LeakyReLU_struct();

// FLOAT -------------------------------------------------------------------------------------------------------------------------------
template <>
__attribute__((always_inline)) inline float LeakyReLU_struct::Act<float, float>(float val) noexcept {
    const float val2 = (val > static_cast<float>(0)) ? val : val*negative_slope;
    return val2;
}



/*
=========================================================================
                                Passthrough
=========================================================================
*/

struct PassThrough_struct {
    template <typename input, typename output = input, typename... ActivationParameters>
    __attribute__((always_inline)) static inline output Act(input val, ActivationParameters... params) noexcept {
        static_assert(std::is_same_v<input, output>, "Passthrough only works with same input and output types");
        return val;
    };
};

// for testing
template <>
__attribute__((always_inline)) inline int16_t PassThrough_struct::Act<float, int16_t, float>(float val, float shift) noexcept {
    return static_cast<int16_t>(static_cast<int>(val) >> static_cast<int>(shift));
}

// for testing
template <>
__attribute__((always_inline)) inline float PassThrough_struct::Act<int32_t, float>(int32_t val) noexcept {
    return static_cast<float>(val);
}

// for testing
template <>
__attribute__((always_inline)) inline int16_t PassThrough_struct::Act<int32_t, int16_t>(int32_t val) noexcept {
    return static_cast<int16_t>(val);
}

// for testing
template <>
__attribute__((always_inline)) inline int8_t PassThrough_struct::Act<int32_t, int8_t>(int32_t val) noexcept {
    return static_cast<int8_t>(val);
}

#ifdef ACTIVATIONS_USE_FP16
template <>
__attribute__((always_inline)) inline __fp16 PassThrough_struct::Act<float, __fp16>(float val) noexcept {
    return static_cast<__fp16>(val);
}
#endif

const auto Passthrough = PassThrough_struct();


#ifdef EXPENSIVE_ACTIVATIONS

/*
=========================================================================
                                Sigmoid
=========================================================================
*/

#include <cmath>

struct Sigmoid_struct {
    template <typename input, typename output = input, typename... ActivationParameters>
    __attribute__((always_inline)) static inline output Act(input val, ActivationParameters... params) noexcept {
        static_assert(std::is_same_v<input, output>, "Sigmoid only works with same input and output types");
        return 1.0/(1+ exp(-val));
    };
};

const auto Sigmoid = Sigmoid_struct();

/*
=========================================================================
                                Tanh
=========================================================================
*/

#include <cmath>

struct Tanh_struct {
    template <typename input, typename output = input, typename... ActivationParameters>
    __attribute__((always_inline)) static inline output Act(input val, ActivationParameters... params) noexcept {
        static_assert(std::is_same_v<input, output>, "Sigmoid only works with same input and output types");
        return tanh(val);
    };
};

const auto Tanh = Tanh_struct();


#endif