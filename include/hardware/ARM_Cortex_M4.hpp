#pragma once
#include "../MAC.hpp"
#include "../Matrix.hpp"

#include <complex>
#include <type_traits>

#warning "Untested ARM Cortex-M4 code, ..."

union SMID32_t_int8 {
    uint32_t smid;

    struct {
        int8_t data[4];
    } data;
};

union SMID32_t_int16 {
    uint32_t smid;

    struct {
        int16_t data[2];
    } data;
};

union SMID32_t_int32 {
    uint32_t smid;

    struct {
        int32_t data[1];
    } data;
};

#if false 
__attribute__((always_inline)) inline uint32_t __SMLAD(uint32_t op1, uint32_t op2, uint32_t op3) {
    uint32_t result;
    __asm volatile("smlad %0, %1, %2, %3" : "=r"(result) : "r"(op1), "r"(op2), "r"(op3));
    return result;
}


__attribute__((always_inline, hot)) inline void __dual_float_mac(float &res1, float &res2, const float &input1, const float &input2, const float &weight1, const float &weight2) {
    float intermediate1;
    float intermediate2;
    /*
    %0 - result1
    %1 - result2
    %2 - intermediate1
    %3 - intermediate2
    %4 - input1
    %5 - input2
    %6 - weight1
    %7 - weight2
    */
   __asm("\
       vmul.f32 %[intermediate1], %[input1], %[weight1]; \
       vmul.f32 %[intermediate2], %[input2], %[weight2]; \
       vadd.f32 %[res1], %[res1], %[intermediate1]; \
       vadd.f32 %[res2], %[res2], %[intermediate2] "
       : /*outputs*/  [res1] "+t" (res1), [res2] "+t" (res2), [intermediate1] "=&t"(intermediate1), [intermediate2] "=&t" (intermediate2)
       : /*inputs*/   [input1] "t" (input1), [input2] "t" (input2), [weight1] "t" (weight1), [weight2] "t"(weight2)
   );
}

__attribute__((always_inline, hot)) inline void __dual_float_pnmac(float &res1, float &res2, const float &input1, const float &input2, const float &weight1, const float &weight2) {
    float intermediate1;
    float intermediate2;
    /*
    %0 - result1
    %1 - result2
    %2 - intermediate1
    %3 - intermediate2
    %4 - input1
    %5 - input2
    %6 - weight1
    %7 - weight2
    */
   __asm("\
       vmul.f32 %[intermediate1], %[input1], %[weight1]; \
       vmul.f32 %[intermediate2], %[input2], %[weight2]; \
       vadd.f32 %[res1], %[res1], %[intermediate1]; \
       vsub.f32 %[res2], %[res2], %[intermediate2] "
       : /*outputs*/  [res1] "+t" (res1), [res2] "+t" (res2), [intermediate1] "=&t"(intermediate1), [intermediate2] "=&t" (intermediate2)
       : /*inputs*/   [input1] "t" (input1), [input2] "t" (input2), [weight1] "t" (weight1), [weight2] "t"(weight2)
   );
}

__attribute__((always_inline)) inline uint32_t __SXTB16_ROR0(const uint32_t op1) {
    uint32_t result;
    __asm("sxtb16 %0, %1" : "=r"(result) : "r"(op1));
    return result;
}

__attribute__((always_inline)) inline uint32_t __SXTB16_ROR8(const uint32_t op1) {
    uint32_t result;
    __asm("sxtb16 %0, %1, ror 8" : "=r"(result) : "r"(op1));
    return result;
}

// Old Float16 code -- maybe implement again later
// __attribute__((always_inline)) inline float __VCVTB(const uint32_t op1) {
//     float result;
//     __asm("vcvtb.f16.f32 %0, %1" : "=w"(result) : "w"(op1));
//     return result;
// }

// __attribute__((always_inline)) inline float __VCVTT(const uint32_t op1) {
//     float result;
//     __asm("vcvtt.f16.f32 %0, %1" : "=w"(result) : "w"(op1));
//     return result;
// }

#else

#warning "Using non-ASM fallback implementations for ARM Cortex-M4"

__attribute__((always_inline)) inline void __dual_float_mac(float &res1, float &res2, const float &input1, const float &input2, const float &weight1, const float &weight2) {
    res1 += input1 * weight1;
    res2 += input2 * weight2;
}

__attribute__((always_inline)) inline void __dual_float_pnmac(float &res1, float &res2, const float &input1, const float &input2, const float &weight1, const float &weight2) {
    res1 += input1 * weight1;
    res2 -= input2 * weight2;
}

__attribute__((always_inline)) inline uint32_t __SXTB16_ROR0(const uint32_t op1) {
    SMID32_t_int8  tmp    = {.smid = op1};
    SMID32_t_int16 result = {.data = {static_cast<int16_t>(tmp.data.data[0]), static_cast<int16_t>(tmp.data.data[2])}};
    return result.smid;
}

__attribute__((always_inline)) inline uint32_t __SXTB16_ROR8(const uint32_t op1) {
    SMID32_t_int8  tmp    = {.smid = op1};
    SMID32_t_int16 result = {.data = {static_cast<int16_t>(tmp.data.data[1]), static_cast<int16_t>(tmp.data.data[3])}};
    return result.smid;
}

__attribute__((always_inline)) inline uint32_t __SMLAD(uint32_t op1, uint32_t op2, uint32_t op3) {
    SMID32_t_int16 a   = {.smid = op1};
    SMID32_t_int16 b   = {.smid = op2};
    SMID32_t_int32 res = {.smid = op3};
    res.data.data[0] += static_cast<int32_t>(a.data.data[0]) * static_cast<int32_t>(b.data.data[0]);
    res.data.data[0] += static_cast<int32_t>(a.data.data[1]) * static_cast<int32_t>(b.data.data[1]);
    return res.smid;
}
#endif

template <std::size_t KernelPeralell, std::size_t InputPeralell, typename AccType, typename InputType, typename WeightType>
__attribute__((always_inline)) inline void vectorMultiplyAccumulate(AccType (&acc)[KernelPeralell],
                                                                    const InputType (&input)[InputPeralell],
                                                                    const WeightType (&weights)[KernelPeralell][InputPeralell]) noexcept {
#pragma GCC                                unroll(65534)
    for (std::size_t kernel_parallel = 0; kernel_parallel < KernelPeralell; kernel_parallel++) {
                               #pragma GCC unroll(65534)
        for (std::size_t input_parallel = 0; input_parallel < InputPeralell; input_parallel++) {
            acc[kernel_parallel] += input[input_parallel] * weights[kernel_parallel][input_parallel];
        }
                                   }
}

template <>
__attribute__((always_inline)) inline void vectorMultiplyAccumulate<1, 2, int32_t, int16_t, int16_t>(int32_t (&acc)[1], const int16_t (&input)[2], const int16_t (&weights)[1][2]) noexcept {
    const SMID32_t_int16 smid_intput{.data = {input[0], input[1]}};
    const SMID32_t_int16 smid_weights{.data = {weights[0][0], weights[0][1]}};
    SMID32_t_int32       acc_smid{.data = {acc[0]}};
    acc_smid.smid = __SMLAD(smid_intput.smid, smid_weights.smid, acc_smid.smid);
    acc[0]        = acc_smid.data.data[0];
}

template <>
__attribute__((always_inline)) inline void vectorMultiplyAccumulate<1, 4, int32_t, int8_t, int8_t>(int32_t (&acc)[1], const int8_t (&input)[4], const int8_t (&weights)[1][4]) noexcept {
    const SMID32_t_int8 smid_intput{.data = {input[0], input[1], input[2], input[3]}};
    const SMID32_t_int8 smid_weights{.data = {weights[0][0], weights[0][1], weights[0][2], weights[0][3]}};
    SMID32_t_int32      acc_smid{.data = {acc[0]}};
    const uint32_t      a = __SXTB16_ROR0(smid_intput.smid);
    const uint32_t      b = __SXTB16_ROR0(smid_weights.smid);
    acc_smid.smid         = __SMLAD(a, b, acc_smid.smid);
    const uint32_t c      = __SXTB16_ROR8(smid_intput.smid);
    const uint32_t d      = __SXTB16_ROR8(smid_weights.smid);
    acc_smid.smid         = __SMLAD(c, d, acc_smid.smid);
    acc[0]                = acc_smid.data.data[0];
}

template <>
__attribute__((always_inline)) inline void vectorMultiplyAccumulate<2, 1, float, float, float>(float (&acc)[2], const float (&input)[1], const float (&weights)[2][1]) noexcept {
    __dual_float_mac(acc[0], acc[1], input[0], input[0], weights[0][0], weights[1][0]);
}

__attribute__((always_inline)) inline void vectorComplexMultiplyAccumulate(Complex<float> (&acc)[1], const float (&input)[1], const Complex<float> (&weights)[1][1]) noexcept {
    __dual_float_pnmac(acc[0].real(), acc[0].imag(), input[0], input[0], weights[0][0].real(), weights[0][0].imag());
}

__attribute__((always_inline)) inline void vectorRealOnlyMultiplyAccumulate(float (&acc)[1], const Complex<float> (&input)[1], const Complex<float> (&weights)[1][1]) noexcept {
    __dual_float_pnmac(acc[0], acc[0], input[0].real(), input[0].imag(), weights[0][0].real(), weights[0][0].imag());
}

template <>
// template <typename AccumulationType, typename InputType, typename WeightType, typename Lambda = decltype([]() {})>
struct OverrideOperation<int32_t, int16_t, int16_t, decltype(multily_accumulate<int32_t, int16_t, int16_t>)> {
    template <IsMatrixType AccumulationMatrixType, IsMatrixType InputMatrixType, IsMatrixType WeightMatrixType, DimensionOrder OperationOrder = "boi">
        requires(std::is_same_v<typename std::remove_cvref_t<AccumulationMatrixType>::value_type, int32_t> && std::is_same_v<typename std::remove_cvref_t<InputMatrixType>::value_type, int16_t> &&
                 std::is_same_v<typename std::remove_cvref_t<WeightMatrixType>::value_type, int16_t>)
    __attribute__((always_inline)) inline static void op(AccumulationMatrixType &output, const InputMatrixType &input, const WeightMatrixType &weights) {
        static_assert(AccumulationMatrixType::order.containsOnly("bo"), "AccumulationMatrixType must be 'ibo' sub-(Input, Batch, Output)");
        static_assert(InputMatrixType::order.containsOnly("ib"), "InputMatrixType must be 'ib' sub-(Input, Batch)");
        static_assert(WeightMatrixType::order.containsOnly("iob"), "WeightMatrixType must be 'io' sub-(Input, Output)");

        constexpr auto input_channels  = InputMatrixType::dimensions[InputMatrixType::order.indexOf('i')];
        constexpr auto output_channels = AccumulationMatrixType::dimensions[AccumulationMatrixType::order.indexOf('o')];
        constexpr auto batch_channels  = AccumulationMatrixType::dimensions[AccumulationMatrixType::order.indexOf('b')];

        constexpr std::size_t output_vec_size         = 1;
        constexpr auto        primary_output_channels = output_channels; // Only primary output as mod 1

        constexpr std::size_t input_vec_size              = 2; // We use 2 the SMID operations for input parallelism
        constexpr auto        primary_input_channels      = input_channels - input_channels % input_vec_size;
        constexpr auto        primary_input_channels_rest = input_channels % input_vec_size;

        const auto input_permuted   = permute<"bi">(input);
        auto       output_permuted  = permute<"bo">(output);
        const auto weights_permuted = permute<"boi">(weights);

        // Perfect unrolling for outputchannels % vec_size == 0
#pragma GCC unroll(65534)
        for (Dim_size_t batch_pos = 0; batch_pos < batch_channels; batch_pos += 1) {
#pragma GCC unroll(65534)
            for (Dim_size_t output_pos = 0; output_pos < primary_output_channels; output_pos += output_vec_size) {
                int32_t local_accumulation[output_vec_size] = {output_permuted.at(batch_pos, output_pos)};
#pragma GCC unroll(65534)
                for (Dim_size_t input_pos = 0; input_pos < primary_input_channels; input_pos += input_vec_size) {
                    const int16_t local_input[input_vec_size]                    = {input_permuted.at(batch_pos, input_pos), input_permuted.at(batch_pos, input_pos + 1)};
                    const int16_t local_weights[output_vec_size][input_vec_size] = {{weights_permuted.at(batch_pos, output_pos, input_pos), weights_permuted.at(batch_pos, output_pos, input_pos + 1)}};
                    vectorMultiplyAccumulate<output_vec_size, input_vec_size, int32_t, int16_t, int16_t>(local_accumulation, local_input, local_weights);
                }
                if constexpr (primary_input_channels_rest > 0) {
                    const int16_t local_input[primary_input_channels_rest]                    = {input_permuted.at(batch_pos, primary_input_channels)};
                    const int16_t local_weights[output_vec_size][primary_input_channels_rest] = {{weights_permuted.at(batch_pos, output_pos, primary_input_channels)}};
                    vectorMultiplyAccumulate<output_vec_size, primary_input_channels_rest, int32_t, int16_t, int16_t>(local_accumulation, local_input, local_weights);
                }

#pragma GCC unroll(65534)
                for (Dim_size_t i = 0; i < output_vec_size; i++) {
                    output_permuted.at(batch_pos, output_pos + i) = local_accumulation[i];
                }
            }
        }
    }
};

template <>
// template <typename AccumulationType, typename InputType, typename WeightType, typename Lambda = decltype([]() {})>
struct OverrideOperation<int32_t, int8_t, int8_t, decltype(multily_accumulate<int32_t, int8_t, int8_t>)> {
    template <IsMatrixType AccumulationMatrixType, IsMatrixType InputMatrixType, IsMatrixType WeightMatrixType, DimensionOrder OperationOrder = "boi">
        requires(std::is_same_v<typename std::remove_cvref_t<AccumulationMatrixType>::value_type, int32_t> && std::is_same_v<typename std::remove_cvref_t<InputMatrixType>::value_type, int8_t> &&
                 std::is_same_v<typename std::remove_cvref_t<WeightMatrixType>::value_type, int8_t>)
    __attribute__((always_inline)) inline static void op(AccumulationMatrixType &output, const InputMatrixType &input, const WeightMatrixType &weights) {
        static_assert(AccumulationMatrixType::order.containsOnly("bo"), "AccumulationMatrixType must be 'ibo' sub-(Input, Batch, Output)");
        static_assert(InputMatrixType::order.containsOnly("ib"), "InputMatrixType must be 'ib' sub-(Input, Batch)");
        static_assert(WeightMatrixType::order.containsOnly("iob"), "WeightMatrixType must be 'io' sub-(Input, Output)");

        constexpr auto input_channels  = InputMatrixType::dimensions[InputMatrixType::order.indexOf('i')];
        constexpr auto output_channels = AccumulationMatrixType::dimensions[AccumulationMatrixType::order.indexOf('o')];
        constexpr auto batch_channels  = AccumulationMatrixType::dimensions[AccumulationMatrixType::order.indexOf('b')];

        constexpr std::size_t output_vec_size         = 1;
        constexpr auto        primary_output_channels = output_channels; // Only primary output as mod 1

        constexpr std::size_t input_vec_size              = 4; // We use 4 the SMID operations for input parallelism
        constexpr auto        primary_input_channels      = input_channels - input_channels % input_vec_size;
        constexpr auto        primary_input_channels_rest = input_channels % input_vec_size;

        const auto input_permuted   = permute<"bi">(input);
        auto       output_permuted  = permute<"bo">(output);
        const auto weights_permuted = permute<"boi">(weights);

        // Perfect unrolling for outputchannels % vec_size == 0
#pragma GCC unroll(65534)
        for (Dim_size_t batch_pos = 0; batch_pos < batch_channels; batch_pos += 1) {
#pragma GCC unroll(65534)
            for (Dim_size_t output_pos = 0; output_pos < primary_output_channels; output_pos += output_vec_size) {
                int32_t local_accumulation[output_vec_size] = {output_permuted.at(batch_pos, output_pos)};
#pragma GCC unroll(65534)
                for (Dim_size_t input_pos = 0; input_pos < primary_input_channels; input_pos += input_vec_size) {
                    const int8_t local_input[input_vec_size] = {input_permuted.at(batch_pos, input_pos), input_permuted.at(batch_pos, input_pos + 1), input_permuted.at(batch_pos, input_pos + 2),
                                                                input_permuted.at(batch_pos, input_pos + 3)};

                    const int8_t local_weights[output_vec_size][input_vec_size] = {{weights_permuted.at(batch_pos, output_pos, input_pos), weights_permuted.at(batch_pos, output_pos, input_pos + 1),
                                                                                    weights_permuted.at(batch_pos, output_pos, input_pos + 2),
                                                                                    weights_permuted.at(batch_pos, output_pos, input_pos + 3)}};
                    vectorMultiplyAccumulate<output_vec_size, input_vec_size, int32_t, int8_t, int8_t>(local_accumulation, local_input, local_weights);
                }
                if constexpr (primary_input_channels_rest > 0) {
                    int8_t local_input[primary_input_channels_rest];
                    int8_t local_weights[output_vec_size][primary_input_channels_rest];
#pragma GCC unroll(65534)
                    for (Dim_size_t i = 0; i < primary_input_channels_rest; i++) {
                        local_input[i] = input_permuted.at(batch_pos, primary_input_channels + i);
                    }
#pragma GCC unroll(65534)
                    for (Dim_size_t i = 0; i < primary_input_channels_rest; i++) {
                        local_weights[0][i] = weights_permuted.at(batch_pos, output_pos, primary_input_channels + i);
                    }
                    vectorMultiplyAccumulate<output_vec_size, primary_input_channels_rest, int32_t, int8_t, int8_t>(local_accumulation, local_input, local_weights);
                }

#pragma GCC unroll(65534)
                for (Dim_size_t i = 0; i < output_vec_size; i++) {
                    output_permuted.at(batch_pos, output_pos + i) = local_accumulation[i];
                }
            }
        }
    }
};

template <>
struct OverrideOperation<float, float, float, decltype(multily_accumulate<float, float, float>)> {
    template <IsMatrixType AccumulationMatrixType, IsMatrixType InputMatrixType, IsMatrixType WeightMatrixType, DimensionOrder OperationOrder = "boi">
        requires(std::is_same_v<typename std::remove_cvref_t<AccumulationMatrixType>::value_type, float> && std::is_same_v<typename std::remove_cvref_t<InputMatrixType>::value_type, float> &&
                 std::is_same_v<typename std::remove_cvref_t<WeightMatrixType>::value_type, float>)
    __attribute__((always_inline, hot)) inline static void op(AccumulationMatrixType &output, const InputMatrixType &input, const WeightMatrixType &weights) {
        static_assert(AccumulationMatrixType::order.containsOnly("bo"), "AccumulationMatrixType must be 'ibo' sub-(Input, Batch, Output)");
        static_assert(InputMatrixType::order.containsOnly("ib"), "InputMatrixType must be 'ib' sub-(Input, Batch)");
        static_assert(WeightMatrixType::order.containsOnly("iob"), "WeightMatrixType must be 'io' sub-(Input, Output)");

        constexpr auto input_channels  = InputMatrixType::dimensions[InputMatrixType::order.indexOf('i')];
        constexpr auto output_channels = AccumulationMatrixType::dimensions[AccumulationMatrixType::order.indexOf('o')];
        constexpr auto batch_channels  = AccumulationMatrixType::dimensions[AccumulationMatrixType::order.indexOf('b')];

        constexpr std::size_t vec_size                     = 2; // We use 2 as the vector size for NEON operations
        constexpr auto        primary_output_channels      = output_channels - output_channels % vec_size;
        constexpr auto        primary_output_channels_rest = output_channels % vec_size;

        const auto input_permuted   = permute<"bi">(input);
        auto       output_permuted  = permute<"bo">(output);
        const auto weights_permuted = permute<"boi">(weights);

          // Perfect unrolling for outputchannels % vec_size == 0
        if constexpr (primary_output_channels > 0) {
#pragma GCC unroll(65534)
            for (Dim_size_t batch_pos = 0; batch_pos < batch_channels; batch_pos += 1) {
#pragma GCC unroll(65534)
                for (Dim_size_t output_pos = 0; output_pos < primary_output_channels; output_pos += vec_size) {
                    float local_accumulation[vec_size] = {output_permuted.at(batch_pos, output_pos), output_permuted.at(batch_pos, output_pos + 1)};
#pragma GCC unroll(65534)
                    for (Dim_size_t input_pos = 0; input_pos < input_channels; input_pos += 1) {
                        const float local_input[1]             = {input_permuted.at(batch_pos, input_pos)};
                        const float local_weights[vec_size][1] = {{weights_permuted.at(batch_pos, output_pos, input_pos)}, {weights_permuted.at(batch_pos, output_pos + 1, input_pos)}};
                        vectorMultiplyAccumulate<vec_size, 1, float, float, float>(local_accumulation, local_input, local_weights);
                    }
#pragma GCC unroll(65534)
                    for (Dim_size_t i = 0; i < vec_size; i++) {
                        output_permuted.at(batch_pos, output_pos + i) = local_accumulation[i];
                    }
                }
            }
        }
        if constexpr (primary_output_channels_rest > 0) {
#pragma GCC unroll(65534)
            for (Dim_size_t batch_pos = 0; batch_pos < batch_channels; batch_pos += 1) {
                // Catch all
                float local_accumulation[primary_output_channels_rest];
#pragma GCC unroll(65534)
                for (Dim_size_t i = 0; i < primary_output_channels_rest; i++) {
                    local_accumulation[i] = output_permuted.at(batch_pos, primary_output_channels + i);
                }
#pragma GCC unroll(65534)
                for (Dim_size_t input_pos = 0; input_pos < input_channels; input_pos += 1) {
                    const float local_input = input_permuted.at(batch_pos, input_pos);
                    float       local_weights[primary_output_channels_rest];
#pragma GCC unroll(65534)
                    for (Dim_size_t i = 0; i < primary_output_channels_rest; i++) {
                        local_weights[i] = weights_permuted.at(batch_pos, primary_output_channels + i, input_pos);
                    }
                    vectorMultiplyAccumulate<primary_output_channels_rest, 1, float, float, float>(local_accumulation, local_input, local_weights);
                }
#pragma GCC unroll(65534)
                for (Dim_size_t i = 0; i < primary_output_channels_rest; i++) {
                    output_permuted.at(batch_pos, primary_output_channels + i) = local_accumulation[i];
                }
            }
        }
    }
};

template <>
struct OverrideOperation<Complex<float>, float, Complex<float>, decltype(multily_accumulate<Complex<float>, float, Complex<float>>)> {
    template <IsMatrixType AccumulationMatrixType, IsMatrixType InputMatrixType, IsMatrixType WeightMatrixType, DimensionOrder OperationOrder = "boi">
        requires(std::is_same_v<typename std::remove_cvref_t<AccumulationMatrixType>::value_type, Complex<float>> && std::is_same_v<typename std::remove_cvref_t<InputMatrixType>::value_type, float> &&
                 std::is_same_v<typename std::remove_cvref_t<WeightMatrixType>::value_type, Complex<float>>)
    __attribute__((always_inline))
    // __attribute__((noinline))
    inline static void
    op(AccumulationMatrixType &output, const InputMatrixType &input, const WeightMatrixType &weights) {
        static_assert(AccumulationMatrixType::order.containsOnly("bo"), "AccumulationMatrixType must be 'ibo' sub-(Input, Batch, Output)");
        static_assert(InputMatrixType::order.containsOnly("ib"), "InputMatrixType must be 'ib' sub-(Input, Batch)");
        static_assert(WeightMatrixType::order.containsOnly("iob"), "WeightMatrixType must be 'io' sub-(Input, Output)");

        constexpr auto input_channels  = InputMatrixType::dimensions[InputMatrixType::order.indexOf('i')];
        constexpr auto output_channels = AccumulationMatrixType::dimensions[AccumulationMatrixType::order.indexOf('o')];
        constexpr auto batch_channels  = AccumulationMatrixType::dimensions[AccumulationMatrixType::order.indexOf('b')];

        const auto input_permuted   = permute<"bi">(input);
        auto       output_permuted  = permute<"bo">(output);
        const auto weights_permuted = permute<"boi">(weights);

#pragma GCC unroll(65534)
        for (Dim_size_t batch_pos = 0; batch_pos < batch_channels; batch_pos += 1) {
#pragma GCC unroll(65534)
            for (Dim_size_t output_pos = 0; output_pos < output_channels; output_pos += 2) {
                Complex<float> local_accumulation[1] = {output_permuted.at(batch_pos, output_pos)};
#pragma GCC unroll(65534)
                for (Dim_size_t input_pos = 0; input_pos < input_channels; input_pos += 1) {
                    const float          local_input[1]      = {input_permuted.at(batch_pos, input_pos)};
                    const Complex<float> local_weights[1][1] = {{{weights_permuted.at(batch_pos, output_pos, input_pos)}}};
                    vectorComplexMultiplyAccumulate(local_accumulation, local_input, local_weights);
                }
#pragma GCC unroll(65534)
                for (Dim_size_t i = 0; i < 1; i++) {
                    output_permuted.at(batch_pos, output_pos + i) = local_accumulation[i];
                }
            }
        }
    }
};