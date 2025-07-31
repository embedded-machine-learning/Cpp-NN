#pragma once
#include "../MAC.hpp"
#include "../Matrix.hpp"

#include <arm_neon.h>
#include <complex>
#include <type_traits>

#warning "Untested NEON code, ..."

union my_float32x4_t {
    struct {
        float data[4];
    } data;

    float32x4_t float32x4;
    float64x2_t float64x2;
};

template <std::size_t KernelPeralell, typename Type>
__attribute__((always_inline)) inline void vectorMultiplyAccumulate(Type (&acc)[KernelPeralell], const Type &input, const Type (&weights)[KernelPeralell]) {

#pragma GCC unroll(65534)
    for (std::size_t kernel_parallel = 0; kernel_parallel < KernelPeralell; kernel_parallel++) {
        acc[kernel_parallel] += input * weights[kernel_parallel];
    }
}

template <>
__attribute__((always_inline)) inline void vectorMultiplyAccumulate<4, float>(float (&acc)[4], const float &input, const float (&weights)[4]) {
    const my_float32x4_t vinput{{input, input, input, input}};
    const my_float32x4_t vweight{{weights[0], weights[1], weights[2], weights[3]}};
    my_float32x4_t       vacc{{acc[0], acc[1], acc[2], acc[3]}};
    vacc.float32x4 = vfmaq_f32(vacc.float32x4, vweight.float32x4, vinput.float32x4);

#pragma GCC unroll(65534)
    for (std::size_t kernel_parallel = 0; kernel_parallel < 4; kernel_parallel++) {
        acc[kernel_parallel] = vacc.data.data[kernel_parallel];
    }
}

template <std::size_t KernelPeralell, typename Type>
__attribute__((always_inline)) inline void vectorComplexMultiplyAccumulate(Complex<Type> (&acc)[KernelPeralell], const Complex<Type> &input, const Complex<Type> (&weights)[KernelPeralell]) {
    static_assert(KernelPeralell > 0, "Not Implemented for speed yet");
#pragma GCC unroll(65534)
    for (std::size_t kernel_parallel = 0; kernel_parallel < KernelPeralell; kernel_parallel++) {
        acc[kernel_parallel] += input * weights[kernel_parallel];
    }
}

template <std::size_t KernelPeralell, typename Type>
__attribute__((always_inline)) inline void vectorComplexMultiplyAccumulate(std::array<Type, 2> (&acc)[KernelPeralell], const Type &input, const Complex<Type> (&weights)[KernelPeralell]) {
    // std::cout << "Not implemented" << std::endl;
#pragma GCC unroll(65534)
    for (std::size_t kernel_parallel = 0; kernel_parallel < KernelPeralell; kernel_parallel++) {
        acc[kernel_parallel][0] += input * weights[kernel_parallel].real();
        acc[kernel_parallel][1] += input * weights[kernel_parallel].imag();
    }
}

template <>
__attribute__((always_inline)) inline void vectorComplexMultiplyAccumulate<2, float>(std::array<float, 2> (&acc)[2], const float &input, const Complex<float> (&weights)[2]) {
    const my_float32x4_t vweight{.float32x4 = vld1q_f32(reinterpret_cast<const float *>(&weights[0]))}; // complex<float> is 2 floats, so we can load it as a float array // very ugly but works
    my_float32x4_t       vacc{.float32x4 = vld1q_f32(&acc[0][0])};                                      // complex<float> is 2 floats, so we can load it as a float array // very ugly but works
    vacc.float32x4 = vmlaq_n_f32(vacc.float32x4, vweight.float32x4, input);

    vst1q_f32(reinterpret_cast<float *>(&acc[0][0]), vacc.float32x4); // at this point lets also store using the m256 union // very ugly but works
}

template <std::size_t KernelPeralell, typename Type>
__attribute__((always_inline)) inline void vectorRealOnlyMultiplyAccumulate(std::array<Type, 2> (&acc)[KernelPeralell], const Complex<Type> &input, const Complex<Type> (&weights)[KernelPeralell]) {

#pragma GCC unroll(65534)
    for (std::size_t kernel_parallel = 0; kernel_parallel < KernelPeralell; kernel_parallel++) {
        acc[kernel_parallel][0] += input.real() * weights[kernel_parallel].real();
        acc[kernel_parallel][1] += input.imag() * weights[kernel_parallel].imag();
    }
}

template <>
__attribute__((always_inline)) inline void vectorRealOnlyMultiplyAccumulate<2, float>(std::array<float, 2> (&acc)[2], const Complex<float> &input, const Complex<float> (&weights)[2]) {
    // const m256 vinput{{
    //      std::real(input), std::imag(input),
    //      std::real(input), std::imag(input),
    //      std::real(input), std::imag(input),
    //      std::real(input), std::imag(input)}};    // gets incorrectly
    // loaded
    const my_float32x4_t vinput{.float64x2 = vld1q_f64(reinterpret_cast<const double *>(&input))}; // complex<float> is 2 floats, so we can broadcast it as a double // very ugly but works

    // const m256 vweight{{std::real(weights[0]), std::imag(weights[0]), // also dosent work bc compiler sees it as something else and wants to blend it
    //                     std::real(weights[1]), std::imag(weights[1]), //
    //                     std::real(weights[2]), std::imag(weights[2]), //
    //                     std::real(weights[3]), std::imag(weights[3])}};
    const my_float32x4_t vweight{.float32x4 = vld1q_f32(reinterpret_cast<const float *>(&weights[0]))}; // complex<float> is 2 floats, so we can load it as a float array // very ugly but works

    // m256 vacc{{acc[0][0], acc[0][1], //
    //            acc[1][0], acc[1][1], //
    //            acc[2][0], acc[2][1], //
    //            acc[3][0], acc[3][1]}};
    my_float32x4_t vacc{.float32x4 = vld1q_f32(reinterpret_cast<float *>(&acc[0][0]))}; // complex<float> is 2 floats, so we can load it as a float array // very ugly but works
    vacc.float32x4 = vmlaq_f32(vacc.float32x4, vweight.float32x4, vinput.float32x4);

    // #pragma GCC unroll(65534)
    //     for (std::size_t kernel_parallel = 0; kernel_parallel < 4; kernel_parallel++) {
    //         acc[kernel_parallel]=std::array<float,2>{vacc.data.data[2*kernel_parallel],vacc.data.data[2*kernel_parallel + 1]};
    //     }
    vst1q_f32(reinterpret_cast<float *>(&acc[0]), vacc.float32x4); // at this point lets also store using the m256 union // very ugly but works
}

template <>
struct OverrideOperation<float, float, float, decltype(multily_accumulate<float, float, float>)> {
    template <IsMatrixType AccumulationMatrixType, IsMatrixType InputMatrixType, IsMatrixType WeightMatrixType, DimensionOrder OperationOrder = "boi">
        requires(std::is_same_v<typename std::remove_cvref_t<AccumulationMatrixType>::value_type, float> && std::is_same_v<typename std::remove_cvref_t<InputMatrixType>::value_type, float> &&
                 std::is_same_v<typename std::remove_cvref_t<WeightMatrixType>::value_type, float>)
    __attribute__((always_inline)) inline static void op(AccumulationMatrixType &output, const InputMatrixType &input, const WeightMatrixType &weights) {
        static_assert(AccumulationMatrixType::order.containsOnly("bo"), "AccumulationMatrixType must be 'ibo' sub-(Input, Batch, Output)");
        static_assert(InputMatrixType::order.containsOnly("ib"), "InputMatrixType must be 'ib' sub-(Input, Batch)");
        static_assert(WeightMatrixType::order.containsOnly("iob"), "WeightMatrixType must be 'io' sub-(Input, Output)");

        constexpr auto input_channels  = InputMatrixType::dimensions[InputMatrixType::order.indexOf('i')];
        constexpr auto output_channels = AccumulationMatrixType::dimensions[AccumulationMatrixType::order.indexOf('o')];
        constexpr auto batch_channels  = AccumulationMatrixType::dimensions[AccumulationMatrixType::order.indexOf('b')];

        constexpr std::size_t vec_size                     = 4; // We use 4 as the vector size for NEON operations
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
                    float local_accumulation[vec_size] = {output_permuted.at(batch_pos, output_pos), output_permuted.at(batch_pos, output_pos + 1), output_permuted.at(batch_pos, output_pos + 2),
                                                          output_permuted.at(batch_pos, output_pos + 3)};
#pragma GCC unroll(65534)
                    for (Dim_size_t input_pos = 0; input_pos < input_channels; input_pos += 1) {
                        const float local_input             = input_permuted.at(batch_pos, input_pos);
                        const float local_weights[vec_size] = {weights_permuted.at(batch_pos, output_pos, input_pos), weights_permuted.at(batch_pos, output_pos + 1, input_pos),
                                                               weights_permuted.at(batch_pos, output_pos + 2, input_pos), weights_permuted.at(batch_pos, output_pos + 3, input_pos)};
                        vectorMultiplyAccumulate<vec_size, float>(local_accumulation, local_input, local_weights);
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
                    vectorMultiplyAccumulate<primary_output_channels_rest, float>(local_accumulation, local_input, local_weights);
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
struct OverrideOperation<Complex<float>, Complex<float>, Complex<float>, decltype(multily_accumulate<Complex<float>, Complex<float>, Complex<float>>)> {
    template <IsMatrixType AccumulationMatrixType, IsMatrixType InputMatrixType, IsMatrixType WeightMatrixType, DimensionOrder OperationOrder = "boi">
        requires(std::is_same_v<typename std::remove_cvref_t<AccumulationMatrixType>::value_type, Complex<float>> &&
                 std::is_same_v<typename std::remove_cvref_t<InputMatrixType>::value_type, Complex<float>> &&
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
        static_assert(input_channels <= 0, "Disabled for now, needs to be fixed");
        constexpr auto primary_output_channels        = output_channels - output_channels % 4;
        constexpr auto primary_output_channels_rest   = output_channels % 4;
        constexpr auto secondary_output_channels      = primary_output_channels_rest - primary_output_channels_rest % 2;
        constexpr auto secondary_output_channels_rest = primary_output_channels_rest % 2;

        const auto input_permuted   = permute<"bi">(input);
        auto       output_permuted  = permute<"bo">(output);
        const auto weights_permuted = permute<"boi">(weights);

// Perfect unrolling for outputchannels % 4 == 0
#pragma GCC unroll(65534)
        for (Dim_size_t batch_pos = 0; batch_pos < batch_channels; batch_pos += 1) {
            if constexpr (primary_output_channels > 0) {
#pragma GCC unroll(65534)
                for (Dim_size_t output_pos = 0; output_pos < primary_output_channels; output_pos += 4) {
                    Complex<float> local_accumulation[4] = {output_permuted.at(batch_pos, output_pos), output_permuted.at(batch_pos, output_pos + 1), output_permuted.at(batch_pos, output_pos + 2),
                                                            output_permuted.at(batch_pos, output_pos + 3)};
#pragma GCC unroll(65534)
                    for (Dim_size_t input_pos = 0; input_pos < input_channels; input_pos += 1) {
                        const Complex<float> local_input      = input_permuted.at(batch_pos, input_pos);
                        const Complex<float> local_weights[4] = {weights_permuted.at(batch_pos, output_pos, input_pos), weights_permuted.at(batch_pos, output_pos + 1, input_pos),
                                                                 weights_permuted.at(batch_pos, output_pos + 2, input_pos), weights_permuted.at(batch_pos, output_pos + 3, input_pos)};
                        vectorComplexMultiplyAccumulate<4, float>(local_accumulation, local_input, local_weights);
                    }
#pragma GCC unroll(65534)
                    for (Dim_size_t i = 0; i < 4; i++) {
                        output_permuted.at(batch_pos, output_pos + i) = local_accumulation[i];
                    }
                }
            }
            // Subperfect unrolling for outputchannels % 2 == 0
            if constexpr (secondary_output_channels > 0) {
#pragma GCC unroll(65534)
                for (Dim_size_t output_pos = primary_output_channels; output_pos < primary_output_channels + secondary_output_channels; output_pos += 2) {
                    Complex<float> local_accumulation[2] = {output_permuted.at(batch_pos, output_pos), output_permuted.at(batch_pos, output_pos + 1)};
#pragma GCC unroll(65534)
                    for (Dim_size_t input_pos = 0; input_pos < input_channels; input_pos += 1) {
                        const Complex<float> local_input      = input_permuted.at(batch_pos, input_pos);
                        const Complex<float> local_weights[2] = {weights_permuted.at(batch_pos, output_pos, input_pos), weights_permuted.at(batch_pos, output_pos + 1, input_pos)};
                        vectorComplexMultiplyAccumulate<2, float>(local_accumulation, local_input, local_weights);
                    }
#pragma GCC unroll(65534)
                    for (Dim_size_t i = 0; i < 4; i++) {
                        output_permuted.at(batch_pos, output_pos + i) = local_accumulation[i];
                    }
                }
            }
            // Catch all
            if constexpr (secondary_output_channels_rest > 0) {
                Complex<float> local_accumulation[secondary_output_channels_rest];
#pragma GCC unroll(65534)
                for (Dim_size_t i = 0; i < secondary_output_channels_rest; i++) {
                    local_accumulation[i] = output_permuted.at(batch_pos, primary_output_channels + secondary_output_channels + i);
                }
#pragma GCC unroll(65534)
                for (Dim_size_t input_pos = 0; input_pos < input_channels; input_pos += 1) {
                    const Complex<float> local_input = input_permuted.at(batch_pos, input_pos);
                    Complex<float>       local_weights[secondary_output_channels_rest];
#pragma GCC unroll(65534)
                    for (Dim_size_t i = 0; i < secondary_output_channels_rest; i++) {
                        local_weights[i] = weights_permuted.at(batch_pos, primary_output_channels + secondary_output_channels + i, input_pos);
                    }
                    vectorComplexMultiplyAccumulate<secondary_output_channels_rest, float>(local_accumulation, local_input, local_weights);
                }
#pragma GCC unroll(65534)
                for (Dim_size_t i = 0; i < secondary_output_channels_rest; i++) {
                    output_permuted.at(batch_pos, primary_output_channels + secondary_output_channels + i) = local_accumulation[i];
                }
            }
        }
    }
};

template <>
struct OverrideOperation<std::array<float, 2>, float, Complex<float>, decltype(multily_accumulate<std::array<float, 2>, float, Complex<float>>)> {
    template <IsMatrixType AccumulationMatrixType, IsMatrixType InputMatrixType, IsMatrixType WeightMatrixType, DimensionOrder OperationOrder = "boi">
        requires(std::is_same_v<typename std::remove_cvref_t<AccumulationMatrixType>::value_type, std::array<float, 2>> &&
                 std::is_same_v<typename std::remove_cvref_t<InputMatrixType>::value_type, float> && std::is_same_v<typename std::remove_cvref_t<WeightMatrixType>::value_type, Complex<float>>)
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

        constexpr auto primary_output_channels      = output_channels - output_channels % 2;
        constexpr auto primary_output_channels_rest = output_channels % 2;

        const auto input_permuted   = permute<"bi">(input);
        auto       output_permuted  = permute<"bo">(output);
        const auto weights_permuted = permute<"boi">(weights);

        // std::cout << "input channels: " << input_channels << ", output channels: " << output_channels << ", batch channels: " << batch_channels
        //           << ", primary output channels: " << primary_output_channels << ", primary output channels rest: " << primary_output_channels_rest << std::endl;

        // Perfect unrolling for outputchannels % 4 == 0
        if constexpr (primary_output_channels > 0) {
#pragma GCC unroll(65534)
            for (Dim_size_t batch_pos = 0; batch_pos < batch_channels; batch_pos += 1) {
#pragma GCC unroll(65534)
                for (Dim_size_t output_pos = 0; output_pos < primary_output_channels; output_pos += 2) {
                    std::array<float, 2> local_accumulation[2] = {output_permuted.at(batch_pos, output_pos), output_permuted.at(batch_pos, output_pos + 1)};
#pragma GCC unroll(65534)
                    for (Dim_size_t input_pos = 0; input_pos < input_channels; input_pos += 1) {
                        const float          local_input      = input_permuted.at(batch_pos, input_pos);
                        const Complex<float> local_weights[2] = {weights_permuted.at(batch_pos, output_pos, input_pos), weights_permuted.at(batch_pos, output_pos + 1, input_pos)};
                        vectorComplexMultiplyAccumulate<2, float>(local_accumulation, local_input, local_weights);
                    }
#pragma GCC unroll(65534)
                    for (Dim_size_t i = 0; i < 2; i++) {
                        output_permuted.at(batch_pos, output_pos + i) = local_accumulation[i];
                    }
                }
            }
        }
        // Catch all
        if constexpr (primary_output_channels_rest > 0) {
#pragma GCC unroll(65534)
            for (Dim_size_t batch_pos = 0; batch_pos < batch_channels; batch_pos += 1) {
                std::array<float, 2> local_accumulation[primary_output_channels_rest];
#pragma GCC unroll(65534)
                for (Dim_size_t i = 0; i < primary_output_channels_rest; i++) {
                    local_accumulation[i] = output_permuted.at(batch_pos, primary_output_channels + i);
                }
#pragma GCC unroll(65534)
                for (Dim_size_t input_pos = 0; input_pos < input_channels; input_pos += 1) {
                    const float    local_input = input_permuted.at(batch_pos, input_pos);
                    Complex<float> local_weights[primary_output_channels_rest];
#pragma GCC unroll(65534)
                    for (Dim_size_t i = 0; i < primary_output_channels_rest; i++) {
                        local_weights[i] = weights_permuted.at(batch_pos, primary_output_channels + i, input_pos);
                    }
                    vectorComplexMultiplyAccumulate<primary_output_channels_rest, float>(local_accumulation, local_input, local_weights);
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
struct OverrideOperation<std::array<float, 2>, Complex<float>, Complex<float>, decltype(split_real_only_multily_accumulate<float, Complex<float>, Complex<float>>)> {
    template <IsMatrixType AccumulationMatrixType, IsMatrixType InputMatrixType, IsMatrixType WeightMatrixType, DimensionOrder OperationOrder = "boi">
        requires(std::is_same_v<typename std::remove_cvref_t<AccumulationMatrixType>::value_type, std::array<float, 2>> &&
                 std::is_same_v<typename std::remove_cvref_t<InputMatrixType>::value_type, Complex<float>> &&
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

        constexpr auto primary_output_channels      = output_channels - output_channels % 2;
        constexpr auto primary_output_channels_rest = output_channels % 2;

        const auto input_permuted   = permute<"bi">(input);
        auto       output_permuted  = permute<"bo">(output);
        const auto weights_permuted = permute<"boi">(weights);
        // std::cout << "input channels: " << input_channels << ", output channels: " << output_channels << ", batch channels: " << batch_channels << std::endl;

        // Perfect unrolling for outputchannels % 2 == 0
        if constexpr (primary_output_channels > 0) {
#pragma GCC unroll(65534)
            for (Dim_size_t batch_pos = 0; batch_pos < batch_channels; batch_pos += 1) {
#pragma GCC unroll(65534)
                for (Dim_size_t output_pos = 0; output_pos < primary_output_channels; output_pos += 2) {
                    std::array<float, 2> local_accumulation[2] = {output_permuted.at(batch_pos, output_pos), output_permuted.at(batch_pos, output_pos + 1)};
#pragma GCC unroll(65534)
                    for (Dim_size_t input_pos = 0; input_pos < input_channels; input_pos += 1) {
                        const Complex<float> local_input      = input_permuted.at(batch_pos, input_pos);
                        const Complex<float> local_weights[2] = {weights_permuted.at(batch_pos, output_pos, input_pos), weights_permuted.at(batch_pos, output_pos + 1, input_pos)};
                        vectorRealOnlyMultiplyAccumulate<2, float>(local_accumulation, local_input, local_weights);
                    }
#pragma GCC unroll(65534)
                    for (Dim_size_t i = 0; i < 2; i++) {
                        output_permuted.at(batch_pos, output_pos + i) = local_accumulation[i];
                    }
                }
            }
        }

        if constexpr (primary_output_channels_rest > 0) {
#pragma GCC unroll(65534)
            for (Dim_size_t batch_pos = 0; batch_pos < batch_channels; batch_pos += 1) {
                // Catch all
                std::array<float, 2> local_accumulation[primary_output_channels_rest];
#pragma GCC unroll(65534)
                for (Dim_size_t i = 0; i < primary_output_channels_rest; i++) {
                    local_accumulation[i] = output_permuted.at(batch_pos, primary_output_channels + i);
                }
#pragma GCC unroll(65534)
                for (Dim_size_t input_pos = 0; input_pos < input_channels; input_pos += 1) {
                    const Complex<float> local_input = input_permuted.at(batch_pos, input_pos);
                    Complex<float>       local_weights[primary_output_channels_rest];
#pragma GCC unroll(65534)
                    for (Dim_size_t i = 0; i < primary_output_channels_rest; i++) {
                        local_weights[i] = weights_permuted.at(batch_pos, primary_output_channels + i, input_pos);
                    }
                    vectorRealOnlyMultiplyAccumulate<primary_output_channels_rest, float>(local_accumulation, local_input, local_weights);
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
struct DefaultMACOperation<float, Complex<float>, Complex<float>> {

    using InputType_  = float;
    using WeightType_ = Complex<float>;
    using BiasType_   = Complex<float>;

    using AccumulationType_ = std::array<float, 2>; // We use a 2D array to store the real and imaginary parts separately

    constexpr static auto lambda = multily_accumulate<AccumulationType_, InputType_, WeightType_>;
    using LambdaType             = decltype(lambda);

    template <IsMatrixType AccumulationMatrixType, IsMatrixType InputMatrixType, IsMatrixType WeightMatrixType, DimensionOrder OperationOrder = "boi">
        requires(std::is_same_v<typename std::remove_cvref_t<AccumulationMatrixType>::value_type, AccumulationType_> &&
                 std::is_same_v<typename std::remove_cvref_t<InputMatrixType>::value_type, InputType_> && std::is_same_v<typename std::remove_cvref_t<WeightMatrixType>::value_type, WeightType_>)
    __attribute__((always_inline)) inline static void op(AccumulationMatrixType &accumulation, const InputMatrixType &input, const WeightMatrixType &weights) {
        static_assert(AccumulationMatrixType::order.containsOnly("bo"), "AccumulationMatrixType must be 'ibo' sub-(Input, Batch, Output)");
        static_assert(InputMatrixType::order.containsOnly("ib"), "InputMatrixType must be 'ib' sub-(Input, Batch)");
        static_assert(WeightMatrixType::order.containsOnly("iob"), "WeightMatrixType must be 'io' sub-(Input, Output)");

        OverrideOperation<AccumulationType_, InputType_, WeightType_, decltype(lambda)>::op(accumulation, input, weights);
    }

    constexpr static auto pre_processing = [](const BiasType_ &bias) -> AccumulationType_ { return {bias.real(), bias.imag()}; };

    constexpr static auto post_processing = [](const AccumulationType_ &acc) -> BiasType_ { return Complex<float>(acc[0], acc[1]); };
};

template <>
struct RealResultMACOperation<Complex<float>, Complex<float>, float> {

    using InputType_  = Complex<float>;
    using WeightType_ = Complex<float>;
    using BiasType_   = float;

    using AccumulationType_ = std::array<float, 2>;

    constexpr static auto lambda = split_real_only_multily_accumulate<float, InputType_, WeightType_>;
    using LambdaType             = decltype(lambda);

    template <IsMatrixType AccumulationMatrixType, IsMatrixType InputMatrixType, IsMatrixType WeightMatrixType, DimensionOrder OperationOrder = "boi">
        requires(std::is_same_v<typename std::remove_cvref_t<AccumulationMatrixType>::value_type, AccumulationType_> &&
                 std::is_same_v<typename std::remove_cvref_t<InputMatrixType>::value_type, InputType_> && std::is_same_v<typename std::remove_cvref_t<WeightMatrixType>::value_type, WeightType_>)
    __attribute__((always_inline))
    // __attribute__((noinline))
    inline static void
    op(AccumulationMatrixType &output, const InputMatrixType &input, const WeightMatrixType &weights) {
        static_assert(AccumulationMatrixType::order.containsOnly("bo"), "AccumulationMatrixType must be 'ibo' sub-(Input, Batch, Output)");
        static_assert(InputMatrixType::order.containsOnly("ib"), "InputMatrixType must be 'ib' sub-(Input, Batch)");
        static_assert(WeightMatrixType::order.containsOnly("iob"), "WeightMatrixType must be 'io' sub-(Input, Output)");

        OverrideOperation<AccumulationType_, InputType_, WeightType_, decltype(lambda)>::op(output, input, weights);
    }

    constexpr static auto pre_processing = [](const BiasType_ &bias) -> AccumulationType_ { return {bias, 0}; };

    constexpr static auto post_processing = [](const AccumulationType_ &acc) -> BiasType_ { return static_cast<BiasType_>(acc[0] - acc[1]); };
};
