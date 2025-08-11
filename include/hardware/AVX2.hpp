#pragma once
#include "../MAC.hpp"
#include "../Matrix.hpp"

#include <complex>
#include <immintrin.h>
#include <type_traits>

union m256 {
    struct {
        float data[8];
    } data;

    __m256  m256;
    __m256d m256d;
};

union m128 {
    struct {
        float data[4];
    } data;

    __m128 m128;
    __m128d m128d;
};

template <std::size_t KernelPeralell, typename Type>
__attribute__((always_inline)) inline void vectorMultiplyAccumulate(Type (&acc)[KernelPeralell], const Type &input, const Type (&weights)[KernelPeralell]) {

#pragma GCC unroll(65534)
    for (std::size_t kernel_parallel = 0; kernel_parallel < KernelPeralell; kernel_parallel++) {
        acc[kernel_parallel] += input * weights[kernel_parallel];
    }
}

template <>
__attribute__((always_inline)) inline void vectorMultiplyAccumulate<8, float>(float (&acc)[8], const float &input, const float (&weights)[8]) {
    const m256 vinput{{input, input, input, input, input, input, input, input}};
    const m256 vweight{{weights[0], weights[1], weights[2], weights[3], weights[4], weights[5], weights[6], weights[7]}};
    m256       vacc{{acc[0], acc[1], acc[2], acc[3], acc[4], acc[5], acc[6], acc[7]}};
    vacc.m256 = _mm256_fmadd_ps(vweight.m256, vinput.m256, vacc.m256);
#pragma GCC unroll(65534)
    for (std::size_t kernel_parallel = 0; kernel_parallel < 8; kernel_parallel++) {
        acc[kernel_parallel] = vacc.data.data[kernel_parallel];
    }
}

template <>
__attribute__((always_inline)) inline void vectorMultiplyAccumulate<4, float>(float (&acc)[4], const float &input, const float (&weights)[4]) {
    const m128 vinput{{input, input, input, input}};
    const m128 vweight{{weights[0], weights[1], weights[2], weights[3]}};
    m128       vacc{{acc[0], acc[1], acc[2], acc[3]}};
    vacc.m128 = _mm_fmadd_ps(vweight.m128, vinput.m128, vacc.m128);
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
__attribute__((always_inline)) inline void vectorComplexMultiplyAccumulate<4, float>(std::array<float, 2> (&acc)[4], const float &input, const Complex<float> (&weights)[4]) {
    // const m256 vinput{{input, input, input, input, input, input, input, input}};
    const m256 vinput{.m256 = _mm256_broadcast_ss(&input)};

    const m256 vweight{.m256d = _mm256_load_pd(reinterpret_cast<const double *>(&weights[0]))}; // complex<float> is 2 floats, so we can load it as a float array // very ugly but works
    m256       vacc{.m256d = _mm256_load_pd(reinterpret_cast<double *>(&acc[0][0]))};           // complex<float> is 2 floats, so we can load it as a float array // very ugly but works
    vacc.m256 = _mm256_fmadd_ps(vweight.m256, vinput.m256, vacc.m256);
    _mm256_store_pd(reinterpret_cast<double *>(&acc[0][0]), vacc.m256d); // at this point lets also store using the m256 union // very ugly but works
}

template <>
__attribute__((always_inline)) inline void vectorComplexMultiplyAccumulate<2, float>(std::array<float, 2> (&acc)[2], const float &input, const Complex<float> (&weights)[2]) {
    const m128 vinput{.m128 = _mm_broadcast_ss(&input)};   // complex<float> is 2 floats, so we can broadcast it as a double // very ugly but works
    const m128 vweight{.m128d = _mm_load_pd(reinterpret_cast<const double *>(&weights[0]))}; // complex<float> is 2 floats, so we can load it as a float array // very ugly but works
    m128       vacc{.m128d = _mm_load_pd(reinterpret_cast<double *>(&acc[0]))};              // complex<float> is 2 floats, so we can load it as a float array // very ugly but works
    vacc.m128 = _mm_fmadd_ps(vweight.m128, vinput.m128, vacc.m128);
    _mm_store_pd(reinterpret_cast<double *>(&acc[0][0]), vacc.m128d); // at this point lets also store using the m256 union // very ugly but works
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
__attribute__((always_inline)) inline void vectorRealOnlyMultiplyAccumulate<4, float>(std::array<float, 2> (&acc)[4], const Complex<float> &input, const Complex<float> (&weights)[4]) {
    // const m256 vinput{{
    //      std::real(input), std::imag(input),
    //      std::real(input), std::imag(input),
    //      std::real(input), std::imag(input),
    //      std::real(input), std::imag(input)}};    // gets incorrectly
    // loaded
    const m256 vinput{.m256d = _mm256_broadcast_sd(reinterpret_cast<const double *>(&input))}; // complex<float> is 2 floats, so we can broadcast it as a double // very ugly but works

    // const m256 vweight{{std::real(weights[0]), std::imag(weights[0]), // also dosent work bc compiler sees it as something else and wants to blend it
    //                     std::real(weights[1]), std::imag(weights[1]), //
    //                     std::real(weights[2]), std::imag(weights[2]), //
    //                     std::real(weights[3]), std::imag(weights[3])}};
    const m256 vweight{.m256d = _mm256_load_pd(reinterpret_cast<const double *>(&weights[0]))}; // complex<float> is 2 floats, so we can load it as a float array // very ugly but works

    // m256 vacc{{acc[0][0], acc[0][1], // 
    //            acc[1][0], acc[1][1], //
    //            acc[2][0], acc[2][1], //
    //            acc[3][0], acc[3][1]}};
    m256 vacc{.m256d = _mm256_load_pd(reinterpret_cast<double *>(&acc[0]))}; // complex<float> is 2 floats, so we can load it as a float array // very ugly but works
    vacc.m256 = _mm256_fmadd_ps(vweight.m256, vinput.m256, vacc.m256);

    // #pragma GCC unroll(65534)
    //     for (std::size_t kernel_parallel = 0; kernel_parallel < 4; kernel_parallel++) {
    //         acc[kernel_parallel]=std::array<float,2>{vacc.data.data[2*kernel_parallel],vacc.data.data[2*kernel_parallel + 1]};
    //     }
    _mm256_store_pd(reinterpret_cast<double *>(&acc[0]), vacc.m256d); // at this point lets also store using the m256 union // very ugly but works
}

template <>
__attribute__((always_inline)) inline void vectorRealOnlyMultiplyAccumulate<2, float>(std::array<float, 2> (&acc)[2], const Complex<float> &input, const Complex<float> (&weights)[2]) {
    const m128 vinput{{input.real(), input.imag(), input.real(), input.imag()}};
    const m128 vweight{{weights[0].real(), weights[0].imag(), //
                        weights[1].real(), weights[1].imag()}};

    m128 vacc{{acc[0][0], acc[0][1], //
               acc[1][0], acc[1][1]}};
    vacc.m128 = _mm_fmadd_ps(vweight.m128, vinput.m128, vacc.m128);
#pragma GCC unroll(65534)
    for (std::size_t kernel_parallel = 0; kernel_parallel < 2; kernel_parallel++) {
        acc[kernel_parallel] = std::array<float, 2>{vacc.data.data[2 * kernel_parallel], vacc.data.data[2 * kernel_parallel + 1]};
    }
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

        constexpr auto primary_output_channels      = output_channels - output_channels % 8;
        constexpr auto primary_output_channels_rest = output_channels % 8;

        const auto input_permuted   = permute<"bi">(input);
        auto       output_permuted  = permute<"bo">(output);
        const auto weights_permuted = permute<"boi">(weights);

        // Perfect unrolling for outputchannels % 8 == 0
        if constexpr (primary_output_channels > 0) {
#pragma GCC unroll(65534)
            for (Dim_size_t batch_pos = 0; batch_pos < batch_channels; batch_pos += 1) {
#pragma GCC unroll(65534)
                for (Dim_size_t output_pos = 0; output_pos < primary_output_channels; output_pos += 8) {
                    float local_accumulation[8] = {output_permuted.at(batch_pos, output_pos),     output_permuted.at(batch_pos, output_pos + 1), output_permuted.at(batch_pos, output_pos + 2),
                                                   output_permuted.at(batch_pos, output_pos + 3), output_permuted.at(batch_pos, output_pos + 4), output_permuted.at(batch_pos, output_pos + 5),
                                                   output_permuted.at(batch_pos, output_pos + 6), output_permuted.at(batch_pos, output_pos + 7)};
#pragma GCC unroll(65534)
                    for (Dim_size_t input_pos = 0; input_pos < input_channels; input_pos += 1) {
                        const float local_input      = input_permuted.at(batch_pos, input_pos);
                        const float local_weights[8] = {weights_permuted.at(batch_pos, output_pos, input_pos),     weights_permuted.at(batch_pos, output_pos + 1, input_pos),
                                                        weights_permuted.at(batch_pos, output_pos + 2, input_pos), weights_permuted.at(batch_pos, output_pos + 3, input_pos),
                                                        weights_permuted.at(batch_pos, output_pos + 4, input_pos), weights_permuted.at(batch_pos, output_pos + 5, input_pos),
                                                        weights_permuted.at(batch_pos, output_pos + 6, input_pos), weights_permuted.at(batch_pos, output_pos + 7, input_pos)};
                        vectorMultiplyAccumulate<8, float>(local_accumulation, local_input, local_weights);
                    }
#pragma GCC unroll(65534)
                    for (Dim_size_t i = 0; i < 8; i++) {
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

        constexpr auto primary_output_channels      = output_channels - output_channels % 4;
        constexpr auto primary_output_channels_rest = output_channels % 4;

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
                for (Dim_size_t output_pos = 0; output_pos < primary_output_channels; output_pos += 4) {
                    std::array<float, 2> local_accumulation[4] = {output_permuted.at(batch_pos, output_pos), output_permuted.at(batch_pos, output_pos + 1),
                                                                  output_permuted.at(batch_pos, output_pos + 2), output_permuted.at(batch_pos, output_pos + 3)};
#pragma GCC unroll(65534)
                    for (Dim_size_t input_pos = 0; input_pos < input_channels; input_pos += 1) {
                        const float          local_input      = input_permuted.at(batch_pos, input_pos);
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

        constexpr auto primary_output_channels      = output_channels - output_channels % 4;
        constexpr auto primary_output_channels_rest = output_channels % 4;

        const auto input_permuted   = permute<"bi">(input);
        auto       output_permuted  = permute<"bo">(output);
        const auto weights_permuted = permute<"boi">(weights);
        // std::cout << "input channels: " << input_channels << ", output channels: " << output_channels << ", batch channels: " << batch_channels << std::endl;

        // Perfect unrolling for outputchannels % 4 == 0
        if constexpr (primary_output_channels > 0) {
#pragma GCC unroll(65534)
            for (Dim_size_t batch_pos = 0; batch_pos < batch_channels; batch_pos += 1) {
#pragma GCC unroll(65534)
                for (Dim_size_t output_pos = 0; output_pos < primary_output_channels; output_pos += 4) {
                    std::array<float, 2> local_accumulation[4] = {output_permuted.at(batch_pos, output_pos), output_permuted.at(batch_pos, output_pos + 1),
                                                                  output_permuted.at(batch_pos, output_pos + 2), output_permuted.at(batch_pos, output_pos + 3)};
#pragma GCC unroll(65534)
                    for (Dim_size_t input_pos = 0; input_pos < input_channels; input_pos += 1) {
                        const Complex<float> local_input      = input_permuted.at(batch_pos, input_pos);
                        const Complex<float> local_weights[4] = {weights_permuted.at(batch_pos, output_pos, input_pos), weights_permuted.at(batch_pos, output_pos + 1, input_pos),
                                                                 weights_permuted.at(batch_pos, output_pos + 2, input_pos), weights_permuted.at(batch_pos, output_pos + 3, input_pos)};
                        vectorRealOnlyMultiplyAccumulate<4, float>(local_accumulation, local_input, local_weights);
                    }
#pragma GCC unroll(65534)
                    for (Dim_size_t i = 0; i < 4; i++) {
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
