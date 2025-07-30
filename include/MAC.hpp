#include "./Matrix.hpp"

#define __ARM_CORTEX_M4

#ifdef __ARM_FP16_FORMAT_IEEE
#define MAC_USE_FP16
#endif

#ifdef __ARM_FP16_FORMAT_ALTERNATIVE
#define MAC_USE_FP16
#endif

#ifdef __ARM_CORTEX_M4
#define MAC_USE_REAL_INSTRUCTIONS
#endif

template <Dim_size_t Unrolled, typename InputType, typename WeightType, typename AccumulationType, size_t... UnrollIndexes>
struct MAC {
    __attribute__((always_inline)) static inline AccumulationType OP(const InputType  input[Unrolled],
                                                                     const WeightType weights[Unrolled],
                                                                     AccumulationType acc,
                                                                     std::index_sequence<UnrollIndexes...>) noexcept {
        //     acc += ((static_cast<AccumulationType>(input[UnrollIndexes]) * static_cast<AccumulationType>(weights[UnrollIndexes])) + ...);
        acc = (acc + ... + (static_cast<const AccumulationType>(input[UnrollIndexes]) * static_cast<const AccumulationType>(weights[UnrollIndexes])));
        return acc;
    }
};

template <Dim_size_t Unrolled, typename Type, size_t... UnrollIndexes>
struct MAC<Unrolled, Complex<Type>, Complex<Type>, Type, UnrollIndexes...> {
    __attribute__((always_inline)) inline static Type OP(const Complex<Type> input[Unrolled], const Complex<Type> weights[Unrolled], Type acc, std::index_sequence<UnrollIndexes...>) noexcept {
        acc = (acc + ... + (input[UnrollIndexes].Mul_only_Real_result(weights[UnrollIndexes])));  // Fix computation
        return acc;
    }
};

template <Dim_size_t Unrolled, typename Type, size_t... UnrollIndexes>
struct MAC<Unrolled, Type, Complex<Type>, Complex<Type>, UnrollIndexes...> {
    __attribute__((always_inline)) inline static Complex<Type> OP(const Type input[Unrolled], const Complex<Type> weights[Unrolled], Complex<Type> acc, std::index_sequence<UnrollIndexes...>) noexcept {
        acc = (acc + ... + (weights[UnrollIndexes]*input[UnrollIndexes]));  // Fix computation
        return acc;
    }
};

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
        int32_t data;
    } data;
};

#ifdef MAC_USE_FP16
union SMID32_t_fp16 {
    uint32_t smid;

    struct {
        __fp16 data[2];
    } data;
};
#endif

#ifdef MAC_USE_REAL_INSTRUCTIONS
__attribute__((always_inline)) inline uint32_t __SMLAD(uint32_t op1, uint32_t op2, uint32_t op3) {
    uint32_t result;
    __asm volatile("smlad %0, %1, %2, %3" : "=r"(result) : "r"(op1), "r"(op2), "r"(op3));
    return result;
}

/* Fake Implementation of __SXTB16 to emulate behaviour*/
__attribute__((always_inline)) inline uint32_t __SXTB16_ROR0(const uint32_t op1) {
    uint32_t result;
    __asm("sxtb16 %0, %1" : "=r"(result) : "r"(op1));
    return result;
}

/* Fake Implementation of __SXTB16 to emulate behaviour*/
__attribute__((always_inline)) inline uint32_t __SXTB16_ROR8(const uint32_t op1) {
    uint32_t result;
    __asm("sxtb16 %0, %1, ror 8" : "=r"(result) : "r"(op1));
    return result;
}
#ifdef MAC_USE_FP16
// Bottom half fp16 to fp32
__attribute__((always_inline)) inline float __VCVTB(const uint32_t op1) {
    float result;
    __asm("vcvtb.f16.f32 %0, %1" : "=w"(result) : "w"(op1));
    return result;
}

// Top half fp16 to fp32
__attribute__((always_inline)) inline float __VCVTT(const uint32_t op1) {
    float result;
    __asm("vcvtt.f16.f32 %0, %1" : "=w"(result) : "w"(op1));
    return result;
}
#endif
#else
/* Fake Implementation of __SMLAD to emulate behaviour*/
__attribute__((always_inline)) inline uint32_t __SMLAD(uint32_t op1, uint32_t op2, uint32_t op3) {
    // uint32_t result;
    //   __asm volatile ("smlad %0, %1, %2, %3" : "=r" (result) : "r" (op1), "r" (op2), "r" (op3) );
    const SMID32_t_int16 smid_op1{.smid = op1};
    const SMID32_t_int16 smid_op2{.smid = op2};
    const SMID32_t_int32 smid_op3{.smid = op3};
    const int32_t        result = static_cast<int32_t>(smid_op1.data.data[0]) * static_cast<int32_t>(smid_op2.data.data[0]) +
                           static_cast<int32_t>(smid_op1.data.data[1]) * static_cast<int32_t>(smid_op2.data.data[1]) + smid_op3.data.data;

    const SMID32_t_int32 smid_result{.data = {result}};
    return smid_result.smid;
}

/* Fake Implementation of __SXTB16 to emulate behaviour*/
__attribute__((always_inline)) inline uint32_t __SXTB16_ROR0(const uint32_t op1) {
    // uint32_t result;
    // __asm ("sxtb16 %0, %1" : "=r" (result) : "r" (op1) );
    const SMID32_t_int8  smid_op1{.smid = op1};
    const SMID32_t_int16 smid_result{.data = {smid_op1.data.data[0], smid_op1.data.data[2]}};

    return smid_result.smid;
}

/* Fake Implementation of __SXTB16 to emulate behaviour*/
__attribute__((always_inline)) inline uint32_t __SXTB16_ROR8(const uint32_t op1) {
    // uint32_t result;
    // __asm ("sxtb16 %0, %1, ror 8" : "=r" (result) : "r" (op1) );
    const SMID32_t_int8  smid_op1{.smid = op1};
    const SMID32_t_int16 smid_result{.data = {smid_op1.data.data[1], smid_op1.data.data[3]}};
    return smid_result.smid;
}
#ifdef MAC_USE_FP16
// Bottom half fp16 to fp32
__attribute__((always_inline)) inline float __VCVTB(const uint32_t op1) {
    // float result;
    // __asm("vcvtb.f16.f32 %0, %1" : "=r"(result) : "r"(op1));
    SMID32_t_fp16 smid_op1{.smid = op1};
    return static_cast<float>(smid_op1.data.data[0]);
}

// Top half fp16 to fp32
__attribute__((always_inline)) inline float __VCVTT(const uint32_t op1) {
    // float result;
    // __asm("vcvtt.f16.f32 %0, %1" : "=r"(result) : "r"(op1));
    SMID32_t_fp16 smid_op1{.smid = op1};
    return static_cast<float>(smid_op1.data.data[1]);
}
#endif
#endif

template <size_t... UnrollIndexes>
struct MAC<2, int16_t, int16_t, int32_t, UnrollIndexes...> {
    __attribute__((always_inline)) inline static int32_t OP(const int16_t input[2], const int16_t weights[2], int32_t acc, std::index_sequence<UnrollIndexes...>) noexcept {
        const SMID32_t_int16 smid_intput{.data = {input[UnrollIndexes]...}};
        const SMID32_t_int16 smid_weights{.data = {weights[UnrollIndexes]...}};
        SMID32_t_int32       acc_smid{.data = {acc}};
        acc_smid.smid = __SMLAD(smid_intput.smid, smid_weights.smid, acc_smid.smid);
        return acc_smid.data.data;
    }
};

template <size_t... UnrollIndexes>
struct MAC<4, int8_t, int8_t, int32_t, UnrollIndexes...> {
    __attribute__((always_inline)) inline static int32_t OP(const int8_t input[4], const int8_t weights[4], int32_t acc, std::index_sequence<UnrollIndexes...>) noexcept {
        const SMID32_t_int8 smid_intput{.data = {input[UnrollIndexes]...}};
        const SMID32_t_int8 smid_weights{.data = {weights[UnrollIndexes]...}};
        SMID32_t_int32      acc_smid{.data = {acc}};
        const uint32_t      a = __SXTB16_ROR0(smid_intput.smid);
        const uint32_t      b = __SXTB16_ROR0(smid_weights.smid);
        acc_smid.smid         = __SMLAD(a, b, acc_smid.smid);
        const uint32_t c      = __SXTB16_ROR8(smid_intput.smid);
        const uint32_t d      = __SXTB16_ROR8(smid_weights.smid);
        acc_smid.smid         = __SMLAD(c, d, acc_smid.smid);
        return acc_smid.data.data;
    }
};

#ifdef MAC_USE_FP16
// template <Dim_size_t Unrolled, typename InputType, typename WeightType, typename AccumulationType, size_t... UnrollIndexes>
template <size_t... UnrollIndexes>
struct MAC<2, __fp16, __fp16, float, UnrollIndexes...> {
    __attribute__((always_inline)) inline static float OP(const __fp16 input[2], const __fp16 weights[2], float acc, std::index_sequence<UnrollIndexes...>) noexcept {

        const SMID32_t_fp16 smid_intput{.data = {input[UnrollIndexes]...}};
        // const SMID32_t_fp16 smid_intput{.smid = *(uint32_t*)(void*)input};
        const SMID32_t_fp16 smid_weights{.data = {weights[UnrollIndexes]...}};
        // const SMID32_t_fp16 smid_weights{.smid = *(uint32_t*)(void*)weights};
        float a = __VCVTB(smid_intput.smid);
        float b = __VCVTB(smid_weights.smid);
        acc += a * b;
        a = __VCVTT(smid_intput.smid);
        b = __VCVTT(smid_weights.smid);
        acc += a * b;

        return acc;
    }
};

// template <Dim_size_t Unrolled, typename InputType, typename WeightType, typename AccumulationType, size_t... UnrollIndexes>
template <size_t... UnrollIndexes>
struct MAC<2, float, __fp16, float, UnrollIndexes...> {
    __attribute__((always_inline)) inline static float OP(const float input[2], const __fp16 weights[2], float acc, std::index_sequence<UnrollIndexes...>) noexcept {

        // const SMID32_t_fp16 smid_weights{.data = {weights[UnrollIndexes]...}};
        // const SMID32_t_fp16 smid_weights{.smid = *(uint32_t*)(void*)weights};
        const SMID32_t_fp16 *smid_weights = reinterpret_cast<const SMID32_t_fp16 *>(weights);

        const float b = __VCVTB(smid_weights->smid);
        acc += input[0] * b;
        const float c = __VCVTT(smid_weights->smid);
        acc += input[1] * c;

        return acc;
    }
};
#endif
