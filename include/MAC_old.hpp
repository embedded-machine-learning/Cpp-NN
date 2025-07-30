#include "./Matrix.hpp"

template <Dim_size_t Unrolled, typename InputType, typename WeightType, typename AccumulationType, size_t... UnrollIndexes>
struct MAC {
    __attribute__((always_inline)) static inline AccumulationType OP(const InputType  input[Unrolled],
                                                                     const WeightType weights[Unrolled],
                                                                     AccumulationType    acc,
                                                                     std::index_sequence<UnrollIndexes...>) noexcept {
        // const InputType  input[Unrolled]   = {LambdaInput(UnrollIndexes)...};
        // const WeightType weights[Unrolled] = {LambdaWeights(UnrollIndexes)...};
    	acc += ((static_cast<AccumulationType>(input[UnrollIndexes]) * static_cast<AccumulationType>(weights[UnrollIndexes]))+...);
        return acc;
    }
};


__attribute__((always_inline)) inline uint32_t __SMLAD (const uint32_t op1,const uint32_t op2, const uint32_t op3)
{
  uint32_t result;
  __asm volatile ("smlad %0, %1, %2, %3" : "=r" (result) : "r" (op1), "r" (op2), "r" (op3) );
  return(result);
}

__attribute__((always_inline)) inline uint32_t __SXTB16_ROR0 (const uint32_t op1)
{
  uint32_t result;
  __asm ("sxtb16 %0, %1" : "=r" (result) : "r" (op1) );
  return(result);
}

__attribute__((always_inline)) inline uint32_t __SXTB16_ROR8 (const uint32_t op1)
{
  uint32_t result;
  __asm ("sxtb16 %0, %1, ror 8" : "=r" (result) : "r" (op1) );
  return(result);
}


template <size_t... UnrollIndexes>
struct MAC<2, int16_t, int16_t, int32_t, UnrollIndexes...> {
    __attribute__((always_inline)) inline static int32_t OP(const int16_t input[2],  const int16_t weights[2], int32_t acc, std::index_sequence<UnrollIndexes...>) noexcept {
        union SMID32_t_16 {
            uint32_t smid;
            struct {
                int16_t data[2];
            } data;
        };

        union SMID32_t_32 {
            uint32_t smid;
            struct {
                int32_t data;
            } data;
        };

        const SMID32_t_16 smid_intput{.data = {input[UnrollIndexes]...}};
        const SMID32_t_16 smid_weights{.data = {weights[UnrollIndexes]...}};
		SMID32_t_32 acc_smid{.data={acc}};
		acc_smid.smid = __SMLAD(smid_intput.smid,smid_weights.smid,acc_smid.smid);
		return acc_smid.data.data;
    }
};

template <size_t... UnrollIndexes>
struct MAC<4, int8_t, int8_t, int32_t, UnrollIndexes...> {
    __attribute__((always_inline)) inline static int32_t OP(const int8_t input[4],  const int8_t weights[4], int32_t acc, std::index_sequence<UnrollIndexes...>) noexcept {
        union SMID32_t_8 {
            uint32_t smid;
            struct {
                int8_t data[4];
            } data;
        };

        union SMID32_t_32 {
            uint32_t smid;
            struct {
                int32_t data;
            } data;
        };
         const SMID32_t_8 smid_intput{.data = {input[UnrollIndexes]...}};
         const SMID32_t_8 smid_weights{.data = {weights[UnrollIndexes]...}};
		 SMID32_t_32 acc_smid{.data={acc}};
		 uint32_t a = __SXTB16_ROR0(smid_intput.smid);
		 uint32_t b = __SXTB16_ROR0(smid_weights.smid);
		 acc_smid.smid = __SMLAD(a,b,acc_smid.smid);
		 a = __SXTB16_ROR8(smid_intput.smid);
		 b = __SXTB16_ROR8(smid_weights.smid);
		 acc_smid.smid = __SMLAD(a,b,acc_smid.smid);
		 return acc_smid.data.data;
    }
};
