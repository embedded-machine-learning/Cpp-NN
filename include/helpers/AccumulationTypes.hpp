#pragma once
#include <cstdint>
#include "Complex.hpp"
#include "Benchmark.hpp"

#ifndef ACCUMULATION_FLOAT_HIGH_PRESCISION
#define ACCUMULATION_FLOAT_HIGH_PRESCISION false
#endif

#ifdef __ARM_FP16_FORMAT_IEEE
#define ACCUMULATION_USE_FP16
#endif

#ifdef __ARM_FP16_FORMAT_ALTERNATIVE
#define ACCUMULATION_USE_FP16
#endif

template<typename Input, typename Weight>
struct AccumulationType_helper_struct{
    using type = void;
};
template<typename Input, typename Weight>
struct AccumulationType_helper_struct<Complex<Input>, Complex<Weight>>{
    using type = Complex<typename AccumulationType_helper_struct<Input, Weight>::type>;
};
template<typename Input, typename Weight>
struct AccumulationType_helper_struct<Complex<Input>, Weight>{
    using type = Complex<typename AccumulationType_helper_struct<Input, Weight>::type>;
};
template<typename Input, typename Weight>
struct AccumulationType_helper_struct<Input, Complex<Weight>>{
    using type = Complex<typename AccumulationType_helper_struct<Input, Weight>::type>;
};
template<typename Input, typename Weight>
struct AccumulationType_helper_struct<helpers::Benchmark::TypeInstance<Input>, helpers::Benchmark::TypeInstance<Weight>>{
    using type = helpers::Benchmark::TypeInstance<typename AccumulationType_helper_struct<Input, Weight>::type>;
};
#if ACCUMULATION_FLOAT_HIGH_PRESCISION
template<>
struct AccumulationType_helper_struct<float, float>{
    using type = double;
};
#else
template<>
struct AccumulationType_helper_struct<float, float>{
    using type = float;
};
template<>
struct AccumulationType_helper_struct<double, double>{
    using type = double;
};
#endif

#ifdef ACCUMULATION_USE_FP16
template<>
struct AccumulationType_helper_struct<__fp16,__fp16>{
    using type = float;
};
template<>
struct AccumulationType_helper_struct<float,__fp16>{
    using type = float;
};
#endif


template<>
struct AccumulationType_helper_struct<int8_t, int8_t>{
    using type = int32_t;
};
template<>
struct AccumulationType_helper_struct<int16_t, int16_t>{
    using type = int32_t;
};
// For Testing
template<>
struct AccumulationType_helper_struct<int8_t,float>{
    using type = float;
};
// For Testing
template<>
struct AccumulationType_helper_struct<uint8_t,float>{
    using type = float;
};


template<typename Input, typename Weight>
using AccumulationType_helper = typename AccumulationType_helper_struct<Input, Weight>::type;


template<typename Input, typename Weight, typename Accumulation>
constexpr bool isAccumulationTypeValid = std::is_same<Accumulation, AccumulationType_helper<Input, Weight>>::value;

template<typename Input, typename Weight>
constexpr bool isAccumulationTypeSupported = !std::is_same<void, AccumulationType_helper<Input, Weight>>::value;



// Special real Output case
template<typename Type>
constexpr bool isAccumulationTypeValid<Complex<Type>,Complex<Type>,Type> = std::is_same<Type, AccumulationType_helper<Type, Type>>::value;
// Special real Input case
template<typename Type>
constexpr bool isAccumulationTypeValid<Type,Complex<Type>,Complex<Type>> = std::is_same<Type, AccumulationType_helper<Type, Type>>::value;
