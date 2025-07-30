#pragma once
#include "./types/Complex.hpp"

#include "Matrix.hpp"
#include "MatrixOperations.hpp"
#include <concepts>
#include <type_traits>

/* Specialized Lambda Operations*/
template <typename AccumulationType, typename InputType, typename WeightType>
constexpr auto multily_accumulate = [](AccumulationType &acc, const InputType &input, const WeightType &weight) { acc += input * weight; };

template <typename AccumulationType, typename InputType, typename WeightType>
constexpr auto multily_accumulate<std::array<AccumulationType, 2>, InputType, Complex<WeightType>> =
        [](std::array<AccumulationType, 2> &acc, const InputType &input, const Complex<WeightType> &weight) {
            acc[0] += input * weight.real();
            acc[1] += input * weight.imag();
        };

template <typename AccumulationType, typename InputType, typename WeightType>
constexpr auto real_only_multily_accumulate = [](AccumulationType &acc, const InputType &input, const WeightType &weight) { acc += input.real() * weight.real() - input.imag() * weight.imag(); };

template <typename AccumulationType, typename InputType, typename WeightType>
constexpr auto split_real_only_multily_accumulate = [](std::array<AccumulationType, 2> &acc, const InputType &input, const WeightType &weight) {
    acc[0] += input.real() * weight.real();
    acc[1] += input.imag() * weight.imag();
};

template <typename AccumulationType, typename InputType, typename WeightType, typename Lambda = decltype([]() {})>
struct OverrideOperation {
    template <IsMatrixType AccumulationMatrixType, IsMatrixType InputMatrixType, IsMatrixType WeightMatrixType, DimensionOrder OperationOrder = "boi">
        requires(std::is_same_v<typename std::remove_cvref_t<AccumulationMatrixType>::value_type, AccumulationType> &&
                 std::is_same_v<typename std::remove_cvref_t<InputMatrixType>::value_type, InputType> && std::is_same_v<typename std::remove_cvref_t<WeightMatrixType>::value_type, WeightType>)
    __attribute__((always_inline)) inline static void op(AccumulationMatrixType &accumulation, const InputMatrixType &input, const WeightMatrixType &weights) {
        const auto  input_broadcasted        = broadcast<"o", {WeightMatrixType::dimensions[WeightMatrixType::order.indexOf('o')]}>(input);
        auto        accumulation_broadcasted = permute<OperationOrder>(broadcast<"i", {WeightMatrixType::dimensions[WeightMatrixType::order.indexOf('i')]}>(accumulation));
        loopUnrolled(Lambda(), accumulation_broadcasted, input_broadcasted, weights);
    }
};

template <typename InputType, typename WeightType, typename BiasType>
struct DefaultMACOperation {

    using InputType_  = InputType;
    using WeightType_ = WeightType;
    using BiasType_   = BiasType;

    using AccumulationType_tmp = decltype(std::declval<InputType>() * std::declval<WeightType>() + std::declval<BiasType>());
    using AccumulationType_    = decltype(std::declval<AccumulationType_tmp>() + std::declval<InputType>() * std::declval<WeightType>());

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

    constexpr static auto pre_processing = [](const BiasType_ &bias) -> AccumulationType_ { return static_cast<AccumulationType_>(bias); };

    constexpr static auto post_processing = [](const AccumulationType_ &acc) -> BiasType_ { return static_cast<BiasType_>(acc); };
};

template <typename InputType, typename WeightType, typename BiasType>
struct RealResultMACOperation {
    static_assert(std::is_same_v<InputType, Complex<typename InputType::value_type>>, "InputType must be a complex type");
    static_assert(std::is_same_v<WeightType, Complex<typename WeightType::value_type>>, "WeightType must be a complex type");

    using InputType_  = InputType;
    using WeightType_ = WeightType;
    using BiasType_   = BiasType;

    using AccumulationType_tmp =
            decltype(std::declval<InputType>().real() * std::declval<WeightType>().real() - std::declval<InputType>().imag() * std::declval<WeightType>().imag() + std::declval<BiasType>());
    using AccumulationType_ = decltype(std::declval<AccumulationType_tmp>() + std::declval<InputType>().real() * std::declval<WeightType>().real() -
                                       std::declval<InputType>().imag() * std::declval<WeightType>().imag());

    constexpr static auto lambda = real_only_multily_accumulate<AccumulationType_, InputType_, WeightType_>;
    using LambdaType             = decltype(lambda);

    template <IsMatrixType AccumulationMatrixType, IsMatrixType InputMatrixType, IsMatrixType WeightMatrixType, DimensionOrder OperationOrder = "boi">
        requires(std::is_same_v<typename std::remove_cvref_t<AccumulationMatrixType>::value_type, AccumulationType_> &&
                 std::is_same_v<typename std::remove_cvref_t<InputMatrixType>::value_type, InputType_> && std::is_same_v<typename std::remove_cvref_t<WeightMatrixType>::value_type, WeightType_>)
    __attribute__((always_inline)) inline static void op(AccumulationMatrixType &output, const InputMatrixType &input, const WeightMatrixType &weights) {
        static_assert(AccumulationMatrixType::order.containsOnly("bo"), "AccumulationMatrixType must be 'ibo' sub-(Input, Batch, Output)");
        static_assert(InputMatrixType::order.containsOnly("ib"), "InputMatrixType must be 'ib' sub-(Input, Batch)");
        static_assert(WeightMatrixType::order.containsOnly("iob"), "WeightMatrixType must be 'io' sub-(Input, Output)");

        OverrideOperation<AccumulationType_, InputType_, WeightType_, decltype(lambda)>::op(output, input, weights);
    }

    constexpr static auto pre_processing = [](const BiasType_ &bias) -> AccumulationType_ { return static_cast<AccumulationType_>(bias); };

    constexpr static auto post_processing = [](const AccumulationType_ &acc) -> BiasType_ { return static_cast<BiasType_>(acc); };
};

/**************************************************************************************************************************************************
                                                      Validation of MACOperation Concept
***************************************************************************************************************************************************/
template <template <typename, typename, typename> class MACOperation, typename InputType, typename WeightType, typename BiasType>
concept IsMACOperation = requires(MACOperation<InputType, WeightType, BiasType> Operation) {
    typename MACOperation<InputType, WeightType, BiasType>::InputType_;
    typename MACOperation<InputType, WeightType, BiasType>::WeightType_;
    typename MACOperation<InputType, WeightType, BiasType>::BiasType_;
    typename MACOperation<InputType, WeightType, BiasType>::AccumulationType_;
    typename MACOperation<InputType, WeightType, BiasType>::LambdaType;
    &MACOperation<InputType, WeightType, BiasType>::lambda;
    &MACOperation<InputType, WeightType, BiasType>::pre_processing;
    &MACOperation<InputType, WeightType, BiasType>::post_processing;
    {
        MACOperation<InputType, WeightType, BiasType>::lambda(std::declval<typename MACOperation<InputType, WeightType, BiasType>::AccumulationType_ &>(), std::declval<const InputType &>(),
                                                              std::declval<const WeightType &>())
    } -> std::same_as<void>;
    {
        MACOperation<InputType, WeightType, BiasType>::op(std::declval<Matrix<typename MACOperation<InputType, WeightType, BiasType>::AccumulationType_, "bo", 1, 1> &>(),
                                                          std::declval<const Matrix<InputType, "bi", 1, 1> &>(), std::declval<const Matrix<WeightType, "boi", 1, 1, 1> &>())
    } -> std::same_as<void>;
    { MACOperation<InputType, WeightType, BiasType>::pre_processing(std::declval<const BiasType &>()) } -> std::same_as<typename MACOperation<InputType, WeightType, BiasType>::AccumulationType_>;
    {
        MACOperation<InputType, WeightType, BiasType>::post_processing(std::declval<const typename MACOperation<InputType, WeightType, BiasType>::AccumulationType_ &>())
    } -> std::same_as<typename MACOperation<InputType, WeightType, BiasType>::BiasType_>;
};

static_assert(IsMACOperation<DefaultMACOperation, float, float, float>, "DefaultMACOperation should be a valid MAC operation");
// static_assert(IsMACOperation<RealResultMACOperation, std::complex<float>,std::complex<float>,float>, "RealResultMACOperation should be a valid MAC operation");