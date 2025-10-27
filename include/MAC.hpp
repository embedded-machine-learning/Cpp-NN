#pragma once
#include "./types/Complex.hpp"

#include "Matrix.hpp"
#include "MatrixOperations.hpp"
#include <concepts>
#include <tuple>
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
        const auto input_broadcasted        = broadcast<"o", {WeightMatrixType::dimensions[WeightMatrixType::order.indexOf('o')]}>(input);
        auto       accumulation_broadcasted = permute<OperationOrder>(broadcast<"i", {WeightMatrixType::dimensions[WeightMatrixType::order.indexOf('i')]}>(accumulation));
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

// Dummy MAC Operation that only fails compilation if it is used
template <typename InputType, typename WeightType, typename BiasType>
struct NonMACOperation {

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
    __attribute__((always_inline)) inline static void op([[maybe_unused]] AccumulationMatrixType &accumulation,
                                                         [[maybe_unused]] const InputMatrixType  &input,
                                                         [[maybe_unused]] const WeightMatrixType &weights) {
        static_assert(sizeof(AccumulationMatrixType) == 0, "NonMACOperation cannot be used for MAC operations");
    }

    constexpr static auto pre_processing = [](const BiasType_ &bias) -> AccumulationType_ { return static_cast<AccumulationType_>(bias); };

    constexpr static auto post_processing = [](const AccumulationType_ &acc) -> BiasType_ { return static_cast<BiasType_>(acc); };
};

// Tuple of Mac Operations for multiple MACs at once
template <typename Is, template <typename InputType, typename WeightType, typename BiasType> class... MACOperations>
struct MACOperationTuple_helper;

template <template <typename InputType, typename WeightType, typename BiasType> class... MACOperations, std::size_t... Is>
struct MACOperationTuple_helper<std::index_sequence<Is...>, MACOperations...> {
    template <typename InputType, typename WeightType, typename BiasType>
    struct FuzedMACOperation;

    template <typename... InputTypes, typename... WeightTypes, typename... BiasTypes>
    struct FuzedMACOperation<std::tuple<InputTypes...>, std::tuple<WeightTypes...>, std::tuple<BiasTypes...>> {
        using InputType_  = std::tuple<InputTypes...>;
        using WeightType_ = std::tuple<WeightTypes...>;
        using BiasType_   = std::tuple<BiasTypes...>;

        using MACOperations_ = std::tuple<MACOperations<InputTypes, WeightTypes, BiasTypes>...>;

        using AccumulationType_tmp = std::tuple<decltype(std::declval<std::tuple_element_t<Is, InputType_>>() * std::declval<std::tuple_element_t<Is, WeightType_>>() +
                                                         std::declval<std::tuple_element_t<Is, BiasType_>>())...>;
        using AccumulationType_    = std::tuple<decltype(std::declval<std::tuple_element_t<Is, AccumulationType_tmp>>() +
                                                      std::declval<std::tuple_element_t<Is, InputType_>>() * std::declval<std::tuple_element_t<Is, WeightType_>>())...>;

        constexpr static auto lambda = [](InputType_ &inputs, const WeightType_ &weights, const AccumulationType_ &accums) {
            // (MACOperations<std::tuple_element_t<Is, InputType_>, std::tuple_element_t<Is, WeightType_>, std::tuple_element_t<Is, BiasType_>>::lambda(std::get<Is>(accums), std::get<Is>(inputs),
            //                                                                                                                                          std::get<Is>(weights)),
            //  ...);
        };
        using LambdaType = decltype(lambda);

        template <IsMatrixType AccumulationMatrixTypes, IsMatrixType InputMatrixTypes, IsMatrixType WeightMatrixTypes, DimensionOrder OperationOrder = "boi">
            requires(std::is_same_v<typename std::remove_cvref_t<AccumulationMatrixTypes>::value_type, AccumulationType_> &&
                     std::is_same_v<typename std::remove_cvref_t<InputMatrixTypes>::value_type, InputType_> && std::is_same_v<typename std::remove_cvref_t<WeightMatrixTypes>::value_type, WeightType_>)
        __attribute__((always_inline)) inline static void op(AccumulationMatrixTypes &accumulations, const InputMatrixTypes &inputs, const WeightMatrixTypes &weights) {
            (std::tuple_element_t<Is, MACOperations_>::template op<std::tuple_element_t<Is, AccumulationMatrixTypes>, std::tuple_element_t<Is, InputMatrixTypes>, std::tuple_element_t<Is, WeightMatrixTypes>,
                                                          OperationOrder>(std::get<Is>(accumulations), std::get<Is>(inputs), std::get<Is>(weights)),
             ...);
        }

        constexpr static auto pre_processing = [](const BiasType_ &biases) -> AccumulationType_ {
            return AccumulationType_{
                    MACOperations<std::tuple_element_t<Is, InputType_>, std::tuple_element_t<Is, WeightType_>, std::tuple_element_t<Is, BiasType_>>::pre_processing(std::get<Is>(biases))...};
        };

        constexpr static auto post_processing = [](const AccumulationType_ &accs) -> BiasType_ {
            return BiasType_{MACOperations<std::tuple_element_t<Is, InputType_>, std::tuple_element_t<Is, WeightType_>, std::tuple_element_t<Is, BiasType_>>::post_processing(std::get<Is>(accs))...};
        };
    };
};

template <template <typename InputType, typename WeightType, typename BiasType> class... MACOperations>
using MACOperationTuple = MACOperationTuple_helper<std::make_index_sequence<sizeof...(MACOperations)>, MACOperations...>;

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
static_assert(IsMACOperation<NonMACOperation, float, float, float>, "NonMACOperation should be a valid MAC operation");
static_assert(IsMACOperation<MACOperationTuple<DefaultMACOperation, DefaultMACOperation>::FuzedMACOperation, std::tuple<float, float>, std::tuple<float, float>, std::tuple<float, float>>,
              "MACOperationTuple should be a valid MAC operation");
// static_assert(IsMACOperation<RealResultMACOperation, std::complex<float>,std::complex<float>,float>, "RealResultMACOperation should be a valid MAC operation");