#pragma once

#include "Matrix.hpp"
#include <cstddef>
// #include <iostream>
#include <tuple>
#include <type_traits>
#include <utility>

template <typename Func, IsMatrixType ReturnMatrixType, IsIndexType... Indices, IsPermutationalSame<ReturnMatrixType>... MatrixTypes, std::size_t... IndicesSeq>
    requires(sizeof...(Indices) < std::remove_cvref_t<ReturnMatrixType>::number_of_dimensions)
__attribute__((always_inline)) inline constexpr void loopHelper(
        Func &&func, std::tuple<Indices...> args, std::index_sequence<IndicesSeq...>, ReturnMatrixType &&returnMatrix, MatrixTypes &&...matrices) {
    constexpr Dim_size_t current_dimension = sizeof...(Indices);
    constexpr Dim_size_t current_size      = std::remove_cvref_t<ReturnMatrixType>::dimensions[current_dimension];
    for (Dim_size_t i = 0; i < current_size; i++) {
        loopHelper(std::forward<Func>(func), std::make_tuple(std::get<IndicesSeq>(args)..., i), std::make_index_sequence<sizeof...(Indices) + 1>(), std::forward<ReturnMatrixType &&>(returnMatrix),
                   std::forward<MatrixTypes &&>(matrices)...);
    }
}

template <typename Func, IsMatrixType ReturnMatrixType, IsIndexType... Indices, IsPermutationalSame<ReturnMatrixType>... MatrixTypes, std::size_t... IndicesSeq>
    requires(sizeof...(Indices) == std::remove_cvref_t<ReturnMatrixType>::number_of_dimensions)
__attribute__((always_inline)) inline constexpr void loopHelper(
        Func &&func, std::tuple<Indices...> args, std::index_sequence<IndicesSeq...>, ReturnMatrixType &&returnMatrix, MatrixTypes &&...matrices) {
    constexpr DimensionOrder return_order = std::remove_cvref_t<ReturnMatrixType>::order;
    std::forward<Func>(func)(returnMatrix.template at<return_order>(std::get<IndicesSeq>(args)...), matrices.template at<return_order>(std::get<IndicesSeq>(args)...)...);
}

template <typename Func, IsMatrixType ReturnMatrixType, IsMatrixType... MatrixTypes>
    requires(!(IsPermutationalSame<MatrixTypes, ReturnMatrixType> && ...)) // Ensure that not all matrices are permutationally same
__attribute__((always_inline)) inline constexpr void loop(Func &&, ReturnMatrixType &&, MatrixTypes &&...) {
    using ReturnMatrixTypeNoRef = std::remove_cvref_t<ReturnMatrixType>;
    // constexpr bool same_dimensions = ((ReturnMatrixTypeNoRef::dimensions == PermutedMatrix<ReturnMatrixTypeNoRef::order, std::remove_cvref_t<MatrixTypes>>::dimensions) && ...);    // breaks gcc
    // static_assert(same_dimensions,"Ensure that all Matrixes have the same dimensions sizes for the same named dimensions");

    static_assert((ReturnMatrixTypeNoRef::order.containsAll(std::remove_cvref_t<MatrixTypes>::order) && ...), "Ensure that all orders have the same NAMED DIMENSIONS");
}

template <typename Func, IsMatrixType ReturnMatrixType, IsMatrixType... MatrixTypes>
    requires((IsPermutationalSame<MatrixTypes, ReturnMatrixType> && ...)) // Ensure that all matrices are permutationally same
__attribute__((always_inline)) inline constexpr void loop(Func &&func, ReturnMatrixType &&returnMatrix, MatrixTypes &&...matrices) {
    loopHelper(std::forward<Func>(func), std::tuple<>(), std::make_index_sequence<0>(), std::forward<ReturnMatrixType &&>(returnMatrix), std::forward<const MatrixTypes>(matrices)...);
}

template <typename Func, IsMatrixType ReturnMatrixType, IsIndexType... Indices, IsPermutationalSame<ReturnMatrixType>... MatrixTypes, std::size_t... IndicesSeq>
    requires(sizeof...(Indices) < std::remove_cvref_t<ReturnMatrixType>::number_of_dimensions)
__attribute__((always_inline)) inline constexpr void loopHelperUnrolled(
        Func &&func, std::tuple<Indices...> args, std::index_sequence<IndicesSeq...>, ReturnMatrixType &&returnMatrix, MatrixTypes &&...matrices) {
    constexpr Dim_size_t current_dimension = sizeof...(Indices);
    constexpr Dim_size_t current_size      = std::remove_cvref_t<ReturnMatrixType>::dimensions[current_dimension];
#pragma GCC unroll(65534)
    for (Dim_size_t i = 0; i < current_size; i++) {
        loopHelperUnrolled(std::forward<Func>(func), std::make_tuple(std::get<IndicesSeq>(args)..., i), std::make_index_sequence<sizeof...(Indices) + 1>(),
                           std::forward<ReturnMatrixType &&>(returnMatrix), std::forward<MatrixTypes &&>(matrices)...);
    }
}

template <typename Func, IsMatrixType ReturnMatrixType, IsIndexType... Indices, IsPermutationalSame<ReturnMatrixType>... MatrixTypes, std::size_t... IndicesSeq>
    requires(sizeof...(Indices) == std::remove_cvref_t<ReturnMatrixType>::number_of_dimensions)
__attribute__((always_inline)) inline constexpr void loopHelperUnrolled(
        Func &&func, std::tuple<Indices...> args, std::index_sequence<IndicesSeq...>, ReturnMatrixType &&returnMatrix, MatrixTypes &&...matrices) {
    constexpr DimensionOrder return_order = std::remove_cvref_t<ReturnMatrixType>::order;
    std::forward<Func>(func)(returnMatrix.template at<return_order>(std::get<IndicesSeq>(args)...), matrices.template at<return_order>(std::get<IndicesSeq>(args)...)...);
}

template <typename Func, IsMatrixType ReturnMatrixType, IsMatrixType... MatrixTypes>
    requires(!(IsPermutationalSame<MatrixTypes, ReturnMatrixType> && ...)) // Ensure that not all matrices are permutationally same
__attribute__((always_inline)) inline constexpr void loopUnrolled(Func &&, ReturnMatrixType &&, MatrixTypes &&...) {
    using ReturnMatrixTypeNoRef = std::remove_cvref_t<ReturnMatrixType>;
    // constexpr bool same_dimensions = ((ReturnMatrixTypeNoRef::dimensions == PermutedMatrix<ReturnMatrixTypeNoRef::order, std::remove_cvref_t<MatrixTypes>>::dimensions) && ...);    // breaks gcc
    // static_assert(same_dimensions,"Ensure that all Matrixes have the same dimensions sizes for the same named dimensions");

    static_assert((ReturnMatrixTypeNoRef::order.containsAll(std::remove_cvref_t<MatrixTypes>::order) && ...), "Ensure that all orders have the same NAMED DIMENSIONS");
}

template <typename Func, IsMatrixType ReturnMatrixType, IsMatrixType... MatrixTypes>
    requires((IsPermutationalSame<MatrixTypes, ReturnMatrixType> && ...)) // Ensure that all matrices are permutationally same
__attribute__((always_inline)) inline constexpr void loopUnrolled(Func &&func, ReturnMatrixType &&returnMatrix, MatrixTypes &&...matrices) {
    loopHelperUnrolled(std::forward<Func>(func), std::tuple<>(), std::make_index_sequence<0>(), std::forward<ReturnMatrixType &&>(returnMatrix), std::forward<MatrixTypes &&>(matrices)...);
}

template <typename T, typename... Ts>
constexpr auto sum = [](T &a, const Ts &...args) { a += (args + ...); };

template <typename T, typename... Ts>
constexpr auto product = [](T &a, const Ts &...args) { a *= (args * ...); };


/**********************************************************************************************************************************************************************************
 *                                                                  Actual Operations
 ***********************************************************************************************************************************************************************************/

template <IsMatrixType BaseMatrixType, typename LambdaType = decltype([](const typename std::remove_cvref_t<BaseMatrixType>::value_type &x) { return x; })>
constexpr auto materialize(BaseMatrixType &&matrix, const LambdaType lambda = LambdaType()) {
    using Type = std::remove_cvref_t<decltype(lambda(std::declval<typename std::remove_cvref_t<BaseMatrixType>::value_type>()))>;
    OverrideTypeMatrix<BaseMatrixType &&, Type> result; // Same as materialized matrix, but with the type overridden to Type
    loop([=](Type &a, const typename std::remove_cvref_t<BaseMatrixType>::value_type b) { a = lambda(b); }, result, std::forward<BaseMatrixType>(matrix));
    return result;
}

template <IsMatrixType BaseMatrixType, typename LambdaType = decltype([](const typename std::remove_cvref_t<BaseMatrixType>::value_type &x) { return x; })>
__attribute__((always_inline)) constexpr inline auto materializeUnrolled(BaseMatrixType &&matrix, const LambdaType lambda = LambdaType()) {
    using Type = std::remove_cvref_t<decltype(lambda(std::declval<typename std::remove_cvref_t<BaseMatrixType>::value_type>()))>;
    OverrideTypeMatrix<BaseMatrixType &&, Type> result; // Same as materialized matrix, but with the type overridden to Type
    loopUnrolled([=](Type &a, const typename std::remove_cvref_t<BaseMatrixType>::value_type b) { a = lambda(b); }, result, std::forward<BaseMatrixType>(matrix));
    return result;
}

template <IsMatrixType... MatrixTypes>
constexpr auto matrixSum(MatrixTypes &&...matrices) {
    loop(sum<typename std::remove_cvref_t<MatrixTypes>::value_type...>, std::forward<MatrixTypes &&>(matrices)...);
}

template <IsMatrixType MatrixTypeA, IsMatrixType MatrixTypeB>
constexpr void matrixAssign(MatrixTypeA &&a, MatrixTypeB &&b) {
    static_assert(std::remove_cvref_t<MatrixTypeA>::order.containsOnly(std::remove_cvref_t<MatrixTypeB>::order), "Matrix orders must be compatible for assignment");
    static_assert(IsPermutationalSame<std::remove_cvref_t<MatrixTypeA>, std::remove_cvref_t<MatrixTypeB>>, "Matrix types must be permutationally same for assignment");
    loop([](auto &a, const auto b) { a = b; }, std::forward<MatrixTypeA>(a), std::forward<MatrixTypeB>(b));
}
/**********************************************************************************************************************************************************************************
 *                                                                  Aligned Storage
 ***********************************************************************************************************************************************************************************/

template <Dim_size_t Align, IsMatrixType... Matrixes>
struct AlignedMatrixCollection;

template <Dim_size_t Align, IsMatrixType CurrentMatrix, IsMatrixType... OtherMatrixes>
requires(sizeof(CurrentMatrix)%Align != 0 && sizeof...(OtherMatrixes) > 0)
struct AlignedMatrixCollection<Align, CurrentMatrix, OtherMatrixes...> {
    using Type = std::remove_cvref_t<CurrentMatrix>;
    constexpr static Dim_size_t align=Align;
    Type                                             data;
    char                                             padding[Align - (sizeof(Type) - 1) % Align - 1];
    AlignedMatrixCollection<Align, OtherMatrixes...> other_matrixes;

    constexpr AlignedMatrixCollection(CurrentMatrix &&matrix, OtherMatrixes &&...other_matrixes)
            : data(std::forward<CurrentMatrix>(matrix)), padding{}, other_matrixes(std::forward<OtherMatrixes>(other_matrixes)...) {
    }

    constexpr AlignedMatrixCollection() = default;

    // static_assert(sizeof...(OtherMatrixes)+1>0, "Disablöed for the time being, as it is not used in the codebase");

    template <std::size_t I>
    constexpr auto &get() const {
        if constexpr (I == 0) {
            return data;
        } else if constexpr (I < sizeof...(OtherMatrixes) + 1) {
            return other_matrixes.template get<I - 1>();
        } else {
            static_assert(I < sizeof...(OtherMatrixes) + 1, "Out of bounds access in AlignedMatrixCollection");
        }
    }
};

template <Dim_size_t Align, IsMatrixType CurrentMatrix>
requires(sizeof(CurrentMatrix)%Align != 0)
struct AlignedMatrixCollection<Align, CurrentMatrix> {
    using Type = std::remove_cvref_t<CurrentMatrix>;
    constexpr static Dim_size_t align=Align;
    Type data;
    char padding[Align - (sizeof(Type) - 1) % Align - 1];

    constexpr AlignedMatrixCollection(CurrentMatrix &&matrix) : data(std::forward<CurrentMatrix>(matrix)), padding{} {
    }

    constexpr AlignedMatrixCollection() = default;

    template <std::size_t I>
    constexpr auto &get() const {
        if constexpr (I == 0) {
            return data;
        } else {
            static_assert(I == 0, "Out of bounds access in AlignedMatrixCollection");
        }
    }
};


template <Dim_size_t Align, IsMatrixType CurrentMatrix, IsMatrixType... OtherMatrixes>
requires(sizeof(CurrentMatrix)%Align == 0 && sizeof...(OtherMatrixes) > 0)
struct AlignedMatrixCollection<Align, CurrentMatrix, OtherMatrixes...> {
    using Type = std::remove_cvref_t<CurrentMatrix>;
    constexpr static Dim_size_t align=Align;
    Type                                             data;
    AlignedMatrixCollection<Align, OtherMatrixes...> other_matrixes;

    constexpr AlignedMatrixCollection(CurrentMatrix &&matrix, OtherMatrixes &&...other_matrixes)
            : data(std::forward<CurrentMatrix>(matrix)), other_matrixes(std::forward<OtherMatrixes>(other_matrixes)...) {
    }

    constexpr AlignedMatrixCollection() = default;

    // static_assert(sizeof...(OtherMatrixes)+1>0, "Disablöed for the time being, as it is not used in the codebase");

    template <std::size_t I>
    constexpr auto &get() const {
        if constexpr (I == 0) {
            return data;
        } else if constexpr (I < sizeof...(OtherMatrixes) + 1) {
            return other_matrixes.template get<I - 1>();
        } else {
            static_assert(I < sizeof...(OtherMatrixes) + 1, "Out of bounds access in AlignedMatrixCollection");
        }
    }
};

template <Dim_size_t Align, IsMatrixType CurrentMatrix>
requires(sizeof(CurrentMatrix)%Align == 0)
struct AlignedMatrixCollection<Align, CurrentMatrix> {
    using Type = std::remove_cvref_t<CurrentMatrix>;
    constexpr static Dim_size_t align=Align;
    Type data;

    constexpr AlignedMatrixCollection(CurrentMatrix &&matrix) : data(std::forward<CurrentMatrix>(matrix)) {
    }

    constexpr AlignedMatrixCollection() = default;

    template <std::size_t I>
    constexpr auto &get() const {
        if constexpr (I == 0) {
            return data;
        } else {
            static_assert(I == 0, "Out of bounds access in AlignedMatrixCollection");
        }
    }
};

template <Dim_size_t Align, IsMatrixType... Matrixes>
constexpr auto makeAlignedMatrixCollection(Matrixes &&...matrices) {
    return AlignedMatrixCollection<Align, Matrixes...>(materialize(std::forward<Matrixes>(matrices))...);
}

template <Dim_size_t Align, IsMatrixType... Matrixes>
constexpr auto makeAlignedMatrixCollectionNoSafeGuard(Matrixes &&...matrices) {
    return AlignedMatrixCollection<Align, Matrixes...>(std::move(matrices)...);
}
template <typename>
struct is_instance_of_AlignedMatrixCollection : std::false_type {};

template <std::size_t align, typename... Args>
struct is_instance_of_AlignedMatrixCollection<AlignedMatrixCollection<align, Args...>> : std::true_type {};

template <typename T>
concept IsAlignedMatrixCollection = is_instance_of_AlignedMatrixCollection<std::remove_cvref_t<T>>::value;

namespace std {
template <Dim_size_t Align, IsMatrixType... Matrixes>
struct tuple_size<AlignedMatrixCollection<Align, Matrixes...>> : std::integral_constant<std::size_t, sizeof...(Matrixes)> {};

template <std::size_t I, Dim_size_t Align, IsMatrixType... Matrixes>
struct tuple_element<I, AlignedMatrixCollection<Align, Matrixes...>> {
    using type = std::tuple_element_t<I, std::tuple<Matrixes...>>;
};

template <std::size_t I, Dim_size_t Align, IsMatrixType... Matrixes>
auto &get(AlignedMatrixCollection<Align, Matrixes...> &v) {
    static_assert(I < sizeof...(Matrixes), "Out of bounds access in AlignedMatrixCollection");
    return v.template get<I>();
}

template <std::size_t I, Dim_size_t Align, IsMatrixType... Matrixes>
constexpr auto &get(const AlignedMatrixCollection<Align, Matrixes...> &v) {
    static_assert(I < sizeof...(Matrixes), "Out of bounds access in AlignedMatrixCollection");
    return v.template get<I>();
}
} // namespace std
