#pragma once
#include "./Matrix.hpp"
#include <type_traits>
#include <utility>

/*
Helper Functions for Matrix Operations
*/
template <typename Type, DimensionOrder Order, typename indxs, typename op, typename...>
struct Matrix_unroll_helper;

template <typename Type, DimensionOrder Order, Dim_size_t M1, typename op, typename... Matries>
struct Matrix_unroll_helper<Type, Order, std::integer_sequence<Dim_size_t, M1>, op, Matries...> {
    __attribute__((always_inline)) inline static void unroll(const op &lambda, Matrix<Type, Order, M1> &ret, const Matries &...A) {
        for (Dim_size_t i1 = 0; i1 < M1; i1++)
            lambda(ret.at(i1), A.at(i1)...);
    }
};

template <typename Type, DimensionOrder Order, Dim_size_t M1, Dim_size_t M2, typename op, typename... Matries>
struct Matrix_unroll_helper<Type, Order, std::integer_sequence<Dim_size_t, M1, M2>, op, Matries...> {
    __attribute__((always_inline)) inline static void unroll(const op &lambda, Matrix<Type, Order, M1, M2> &ret, const Matries &...A) {
        for (Dim_size_t i1 = 0; i1 < M1; i1++)
            for (Dim_size_t i2 = 0; i2 < M2; i2++)
                lambda(ret.at(i1, i2), (A.at(i1, i2))...);
    }
};

template <typename Type, DimensionOrder Order, Dim_size_t M1, Dim_size_t M2, Dim_size_t M3, typename op, typename... Matries>
struct Matrix_unroll_helper<Type, Order, std::integer_sequence<Dim_size_t, M1, M2, M3>, op, Matries...> {
    __attribute__((always_inline)) inline static void unroll(const op &lambda, Matrix<Type, Order, M1, M2, M3> &ret, const Matries &...A) {
        for (Dim_size_t i1 = 0; i1 < M1; i1++)
            for (Dim_size_t i2 = 0; i2 < M2; i2++)
                for (Dim_size_t i3 = 0; i3 < M3; i3++)
                    lambda(ret.at(i1, i2, i3), (A.at(i1, i2, i3))...);
    }
};

template <typename Type, DimensionOrder Order, Dim_size_t M1, Dim_size_t M2, Dim_size_t M3, Dim_size_t M4, typename op, typename... Matries>
struct Matrix_unroll_helper<Type, Order, std::integer_sequence<Dim_size_t, M1, M2, M3, M4>, op, Matries...> {
    __attribute__((always_inline)) inline static void unroll(const op &lambda, Matrix<Type, Order, M1, M2, M3, M4> &ret, const Matries &...A) {
        for (Dim_size_t i1 = 0; i1 < M1; i1++)
            for (Dim_size_t i2 = 0; i2 < M2; i2++)
                for (Dim_size_t i3 = 0; i3 < M3; i3++)
                    for (Dim_size_t i4 = 0; i4 < M4; i4++)
                        lambda(ret.at(i1, i2, i3, i4), (A.at(i1, i2, i3, i4))...);
    }
};

/*
Elemtwise Lambda Operations
*/
template <typename Type>
auto add_lambda = [](Type &ret, const Type &a, const Type &b) -> void { ret = a + b; };

template <typename Type>
auto sub_lambda = [](Type &ret, const Type &a, const Type &b) -> void { ret = a - b; };

template <typename Type>
auto mul_lambda = [](Type &ret, const Type &a, const Type &b) -> void { ret = a * b; };

template <typename Type>
auto div_lambda = [](Type &ret, const Type &a, const Type &b) -> void { ret = a / b; };

template <typename Type>
auto acc_lambda = [](Type &ret, const Type &a) -> void { ret += a; };

template <typename Type>
auto acc_neg_lambda = [](Type &ret, const Type &a) -> void { ret -= a; };

template <typename Type>
auto acc_scalar_lambda = [](Type scalar_mul) { return [=](Type &ret, const Type &a) { ret += scalar_mul * a; }; };
template <typename Type>
auto acc_neg_scalar_lambda = [](Type scalar_mul) { return [=](Type &ret, const Type &a) { ret -= scalar_mul * a; }; };

template <typename Type>
auto acc_div_scalar_lambda = [](Type scalar_div) { return [=](Type &ret, const Type &a) { ret += a / scalar_div; }; };

/*
Matrix Operations
*/
// ret=A+B
template <typename Type, DimensionOrder Order, Dim_size_t... dims>
__attribute__((always_inline)) inline Matrix<Type, Order, dims...> Matrix_Add(const Matrix<Type, Order, dims...> &A, const Matrix<Type, Order, dims...> &B) {
    Matrix<Type, Order, dims...> ret{};
    Matrix_unroll_helper<Type, Order, std::integer_sequence<Dim_size_t, dims...>, decltype(add_lambda<Type>), Matrix<Type, Order, dims...>, Matrix<Type, Order, dims...>>::unroll(add_lambda<Type>, ret,
                                                                                                                                                                                  A, B);
    return ret;
};

// ret=A-B
template <typename Type, DimensionOrder Order, Dim_size_t... dims>
__attribute__((always_inline)) inline Matrix<Type, Order, dims...> Matrix_Sub(const Matrix<Type, Order, dims...> &A, const Matrix<Type, Order, dims...> &B) {
    Matrix<Type, Order, dims...> ret{};
    Matrix_unroll_helper<Type, Order, std::integer_sequence<Dim_size_t, dims...>, decltype(sub_lambda<Type>), Matrix<Type, Order, dims...>, Matrix<Type, Order, dims...>>::unroll(sub_lambda<Type>, ret,
                                                                                                                                                                                  A, B);
    return ret;
};

// ret= elemtwise A*B
template <typename Type, DimensionOrder Order, Dim_size_t... dims>
__attribute__((always_inline)) inline Matrix<Type, Order, dims...> Matrix_Mul(const Matrix<Type, Order, dims...> &A, const Matrix<Type, Order, dims...> &B) {
    Matrix<Type, Order, dims...> ret{};
    Matrix_unroll_helper<Type, Order, std::integer_sequence<Dim_size_t, dims...>, decltype(mul_lambda<Type>), Matrix<Type, Order, dims...>, Matrix<Type, Order, dims...>>::unroll(mul_lambda<Type>, ret,
                                                                                                                                                                                  A, B);
    return ret;
};

// ret = elemtwise A/B
template <typename Type, DimensionOrder Order, Dim_size_t... dims>
__attribute__((always_inline)) inline Matrix<Type, Order, dims...> Matrix_Div(const Matrix<Type, Order, dims...> &A, const Matrix<Type, Order, dims...> &B) {
    Matrix<Type, Order, dims...> ret{};
    Matrix_unroll_helper<Type, Order, std::integer_sequence<Dim_size_t, dims...>, decltype(div_lambda<Type>), Matrix<Type, Order, dims...>, Matrix<Type, Order, dims...>>::unroll(div_lambda<Type>, ret,
                                                                                                                                                                                  A, B);
    return ret;
};

// ret=scalar*A
template <typename Type, DimensionOrder Order, Dim_size_t... dims>
__attribute__((always_inline)) inline Matrix<Type, Order, dims...> Matrix_scalar_Mul(Type scalar, const Matrix<Type, Order, dims...> &A) {
    Matrix<Type, Order, dims...> ret{};
    Matrix_unroll_helper<Type, Order, std::integer_sequence<Dim_size_t, dims...>, decltype(acc_scalar_lambda<Type>(scalar)), Matrix<Type, Order, dims...>>::unroll(acc_scalar_lambda<Type>(scalar), ret,
                                                                                                                                                                   A);
    return ret;
};

// ret=A/scalar
template <typename Type, DimensionOrder Order, Dim_size_t... dims>
__attribute__((always_inline)) inline Matrix<Type, Order, dims...> Matrix_scalar_Div(const Matrix<Type, Order, dims...> &A, Type scalar) {
    Matrix<Type, Order, dims...> ret{};
    Matrix_unroll_helper<Type, Order, std::integer_sequence<Dim_size_t, dims...>, decltype(acc_div_scalar_lambda<Type>(scalar)), Matrix<Type, Order, dims...>>::unroll(
            acc_div_scalar_lambda<Type>(scalar), ret, A);
    return ret;
};

// A=A+B
template <typename Type, DimensionOrder Order, Dim_size_t... dims>
__attribute__((always_inline)) inline void Matrix_Acc(Matrix<Type, Order, dims...> &A, const Matrix<Type, Order, dims...> &B) {
    Matrix_unroll_helper<Type, Order, std::integer_sequence<Dim_size_t, dims...>, decltype(acc_lambda<Type>), Matrix<Type, Order, dims...>>::unroll(acc_lambda<Type>, A, B);
};

// A=A-scalar*B
template <typename Type, DimensionOrder Order, Dim_size_t... dims>
__attribute__((always_inline)) inline void Matrix_scN_Acc(float scalar, Matrix<Type, Order, dims...> &A, const Matrix<Type, Order, dims...> &B) {
    Matrix_unroll_helper<Type, Order, std::integer_sequence<Dim_size_t, dims...>, decltype(acc_neg_scalar_lambda<Type>(scalar)), Matrix<Type, Order, dims...>>::unroll(
            acc_neg_scalar_lambda<Type>(scalar), A, B);
};
