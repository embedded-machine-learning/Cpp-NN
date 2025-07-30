#pragma once

#include "../../Matrix.hpp"
#include "../../helpers/AccumulationTypes.hpp"

namespace functions {
/*
================================================================================================================================================
                                                        Flatten
================================================================================================================================================
*/

template <DimensionOrder InOrder, DimensionOrder OutOrder, Dim_size_t M1_1, Dim_size_t M1_2, Dim_size_t M1_3, Dim_size_t M1_4, typename Type = float>
__attribute__((always_inline)) inline void Flatten(const Matrix<Type, InOrder, M1_1, M1_2, M1_3, M1_4> &Input, Matrix<Type, OutOrder, M1_1, M1_2 * M1_3 * M1_4> &Out) noexcept {
    for (Dim_size_t i1_1 = 0; i1_1 < M1_1; i1_1++)
        for (Dim_size_t i1_2 = 0; i1_2 < M1_2; i1_2++)
            for (Dim_size_t i1_3 = 0; i1_3 < M1_3; i1_3++)
                for (Dim_size_t i1_4 = 0; i1_4 < M1_4; i1_4++)
                    Out.data[i1_1][i1_2 * M1_3 * M1_4 + i1_3 * M1_4 + i1_4] = Input.data[i1_1][i1_2][i1_3][i1_4];
}

template <Dim_size_t M1_1, Dim_size_t M1_2, Dim_size_t M1_3, Dim_size_t M1_4, typename Type = float>
__attribute__((always_inline)) inline Matrix<Type, DimensionOrder::D2_Batch_Channel, M1_1, M1_2 * M1_3 * M1_4> Flatten(
        const Matrix<Type, DimensionOrder::D4_Batch_Channel_Width_Height, M1_1, M1_2, M1_3, M1_4> &Input) noexcept {
    Matrix<Type, DimensionOrder::D2_Batch_Channel, M1_1, M1_2 * M1_3 * M1_4> out;
    Flatten(Input, out);
    return out;
}

template <DimensionOrder InOrder, DimensionOrder OutOrder, Dim_size_t M1_1, Dim_size_t M1_2, Dim_size_t M1_3, typename Type = float>
__attribute__((always_inline)) inline void Flatten(const Matrix<Type, InOrder, M1_1, M1_2, M1_3> &Input, Matrix<Type, OutOrder, M1_1, M1_2 * M1_3> &Out) noexcept {
    for (Dim_size_t i1_1 = 0; i1_1 < M1_1; i1_1++)
        for (Dim_size_t i1_2 = 0; i1_2 < M1_2; i1_2++)
            for (Dim_size_t i1_3 = 0; i1_3 < M1_3; i1_3++)
                Out.data[i1_1][i1_2 * M1_3 + i1_3] = Input.data[i1_1][i1_2][i1_3];
}

template <Dim_size_t M1_1, Dim_size_t M1_2, Dim_size_t M1_3, typename Type = float>
__attribute__((always_inline)) inline Matrix<Type, DimensionOrder::D2_Batch_Channel, M1_1, M1_2 * M1_3> Flatten(
        const Matrix<Type, DimensionOrder::D3_Batch_Channel_Width, M1_1, M1_2, M1_3> &Input) noexcept {
    Matrix<Type, DimensionOrder::D2_Batch_Channel, M1_1, M1_2 * M1_3> out;
    Flatten(Input, out);
    return out;
}

/*
================================================================================================================================================
                                                        AdaptiveAveragePool
================================================================================================================================================
*/

template <DimensionOrder InOrder, DimensionOrder OutOrder, Dim_size_t M1_1, Dim_size_t M1_2, Dim_size_t M1_3, typename Type = float>
__attribute__((always_inline)) inline void AdaptiveAveragePool(const Matrix<Type, InOrder, M1_1, M1_2, M1_3> &Input, Matrix<Type, OutOrder, M1_1, M1_2> &Out) {
    for (Dim_size_t i1_1 = 0; i1_1 < M1_1; i1_1++)
        for (Dim_size_t i1_2 = 0; i1_2 < M1_2; i1_2++) {
            Type sum{static_cast<Type>(0)};
            for (Dim_size_t i1_3 = 0; i1_3 < M1_3; i1_3++)
                sum += Input.data[i1_1][i1_2][i1_3];
            Out.data[i1_1][i1_2] = sum / M1_3;
        }
}

template <Dim_size_t M1_1, Dim_size_t M1_2, Dim_size_t M1_3, typename Type = float>
__attribute__((always_inline)) inline Matrix<Type, DimensionOrder::D2_Batch_Channel, M1_1, M1_2> AdaptiveAveragePool(
        const Matrix<Type, DimensionOrder::D3_Batch_Channel_Width, M1_1, M1_2, M1_3> &Input) {
    Matrix<Type, DimensionOrder::D2_Batch_Channel, M1_1, M1_2> out;
    AdaptiveAveragePool(Input, out);
    return out;
}

template <DimensionOrder InOrder, DimensionOrder OutOrder, Dim_size_t M1_1, Dim_size_t M1_2, Dim_size_t M1_3, Dim_size_t M1_4, typename Type = float>
__attribute__((always_inline)) inline void AdaptiveAveragePool(const Matrix<Type, InOrder, M1_1, M1_2, M1_3> &Input, Matrix<Type, OutOrder, M1_1, M1_2> &Out) {
    for (Dim_size_t i1_1 = 0; i1_1 < M1_1; i1_1++)
        for (Dim_size_t i1_2 = 0; i1_2 < M1_2; i1_2++) {
            Type sum{static_cast<Type>(0)};
            for (Dim_size_t i1_3 = 0; i1_3 < M1_3; i1_3++)
                for (Dim_size_t i1_4 = 0; i1_4 < M1_4; i1_4++)
                    sum += Input.data[i1_1][i1_2][i1_3][i1_4];
            Out.data[i1_1][i1_2] = sum / (M1_3 * M1_4);
        }
}

template <Dim_size_t M1_1, Dim_size_t M1_2, Dim_size_t M1_3, Dim_size_t M1_4, typename Type = float>
__attribute__((always_inline)) inline Matrix<Type, DimensionOrder::D2_Batch_Channel, M1_1, M1_2> AdaptiveAveragePool(
        const Matrix<Type, DimensionOrder::D4_Batch_Channel_Width_Height, M1_1, M1_2, M1_3, M1_4> &Input) {
    Matrix<Type, DimensionOrder::D2_Batch_Channel, M1_1, M1_2> out;
    AdaptiveAveragePool(Input, out);
    return out;
}

/*
================================================================================================================================================
                                                        MaxPool
================================================================================================================================================
*/

template <Dim_size_t pool_size,DimensionOrder InOrder, DimensionOrder OutOrder, Dim_size_t M1_1, Dim_size_t M1_2, Dim_size_t M1_3, Dim_size_t M1_4, typename Type = float>
__attribute__((always_inline)) inline void MaxPool(const Matrix<Type,InOrder, M1_1, M1_2, M1_3, M1_4> &Input, Matrix<Type,OutOrder, M1_1, M1_2, M1_3 / pool_size, M1_4 / pool_size> &Out) {
    static_assert(M1_3 % pool_size == 0, "Input Width has to be divisible by pool_size");
    static_assert(M1_4 % pool_size == 0, "Input Height has to be divisible by pool_size");

    for (Dim_size_t i1_1 = 0; i1_1 < M1_1; i1_1++)
        for (Dim_size_t i1_2 = 0; i1_2 < M1_2; i1_2++) {
            for (Dim_size_t i1_3 = 0; i1_3 < M1_3; i1_3 += pool_size)
                for (Dim_size_t i1_4 = 0; i1_4 < M1_4; i1_4 += pool_size) {
                    Type max{static_cast<Type>(0)};
                    for (Dim_size_t i = 0; i < pool_size; i++)
                        for (Dim_size_t j = 0; j < pool_size; j++)
                            if (Input.data[i1_1][i1_2][i1_3 + i][i1_4 + j] > max)
                                max = Input.data[i1_1][i1_2][i1_3 + i][i1_4 + j];
                    Out.data[i1_1][i1_2][i1_3 / pool_size][i1_4 / pool_size] = max;
                }
        }
}

template <Dim_size_t pool_size, Dim_size_t M1_1, Dim_size_t M1_2, Dim_size_t M1_3, Dim_size_t M1_4, typename Type = float>
__attribute__((always_inline)) inline Matrix<Type,DimensionOrder::D4_Batch_Channel_Width_Height, M1_1, M1_2, M1_3 / pool_size, M1_4 / pool_size> MaxPool(const Matrix<Type,DimensionOrder::D4_Batch_Channel_Width_Height, M1_1, M1_2, M1_3, M1_4> &Input) {
    Matrix<Type,DimensionOrder::D4_Batch_Channel_Width_Height, M1_1, M1_2, M1_3 / pool_size, M1_4 / pool_size> out;
    MaxPool<pool_size>(Input, out);
    return out;
}

template <Dim_size_t pool_size,DimensionOrder InOrder, DimensionOrder OutOrder, Dim_size_t M1_1, Dim_size_t M1_2, Dim_size_t M1_3, typename Type = float>
__attribute__((always_inline)) inline void MaxPool(const Matrix<Type,InOrder, M1_1, M1_2, M1_3> &Input, Matrix<Type,OutOrder, M1_1, M1_2, M1_3 / pool_size> &Out) {
    static_assert(M1_3 % pool_size == 0, "Input Width has to be divisible by pool_size");

    for (Dim_size_t i1_1 = 0; i1_1 < M1_1; i1_1++)
        for (Dim_size_t i1_2 = 0; i1_2 < M1_2; i1_2++) {
            for (Dim_size_t i1_3 = 0; i1_3 < M1_3; i1_3 += pool_size) {
                Type max{static_cast<Type>(0)};
                for (Dim_size_t i = 0; i < pool_size; i++)
                    if (Input.data[i1_1][i1_2][i1_3 + i] > max)
                        max = Input.data[i1_1][i1_2][i1_3 + i];
                Out.data[i1_1][i1_2][i1_3 / pool_size] = max;
            }
        }
}

template <Dim_size_t pool_size, Dim_size_t M1_1, Dim_size_t M1_2, Dim_size_t M1_3, typename Type = float>
__attribute__((always_inline)) inline Matrix<Type,DimensionOrder::D3_Batch_Channel_Width, M1_1, M1_2, M1_3 / pool_size> MaxPool(const Matrix<Type,DimensionOrder::D3_Batch_Channel_Width, M1_1, M1_2, M1_3> &Input) {
    Matrix<Type,DimensionOrder::D3_Batch_Channel_Width, M1_1, M1_2, M1_3 / pool_size> out;
    MaxPool<pool_size>(Input, out);
    return out;
}

}; // namespace functions