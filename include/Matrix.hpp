#pragma once

#include <array>
#include <cstddef>
#include <tuple>
#include <utility>

#include "helpers/c++17_helpers.hpp"
#include "helpers/Algorithm.hpp"    

typedef size_t Dim_size_t;

enum DimensionOrder {
    ERROR, // Error
    // 1D
    D1_Channel, // Default 1D
    // 2D
    D2_Batch_Channel,        // Default 2D
    D2_Sequence_Channel,     // Default Time Series
    D2_Channel_Batch,        // Maybe Bacthnorm?
    D2_OutChannel_InChannel, // Conv1d Weights
    D2_InChannel_OutChannel, // Conv1d Weights Transposed
    // 3D
    D3_Batch_Channel_Width,         // Default 3D, Conv2d Depthwise Preffered
    D3_Batch_Width_Channel,         // Conv1d Preffered
    D3_Width_Batch_Channel,         // Conv1d  Strange
    D3_OutChannel_InChannel_Kernel, // Conv1d Weights
    D3_OutChannel_Kernel_InChannel, // Conv1d Weights Transposed
    D3_Batch_Sequence_Channel,     // Time Series
    // 4D
    D4_Batch_Channel_Width_Height,                    // Default 4D, Conv2d Depthwise Preffered
    D4_Batch_Width_Height_Channel,                    // Conv2d Preffered
    D4_Width_Height_Batch_Channel,                    // Conv2d Strange
    D4_OutChannel_InChannel_KernelWidth_KernelHeight, // Conv2d Weights
    D4_OutChannel_KernelWidth_KernelHeight_InChannel, // Conv2d Weights
    D4_OutChannel_InChannel_KernelParallel_Unrolled,  // Linear Parallel Unrolled Weights
    // 5D
    D5_OutChannel_InChannel_Kernel_KernelParallel_Unrolled, // Conv1d Parallel Unrolled Weights suboptimal Order
    D5_OutChannel_Kernel_InChannel_KernelParallel_Unrolled, // Conv1d Parallel Unrolled Weights
    // 6D
    D6_OutChannel_KernelWidth_KernelHeight_InChannel_KernelParallel_Unrolled, // Conv2d Parallel Unrolled Weights
};

constexpr size_t DimensionOrderStringsLength = 7; // 1 more than the longest string

// How to Interprete the DimensionOrder
// \0 is ignored
// B is for Batch
// C is for Channels
// W is for Width either kernel or data
// H is for Height either kernel or data
// O is for OutChannel
// I is for InChannel
// P is for Parallel
// U is for Unrolled
// S is for Sequence
constexpr auto allowed_permutation_chars = std::array{'\0', 'B', 'C', 'W', 'H', 'O', 'I', 'P', 'U' , 'S'};

constexpr std::array<char, DimensionOrderStringsLength> DimensionOrderStrings(DimensionOrder order) {
    switch (order) {
    case DimensionOrder::ERROR: return to_array<DimensionOrderStringsLength>(""); break;
    case DimensionOrder::D1_Channel: return to_array<DimensionOrderStringsLength>("C"); break;

    case DimensionOrder::D2_Batch_Channel: return to_array<DimensionOrderStringsLength>("BC"); break;
    case DimensionOrder::D2_Sequence_Channel: return to_array<DimensionOrderStringsLength>("SC"); break;
    case DimensionOrder::D2_Channel_Batch: return to_array<DimensionOrderStringsLength>("CB"); break;
    case DimensionOrder::D2_OutChannel_InChannel: return to_array<DimensionOrderStringsLength>("OI"); break;
    case DimensionOrder::D2_InChannel_OutChannel: return to_array<DimensionOrderStringsLength>("IO"); break;

    case DimensionOrder::D3_Batch_Channel_Width: return to_array<DimensionOrderStringsLength>("BCW"); break;
    case DimensionOrder::D3_Batch_Width_Channel: return to_array<DimensionOrderStringsLength>("BWC"); break;
    case DimensionOrder::D3_Width_Batch_Channel: return to_array<DimensionOrderStringsLength>("WBC"); break;
    case DimensionOrder::D3_OutChannel_InChannel_Kernel: return to_array<DimensionOrderStringsLength>("OIW"); break;
    case DimensionOrder::D3_OutChannel_Kernel_InChannel: return to_array<DimensionOrderStringsLength>("OWI"); break;
    case DimensionOrder::D3_Batch_Sequence_Channel: return to_array<DimensionOrderStringsLength>("BSC"); break;

    case DimensionOrder::D4_Batch_Channel_Width_Height: return to_array<DimensionOrderStringsLength>("BCWH"); break;
    case DimensionOrder::D4_Batch_Width_Height_Channel: return to_array<DimensionOrderStringsLength>("BWHC"); break;
    case DimensionOrder::D4_Width_Height_Batch_Channel: return to_array<DimensionOrderStringsLength>("WHBC"); break;
    case DimensionOrder::D4_OutChannel_InChannel_KernelWidth_KernelHeight: return to_array<DimensionOrderStringsLength>("OIWH"); break;
    case DimensionOrder::D4_OutChannel_KernelWidth_KernelHeight_InChannel: return to_array<DimensionOrderStringsLength>("OWHI"); break;
    case DimensionOrder::D4_OutChannel_InChannel_KernelParallel_Unrolled: return to_array<DimensionOrderStringsLength>("OIPU"); break;

    case DimensionOrder::D5_OutChannel_InChannel_Kernel_KernelParallel_Unrolled: return to_array<DimensionOrderStringsLength>("OIWPU"); break;
    case DimensionOrder::D5_OutChannel_Kernel_InChannel_KernelParallel_Unrolled: return to_array<DimensionOrderStringsLength>("OWIPU"); break;

    case DimensionOrder::D6_OutChannel_KernelWidth_KernelHeight_InChannel_KernelParallel_Unrolled: return to_array<DimensionOrderStringsLength>("OWHIPU"); break;

    default: return to_array<DimensionOrderStringsLength>(""); break;
    }
}

template <size_t N, DimensionOrder From, DimensionOrder To>
struct DimensionStringPermutation;

#if false
#include <iostream>

template <size_t N>
constexpr void check_bounds_helper(Dim_size_t Dim, Dim_size_t accessed_index) {
    // std::cout << "Dim: " << Dim << " accessed_index: " << accessed_index << std::endl;
    if (accessed_index < 0) {
        std::cout << "Index out of bounds" << std::endl;
        std::cout << "Dim: " << Dim << " accessed_index: " << accessed_index << std::endl;
        std::cout << "Number of dimensions: " << N << std::endl;
        throw std::out_of_range("Index out of bounds");
    }
    if (accessed_index >= Dim) {
        std::cout << "Index out of bounds" << std::endl;
        std::cout << "Dim: " << Dim << " accessed_index: " << accessed_index << std::endl;
        std::cout << "Number of dimensions: " << N << std::endl;
        throw std::out_of_range("Index out of bounds");
    }
}

template <Dim_size_t... Dims>
struct check_bounds_t {
    template <size_t... indexes>
    __attribute__((always_inline)) constexpr static inline void check(std::array<Dim_size_t, sizeof...(Dims)> dims, std::index_sequence<indexes...>) {
        ((check_bounds_helper<sizeof...(Dims)>(std::array<Dim_size_t, sizeof...(Dims)>{Dims...}[indexes], dims[indexes])), ...);
    }
};

template <Dim_size_t... Dims>
__attribute__((always_inline)) constexpr inline void check_bounds(std::array<Dim_size_t, sizeof...(Dims)> dims) {
    check_bounds_t<Dims...>::check(dims, std::make_index_sequence<sizeof...(Dims)>());
}
#else
template <Dim_size_t... Dims>
__attribute__((always_inline)) constexpr inline void check_bounds(std::array<Dim_size_t, sizeof...(Dims)> dims) {
}
#endif

// template <DimensionOrder OrderTo>
// struct MatrixPermutation;

template <DimensionOrder From, DimensionOrder To>
using PermutationIndex = decltype(DimensionStringPermutation<DimensionOrderStringsLength, From, To>::permutation());

template <typename Matrix, DimensionOrder OrderTo>
struct MatrixPermutationHelper;

template <typename Matrix, DimensionOrder OrderTo>
using MatrixOrderConversion = MatrixPermutationHelper<remove_cvref_t<Matrix>, OrderTo>;

template <typename Type = float, DimensionOrder = DimensionOrder::ERROR, Dim_size_t...>
struct Matrix;

template <typename Type>
constexpr bool is_Sequence = false;

template <typename Type, DimensionOrder Order, Dim_size_t... Dims> 
constexpr bool is_Sequence<Matrix<Type,Order,Dims...>> = helpers::char_Array_contains(DimensionOrderStrings(Order) , 'S'); 

template <typename Type, DimensionOrder Order, Dim_size_t Dim1>
struct Matrix<Type, Order, Dim1> {
    using type                            = Type;
    static constexpr size_t         dims  = 1;
    static constexpr Dim_size_t     dim1  = Dim1;
    static constexpr DimensionOrder order = Order;

    template <DimensionOrder OrderTo>
    using Permutation = typename MatrixOrderConversion<Matrix<Type, Order, Dim1>, OrderTo>::type;

    Type data[Dim1] = {};

    __attribute__((always_inline)) inline Type &at(Dim_size_t dim1) {
        check_bounds<Dim1>({dim1});
        return data[dim1];
    }

    __attribute__((always_inline)) constexpr inline const Type &at(Dim_size_t dim1) const {
        check_bounds<Dim1>({dim1});
        return data[dim1];
    }

    template <DimensionOrder OrderAs>
    __attribute__((always_inline)) inline Type &at(Dim_size_t dim1) {
        return MatrixOrderConversion<Matrix<Type, Order, Dim1>, OrderAs>::at(*this, dim1);
    }

    template <DimensionOrder OrderAs>
    __attribute__((always_inline)) constexpr inline const Type &at(Dim_size_t dim1) const {
        return MatrixOrderConversion<Matrix<Type, Order, Dim1>, OrderAs>::at(*this, dim1);
    }
};

template <typename Type, DimensionOrder Order, Dim_size_t Dim1, Dim_size_t Dim2>
struct Matrix<Type, Order, Dim1, Dim2> {
    using type                            = Type;
    static constexpr size_t         dims  = 2;
    static constexpr Dim_size_t     dim1  = Dim1;
    static constexpr Dim_size_t     dim2  = Dim2;
    static constexpr DimensionOrder order = Order;

    template <DimensionOrder OrderTo>
    using Permutation = typename MatrixOrderConversion<Matrix<Type, Order, Dim1, Dim2>, OrderTo>::type;

    Type data[Dim1][Dim2] = {};

    __attribute__((always_inline)) inline Type &at(Dim_size_t dim1, Dim_size_t dim2) {
        check_bounds<Dim1, Dim2>({dim1, dim2});
        return data[dim1][dim2];
    }

    __attribute__((always_inline)) constexpr inline const Type &at(Dim_size_t dim1, Dim_size_t dim2) const {
        check_bounds<Dim1, Dim2>({dim1, dim2});
        return data[dim1][dim2];
    }

    template <DimensionOrder OrderAs = Order>
    __attribute__((always_inline)) inline Type &at(Dim_size_t dim1, Dim_size_t dim2) {
        return MatrixOrderConversion<Matrix<Type, Order, Dim1, Dim2>, OrderAs>::at(*this, dim1, dim2);
    }

    template <DimensionOrder OrderAs = Order>
    __attribute__((always_inline)) constexpr inline const Type &at(Dim_size_t dim1, Dim_size_t dim2) const {
        return MatrixOrderConversion<Matrix<Type, Order, Dim1, Dim2>, OrderAs>::at(*this, dim1, dim2);
    }
};

template <typename Type, DimensionOrder Order, Dim_size_t Dim1, Dim_size_t Dim2, Dim_size_t Dim3>
struct Matrix<Type, Order, Dim1, Dim2, Dim3> {
    using type                            = Type;
    static constexpr size_t         dims  = 3;
    static constexpr Dim_size_t     dim1  = Dim1;
    static constexpr Dim_size_t     dim2  = Dim2;
    static constexpr Dim_size_t     dim3  = Dim3;
    static constexpr DimensionOrder order = Order;

    template <DimensionOrder OrderTo>
    using Permutation = typename MatrixOrderConversion<Matrix<Type, Order, Dim1, Dim2, Dim3>, OrderTo>::type;

    Type data[Dim1][Dim2][Dim3] = {};

    __attribute__((always_inline)) inline Type &at(Dim_size_t dim1, Dim_size_t dim2, Dim_size_t dim3) {
        check_bounds<Dim1, Dim2, Dim3>({dim1, dim2, dim3});
        return data[dim1][dim2][dim3];
    }

    __attribute__((always_inline)) constexpr inline const Type &at(Dim_size_t dim1, Dim_size_t dim2, Dim_size_t dim3) const {
        check_bounds<Dim1, Dim2, Dim3>({dim1, dim2, dim3});
        return data[dim1][dim2][dim3];
    }

    template <DimensionOrder OrderAs>
    __attribute__((always_inline)) inline Type &at(Dim_size_t dim1, Dim_size_t dim2, Dim_size_t dim3) {
        return MatrixOrderConversion<Matrix<Type, Order, Dim1, Dim2, Dim3>, OrderAs>::at(*this, dim1, dim2, dim3);
    }

    template <DimensionOrder OrderAs>
    __attribute__((always_inline)) constexpr inline const Type &at(Dim_size_t dim1, Dim_size_t dim2, Dim_size_t dim3) const {
        return MatrixOrderConversion<Matrix<Type, Order, Dim1, Dim2, Dim3>, OrderAs>::at(*this, dim1, dim2, dim3);
    }
};

template <typename Type, DimensionOrder Order, Dim_size_t Dim1, Dim_size_t Dim2, Dim_size_t Dim3, Dim_size_t Dim4>
struct Matrix<Type, Order, Dim1, Dim2, Dim3, Dim4> {
    using type                            = Type;
    static constexpr size_t         dims  = 4;
    static constexpr Dim_size_t     dim1  = Dim1;
    static constexpr Dim_size_t     dim2  = Dim2;
    static constexpr Dim_size_t     dim3  = Dim3;
    static constexpr Dim_size_t     dim4  = Dim4;
    static constexpr DimensionOrder order = Order;

    template <DimensionOrder OrderTo>
    using Permutation = typename MatrixOrderConversion<Matrix<Type, Order, Dim1, Dim2, Dim3, Dim4>, OrderTo>::type;

    Type data[Dim1][Dim2][Dim3][Dim4] = {};

    __attribute__((always_inline)) inline Type &at(Dim_size_t dim1, Dim_size_t dim2, Dim_size_t dim3, Dim_size_t dim4) {
        check_bounds<Dim1, Dim2, Dim3, Dim4>({dim1, dim2, dim3, dim4});
        return data[dim1][dim2][dim3][dim4];
    }

    __attribute__((always_inline)) constexpr inline const Type &at(Dim_size_t dim1, Dim_size_t dim2, Dim_size_t dim3, Dim_size_t dim4) const {
        check_bounds<Dim1, Dim2, Dim3, Dim4>({dim1, dim2, dim3, dim4});
        return data[dim1][dim2][dim3][dim4];
    }

    template <DimensionOrder OrderAs>
    __attribute__((always_inline)) inline Type &at(Dim_size_t dim1, Dim_size_t dim2, Dim_size_t dim3, Dim_size_t dim4) {
        return MatrixOrderConversion<Matrix<Type, Order, Dim1, Dim2, Dim3, Dim4>, OrderAs>::at(*this, dim1, dim2, dim3, dim4);
    }

    template <DimensionOrder OrderAs>
    __attribute__((always_inline)) constexpr inline const Type &at(Dim_size_t dim1, Dim_size_t dim2, Dim_size_t dim3, Dim_size_t dim4) const {
        return MatrixOrderConversion<Matrix<Type, Order, Dim1, Dim2, Dim3, Dim4>, OrderAs>::at(*this, dim1, dim2, dim3, dim4);
    }
};

template <typename Type, DimensionOrder Order, Dim_size_t Dim1, Dim_size_t Dim2, Dim_size_t Dim3, Dim_size_t Dim4, Dim_size_t Dim5>
struct Matrix<Type, Order, Dim1, Dim2, Dim3, Dim4, Dim5> {
    using type                            = Type;
    static constexpr size_t         dims  = 5;
    static constexpr Dim_size_t     dim1  = Dim1;
    static constexpr Dim_size_t     dim2  = Dim2;
    static constexpr Dim_size_t     dim3  = Dim3;
    static constexpr Dim_size_t     dim4  = Dim4;
    static constexpr Dim_size_t     dim5  = Dim5;
    static constexpr DimensionOrder order = Order;

    template <DimensionOrder OrderTo>
    using Permutation = typename MatrixOrderConversion<Matrix<Type, Order, Dim1, Dim2, Dim3, Dim4, Dim5>, OrderTo>::type;

    Type data[Dim1][Dim2][Dim3][Dim4][Dim5] = {};

    __attribute__((always_inline)) inline Type &at(Dim_size_t dim1, Dim_size_t dim2, Dim_size_t dim3, Dim_size_t dim4, Dim_size_t dim5) {
        check_bounds<Dim1, Dim2, Dim3, Dim4, Dim5>({dim1, dim2, dim3, dim4, dim5});
        return data[dim1][dim2][dim3][dim4][dim5];
    }

    __attribute__((always_inline)) constexpr inline const Type &at(Dim_size_t dim1, Dim_size_t dim2, Dim_size_t dim3, Dim_size_t dim4, Dim_size_t dim5) const {
        check_bounds<Dim1, Dim2, Dim3, Dim4, Dim5>({dim1, dim2, dim3, dim4, dim5});
        return data[dim1][dim2][dim3][dim4][dim5];
    }

    template <DimensionOrder OrderAs>
    __attribute__((always_inline)) inline Type &at(Dim_size_t dim1, Dim_size_t dim2, Dim_size_t dim3, Dim_size_t dim4, Dim_size_t dim5) {
        return MatrixOrderConversion<Matrix<Type, Order, Dim1, Dim2, Dim3, Dim4, Dim5>, OrderAs>::at(*this, dim1, dim2, dim3, dim4, dim5);
    }

    template <DimensionOrder OrderAs>
    __attribute__((always_inline)) constexpr inline const Type &at(Dim_size_t dim1, Dim_size_t dim2, Dim_size_t dim3, Dim_size_t dim4, Dim_size_t dim5) const {
        return MatrixOrderConversion<Matrix<Type, Order, Dim1, Dim2, Dim3, Dim4, Dim5>, OrderAs>::at(*this, dim1, dim2, dim3, dim4, dim5);
    }
};

template <typename Type, DimensionOrder Order, Dim_size_t Dim1, Dim_size_t Dim2, Dim_size_t Dim3, Dim_size_t Dim4, Dim_size_t Dim5, Dim_size_t Dim6>
struct Matrix<Type, Order, Dim1, Dim2, Dim3, Dim4, Dim5, Dim6> {
    using type                            = Type;
    static constexpr size_t         dims  = 6;
    static constexpr Dim_size_t     dim1  = Dim1;
    static constexpr Dim_size_t     dim2  = Dim2;
    static constexpr Dim_size_t     dim3  = Dim3;
    static constexpr Dim_size_t     dim4  = Dim4;
    static constexpr Dim_size_t     dim5  = Dim5;
    static constexpr Dim_size_t     dim6  = Dim6;
    static constexpr DimensionOrder order = Order;

    template <DimensionOrder OrderTo>
    using Permutation = typename MatrixOrderConversion<Matrix<Type, Order, Dim1, Dim2, Dim3, Dim4, Dim5, Dim6>, OrderTo>::type;

    Type data[Dim1][Dim2][Dim3][Dim4][Dim5][Dim6] = {};

    __attribute__((always_inline)) inline Type &at(Dim_size_t dim1, Dim_size_t dim2, Dim_size_t dim3, Dim_size_t dim4, Dim_size_t dim5, Dim_size_t dim6) {
        check_bounds<Dim1, Dim2, Dim3, Dim4, Dim5, Dim6>({dim1, dim2, dim3, dim4, dim5, dim6});
        return data[dim1][dim2][dim3][dim4][dim5][dim6];
    }

    __attribute__((always_inline)) constexpr inline const Type &at(Dim_size_t dim1, Dim_size_t dim2, Dim_size_t dim3, Dim_size_t dim4, Dim_size_t dim5, Dim_size_t dim6) const {
        check_bounds<Dim1, Dim2, Dim3, Dim4, Dim5, Dim6>({dim1, dim2, dim3, dim4, dim5, dim6});
        return data[dim1][dim2][dim3][dim4][dim5][dim6];
    }

    template <DimensionOrder OrderAs>
    __attribute__((always_inline)) inline Type &at(Dim_size_t dim1, Dim_size_t dim2, Dim_size_t dim3, Dim_size_t dim4, Dim_size_t dim5, Dim_size_t dim6) {
        return MatrixOrderConversion<Matrix<Type, Order, Dim1, Dim2, Dim3, Dim4, Dim5, Dim6>, OrderAs>::at(*this, dim1, dim2, dim3, dim4, dim5, dim6);
    }

    template <DimensionOrder OrderAs>
    __attribute__((always_inline)) constexpr inline const Type &at(Dim_size_t dim1, Dim_size_t dim2, Dim_size_t dim3, Dim_size_t dim4, Dim_size_t dim5, Dim_size_t dim6) const {
        return MatrixOrderConversion<Matrix<Type, Order, Dim1, Dim2, Dim3, Dim4, Dim5, Dim6>, OrderAs>::at(*this, dim1, dim2, dim3, dim4, dim5, dim6);
    }
};

// Error Type
template <DimensionOrder Order, Dim_size_t Dim1>
struct Matrix<void, Order, Dim1> {
    using type                            = void;
    static constexpr size_t         dims  = 1;
    static constexpr Dim_size_t     dim1  = Dim1;
    static constexpr DimensionOrder order = Order;

    template <DimensionOrder OrderAs = Order>
    __attribute__((always_inline)) inline void at(Dim_size_t dim1) {
        return;
    }

    template <DimensionOrder OrderAs = Order>
    __attribute__((always_inline)) constexpr inline const void at(Dim_size_t dim1) const {
        return;
    }
};

template <DimensionOrder Order, Dim_size_t Dim1, Dim_size_t Dim2>
struct Matrix<void, Order, Dim1, Dim2> {
    using type                            = void;
    static constexpr size_t         dims  = 2;
    static constexpr Dim_size_t     dim1  = Dim1;
    static constexpr Dim_size_t     dim2  = Dim2;
    static constexpr DimensionOrder order = Order;

    template <DimensionOrder OrderAs = Order>
    __attribute__((always_inline)) inline void at(Dim_size_t dim1, Dim_size_t dim2) {
        return;
    }

    template <DimensionOrder OrderAs = Order>
    __attribute__((always_inline)) constexpr inline const void at(Dim_size_t dim1, Dim_size_t dim2) const {
        return;
    }
};

template <DimensionOrder Order, Dim_size_t Dim1, Dim_size_t Dim2, Dim_size_t Dim3>
struct Matrix<void, Order, Dim1, Dim2, Dim3> {
    using type                            = void;
    static constexpr size_t         dims  = 3;
    static constexpr Dim_size_t     dim1  = Dim1;
    static constexpr Dim_size_t     dim2  = Dim2;
    static constexpr Dim_size_t     dim3  = Dim3;
    static constexpr DimensionOrder order = Order;

    template <DimensionOrder OrderAs = Order>
    __attribute__((always_inline)) inline void at(Dim_size_t dim1, Dim_size_t dim2, Dim_size_t dim3) {
        return;
    }

    template <DimensionOrder OrderAs = Order>
    __attribute__((always_inline)) constexpr inline const void at(Dim_size_t dim1, Dim_size_t dim2, Dim_size_t dim3) const {
        return;
    }
};

template <DimensionOrder Order, Dim_size_t Dim1, Dim_size_t Dim2, Dim_size_t Dim3, Dim_size_t Dim4>
struct Matrix<void, Order, Dim1, Dim2, Dim3, Dim4> {
    using type                            = void;
    static constexpr size_t         dims  = 4;
    static constexpr Dim_size_t     dim1  = Dim1;
    static constexpr Dim_size_t     dim2  = Dim2;
    static constexpr Dim_size_t     dim3  = Dim3;
    static constexpr Dim_size_t     dim4  = Dim4;
    static constexpr DimensionOrder order = Order;

    template <DimensionOrder OrderAs = Order>
    __attribute__((always_inline)) inline void at(Dim_size_t dim1, Dim_size_t dim2, Dim_size_t dim3, Dim_size_t dim4) {
        return;
    }

    template <DimensionOrder OrderAs = Order>
    __attribute__((always_inline)) constexpr inline const void at(Dim_size_t dim1, Dim_size_t dim2, Dim_size_t dim3, Dim_size_t dim4) const {
        return;
    }
};

template <DimensionOrder To,
          typename InputMatrixType,
          DimensionOrder From               = InputMatrixType::order,
          typename OutputMatrixType         = typename MatrixOrderConversion<InputMatrixType, To>::type,
          std::enable_if_t<From == To, int> = 0>
constexpr OutputMatrixType PermuteMatrix(const InputMatrixType Input) {
    return Input;
}

template <DimensionOrder To,
          typename InputMatrixType,
          DimensionOrder From                                             = InputMatrixType::order,
          typename OutputMatrixType                                       = typename MatrixOrderConversion<InputMatrixType, To>::type,
          std::enable_if_t<InputMatrixType::dims == 2 && From != To, int> = 0>
constexpr OutputMatrixType PermuteMatrix(const InputMatrixType Input) {
    OutputMatrixType Out;

    for (Dim_size_t i = 0; i < OutputMatrixType::dim1; i++) {
        for (Dim_size_t j = 0; j < OutputMatrixType::dim2; j++) {
            Out.data[i][j] = Input.template at<To>(i, j);
        }
    }
    return Out;
}

template <DimensionOrder To,
          typename InputMatrixType,
          DimensionOrder From                                             = InputMatrixType::order,
          typename OutputMatrixType                                       = typename MatrixOrderConversion<InputMatrixType, To>::type,
          std::enable_if_t<InputMatrixType::dims == 3 && From != To, int> = 0>
constexpr OutputMatrixType PermuteMatrix(const InputMatrixType Input) {
    OutputMatrixType Out;

    for (Dim_size_t i = 0; i < OutputMatrixType::dim1; i++) {
        for (Dim_size_t j = 0; j < OutputMatrixType::dim2; j++) {
            for (Dim_size_t k = 0; k < OutputMatrixType::dim3; k++) {
                Out.data[i][j][k] = Input.template at<To>(i, j, k);
            }
        }
    }
    return Out;
}

template <DimensionOrder To,
          typename InputMatrixType,
          DimensionOrder From                                             = InputMatrixType::order,
          typename OutputMatrixType                                       = typename MatrixOrderConversion<InputMatrixType, To>::type,
          std::enable_if_t<InputMatrixType::dims == 4 && From != To, int> = 0>
constexpr OutputMatrixType PermuteMatrix(const InputMatrixType Input) {
    OutputMatrixType Out;

    for (Dim_size_t i = 0; i < OutputMatrixType::dim1; i++) {
        for (Dim_size_t j = 0; j < OutputMatrixType::dim2; j++) {
            for (Dim_size_t k = 0; k < OutputMatrixType::dim3; k++) {
                for (Dim_size_t l = 0; l < OutputMatrixType::dim4; l++) {
                    Out.data[i][j][k][l] = Input.template at<To>(i, j, k, l);
                }
            }
        }
    }
    return Out;
}

template <DimensionOrder To,
          typename InputMatrixType,
          DimensionOrder From                                             = InputMatrixType::order,
          typename OutputMatrixType                                       = typename MatrixOrderConversion<InputMatrixType, To>::type,
          std::enable_if_t<InputMatrixType::dims == 5 && From != To, int> = 0>
constexpr OutputMatrixType PermuteMatrix(const InputMatrixType Input) {
    OutputMatrixType Out;

    for (Dim_size_t i = 0; i < OutputMatrixType::dim1; i++) {
        for (Dim_size_t j = 0; j < OutputMatrixType::dim2; j++) {
            for (Dim_size_t k = 0; k < OutputMatrixType::dim3; k++) {
                for (Dim_size_t l = 0; l < OutputMatrixType::dim4; l++) {
                    for (Dim_size_t m = 0; m < OutputMatrixType::dim5; m++) {
                        Out.data[i][j][k][l][m] = Input.template at<To>(i, j, k, l, m);
                    }
                }
            }
        }
    }
    return Out;
}

// Dimension Permutation Type

// N is the maximum number of dimensions so any permutation will have N + 1 as the error state
template <size_t N>
constexpr size_t PermutationErrorState = N + 1;

template <size_t, typename, typename>
struct remove_ErrorStates;

template <size_t N, size_t... OutIndexes, size_t current, size_t... InIndexes>
struct remove_ErrorStates<N, std::index_sequence<OutIndexes...>, std::index_sequence<current, InIndexes...>> {
    using type = typename std::conditional<current == PermutationErrorState<N>,
                                           std::index_sequence<OutIndexes...>,
                                           typename remove_ErrorStates<N, std::index_sequence<OutIndexes..., current>, std::index_sequence<InIndexes...>>::type>::type;
};

template <size_t N, size_t... OutIndexes, size_t current>
struct remove_ErrorStates<N, std::index_sequence<OutIndexes...>, std::index_sequence<current>> {
    using type = typename std::conditional<current == PermutationErrorState<N>, std::index_sequence<OutIndexes...>, std::index_sequence<OutIndexes..., current>>::type;
};

template <size_t N, DimensionOrder From, DimensionOrder To>
struct DimensionStringPermutation {
  private:
    template <char v>
    constexpr static auto allowed_char() {
        bool part_of_allowed_chars = false;
        for (size_t i = 0; i < allowed_permutation_chars.size(); ++i) {
            if (v == allowed_permutation_chars[i]) {
                part_of_allowed_chars = true;
            }
        }
        return part_of_allowed_chars;
    }

    template <char v, DimensionOrder Order>
    constexpr static auto order_helper() {
        static_assert(allowed_char<v>(), "Invalid character in DimensionOrder");
        constexpr auto comp = DimensionOrderStrings(Order);
        size_t         i    = PermutationErrorState<N>;
        for (size_t j = 0; j < N; ++j) {
            if (comp[j] != '\0' && comp[j] == v) {
                i = j;
                break;
            }
        }
        return i;
    }

    template <size_t... sequence>
    constexpr static auto permutation(std::index_sequence<sequence...>) {
        return typename remove_ErrorStates<N, std::index_sequence<>, std::index_sequence<order_helper<DimensionOrderStrings(To)[sequence], From>()...>>::type{};
    }

  public:
    constexpr static auto permutation() {
        return permutation(std::make_index_sequence<N>{});
    }
};

// Dimension Permutation

template <typename Type, DimensionOrder OrderFrom, Dim_size_t... Dims, DimensionOrder OrderTo>
struct MatrixPermutationHelper<Matrix<Type, OrderFrom, Dims...>, OrderTo> {
    template <typename index_sequence>
    struct helper;

    template <size_t... Indexs>
    struct helper<std::index_sequence<Indexs...>> {
        static_assert(sizeof...(Indexs) == sizeof...(Dims), "Number of indexes must match number of dimensions");
        using type = Matrix<Type, OrderTo, std::get<Indexs>(std::make_tuple(Dims...))...>;
    };

    using type = typename helper<PermutationIndex<OrderFrom, OrderTo>>::type;

    using MatrixValueType = Type;
    using MatrixType      = Matrix<Type, OrderFrom, Dims...>;

    template <typename... DimTypes>
    __attribute__((always_inline)) constexpr inline static Type &at(Matrix<Type, OrderFrom, Dims...> &matrix, DimTypes... indexes) {
        return at(matrix, PermutationIndex<OrderTo, OrderFrom>(), indexes...);
    }

    template <typename... DimTypes, size_t... Indexs>
    __attribute__((always_inline)) constexpr inline static Type &at(Matrix<Type, OrderFrom, Dims...> &matrix, std::index_sequence<Indexs...>, DimTypes... indexes) {
        static_assert(sizeof...(DimTypes) == sizeof...(Dims), "Number of indexes must match number of dimensions");
        static_assert(sizeof...(DimTypes) == sizeof...(Indexs), "Number of indexes must match number of dimensions, check permutation code/dimension order for errors");
        return matrix.at(std::get<Indexs>(std::make_tuple(indexes...))...);
    }

    template <typename... DimTypes>
    __attribute__((always_inline)) constexpr inline static const Type &at(const Matrix<Type, OrderFrom, Dims...> &matrix, DimTypes... indexes) {
        return at(matrix, PermutationIndex<OrderTo, OrderFrom>(), indexes...);
    }

    template <typename... DimTypes, size_t... Indexs>
    __attribute__((always_inline)) constexpr inline static const Type &at(const Matrix<Type, OrderFrom, Dims...> &matrix, std::index_sequence<Indexs...>, DimTypes... indexes) {
        static_assert(sizeof...(DimTypes) == sizeof...(Dims), "Number of indexes must match number of dimensions");
        static_assert(sizeof...(DimTypes) == sizeof...(Indexs), "Number of indexes must match number of dimensions, check permutation code/dimension order for errors");
        return matrix.at(std::get<Indexs>(std::make_tuple(indexes...))...);
    }
};