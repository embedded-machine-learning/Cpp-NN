#include "../MatrixOperations.hpp"


template <typename T>
constexpr auto random_value = [](T &a) {
    a = static_cast<T>((((double)rand()) / RAND_MAX) * 100.0); // Generate a random value between 0 and 100
};

template <typename T>
constexpr auto cmp = [](bool &ret, const T &a, const T &b) {
    if constexpr (std::is_floating_point_v<T>) {
        ret = std::abs(a - b) / (std::abs(a) + std::abs(b)) < 1e-6; // Use a small epsilon for floating point comparison
    } else {
        ret = a == b; // For integral types, use direct comparison
    }
};


template <IsMatrixType BaseMatrixType>
constexpr auto randomize(BaseMatrixType &matrix) {
    using Type = typename std::remove_reference_t<BaseMatrixType>::value_type;
    loop(random_value<Type>, std::forward<BaseMatrixType &>(matrix));
}

template <IsMatrixType MatrixTypeA, IsMatrixType MatrixTypeB>
constexpr auto matrixCmp(MatrixTypeA &&a, MatrixTypeB &&b) {
    static_assert(std::remove_cvref_t<MatrixTypeA>::order.containsOnly(std::remove_cvref_t<MatrixTypeB>::order), "Matrix orders must be compatible for comparison");
    OverrideTypeMatrix<MatrixTypeA, bool> result; // Result matrix with bool type
    loop(cmp<typename std::remove_cvref_t<MatrixTypeA>::value_type>, result, std::forward<MatrixTypeA>(a), std::forward<MatrixTypeB>(b));
    return result;
}
