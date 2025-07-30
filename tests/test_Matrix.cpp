#include <iostream>

#include "../include/MatrixOperations.hpp"
#include "../include/helpers/print.hpp"

__attribute__((noinline)) void testMatrix() {
    std::cout << "************************************************************************" << std::endl;
    std::cout << "Matrix test function called." << std::endl;
    std::cout << "************************************************************************" << std::endl;
    Matrix<float, DimensionOrder("CWH"), 2, 3, 4>           matrix;
    constexpr Matrix<float, DimensionOrder("CWH"), 2, 3, 4> constmatrix;

    std::cout << "Matrix created with type: " << typeid(matrix).name() << std::endl;
    matrix.at(0, 0, 0) = 1.0f;
    std::cout << "Matrix at position (0): " << matrix.at(0, 0, 0) << std::endl;
    std::cout << "Matrix offsets: " << matrix.offsets << std::endl;

    std::cout << "size of matrix: " << sizeof(matrix) << " bytes" << std::endl;
    std::cout << "size of data: " << sizeof(matrix.data) << " bytes" << std::endl;

    // constmatrix.at(0,0,0) = 1.0f; // This line should not compile, as constmatrix is const
    std::cout << "Const Matrix at position (0): " << constmatrix.at(0, 0, 0) << std::endl;

    matrix.template at<"HWC">(3, 2, 1) = 3.14f;
    std::cout << "Matrix at position (1,2,3) after template access: " << matrix.at(1, 2, 3) << std::endl;
}

Matrix<float, DimensionOrder("CWH"), 2, 3, 4>           matrix;
constexpr Matrix<float, DimensionOrder("CWH"), 2, 3, 4> const_matrix;

__attribute__((noinline)) void testPermutation() {
    std::cout << "************************************************************************" << std::endl;
    std::cout << "Matrix permutation test function called." << std::endl;
    std::cout << "************************************************************************" << std::endl;
    constexpr auto From = DimensionOrder("CWH");
    constexpr auto To   = DimensionOrder("HCW");

    constexpr auto PermutationIndexes = From.permutationOrderComputation<3>(To);
    std::cout << "Permutation indexes from CWH to HWC: " << PermutationIndexes << std::endl;

    matrix.at(1, 2, 3) = 42.0f;
    std::cout << "Matrix at position (1,2,3): " << matrix.at(1, 2, 3) << std::endl;
    matrix.template at<DimensionOrder("HCW")>(3, 1, 2) = 3.14f;
    std::cout << "Matrix at position (1,2,3): " << matrix.at(1, 2, 3) << std::endl;

    std::cout << "Matrix data: " << matrix.data << std::endl;

    auto permutedMatrix = permute<DimensionOrder("HCW")>(matrix);
    std::cout << "Permuted Matrix created with type: " << typeid(permutedMatrix).name() << std::endl;
    std::cout << "Permuted Matrix has sizeof: " << sizeof(permutedMatrix) << " bytes" << std::endl;
    // PermutedMatrix<DimensionOrder("HCW"), decltype(matrix)> permutedMatrix = PermutedMatrix<DimensionOrder("HCW"),decltype(matrix)>(matrix);
    std::cout << "Permuted Matrix dimensions: " << permutedMatrix.dimensions << std::endl;
    std::cout << "Permuted Matrix at position (3,1,2): " << permutedMatrix.at(3, 1, 2) << std::endl;
    permutedMatrix.at(3, 1, 2) = 99.0f;
    std::cout << "Permuted Matrix at position (3,1,2) after modification: " << permutedMatrix.at(3, 1, 2) << std::endl;
    std::cout << "Matrix at position (1,2,3) should be same: " << matrix.at(1, 2, 3) << std::endl;

    auto permutedMatrix2 = permute<DimensionOrder("CWH")>(permutedMatrix);
    std::cout << "Permuted Matrix 2 created with type: " << typeid(permutedMatrix2).name() << std::endl;
    std::cout << "Permuted Matrix 2 has sizeof: " << sizeof(permutedMatrix2) << " bytes" << std::endl;

    auto permutedMatrix3 = permute<DimensionOrder("CWH")>(permute<DimensionOrder("HCW")>(permute<DimensionOrder("CWH")>(permute<DimensionOrder("HCW")>(matrix))));
    std::cout << "Permuted Matrix 3 created with type: " << typeid(permutedMatrix3).name() << std::endl;
    std::cout << "Permuted Matrix 3 has sizeof: " << sizeof(permutedMatrix3) << " bytes" << std::endl;
    // PermutedMatrix<Dimension
    std::cout << "Permuted Matrix 3 dimensions: " << permutedMatrix3.dimensions << std::endl;
    std::cout << "Permuted Matrix 3 at position (1,2,3): " << permutedMatrix3.at(1, 2, 3) << std::endl;
    permutedMatrix3.at(1, 2, 3) = 88.0f;
    std::cout << "Permuted Matrix 3 at position (1,2,3) after modification: " << permutedMatrix3.at(1, 2, 3) << std::endl;
    std::cout << "Matrix at position (1,2,3) should be same: " << matrix.at(1, 2, 3) << std::endl;

    constexpr auto const_permutedMatrix = permute<DimensionOrder("HCW")>(const_matrix);
    std::cout << "Const Permuted Matrix created with type: " << typeid(const_permutedMatrix).name() << std::endl;
    std::cout << "Const Permuted Matrix has sizeof: " << sizeof(const_permutedMatrix) << " bytes" << std::endl;
    std::cout << "Const Permuted Matrix dimensions: " << const_permutedMatrix.dimensions << std::endl;
    std::cout << "Const Permuted Matrix at position (3,1,2): " << const_permutedMatrix.at(0, 1, 2) << std::endl;
    // const_permutedMatrix.at(3, 1, 2) = 99.0f; // This line should not compile, as const_permutedMatrix is const

    auto permutedMatrix4 = permute<DimensionOrder("CH")>(Matrix<float, DimensionOrder("HC"), 20, 20>());
    std::cout << "Permuted Matrix 4 created with type: " << typeid(permutedMatrix4).name() << std::endl;
    std::cout << "Permuted Matrix 4 has sizeof: " << sizeof(permutedMatrix4) << " bytes" << std::endl;
    std::cout << "Permuted Matrix 4 dimensions: " << permutedMatrix4.dimensions << std::endl;
    std::cout << "Permuted Matrix 4 at position (0,1): " << permutedMatrix4.at(0, 1) << std::endl;
    permutedMatrix4.at(0, 1) = 42.0f;
    std::cout << "Permuted Matrix 4 at position (0,1) after modification: " << permutedMatrix4.at(0, 1) << std::endl;
    std::cout << "Matrix at position (0,1) should be same: " << matrix.at(0, 1, 0) << std::endl;

    permutedMatrix.template at<"HWC">(3, 2, 1) = 99.0f;
    std::cout << "Permuted Matrix at position (3,2,1) after template access: " << permutedMatrix.at(3, 1, 2) << std::endl;
    std::cout << "Base Matrix at position (1,2,3) should be same: " << matrix.at(1, 2, 3) << std::endl;

}

Matrix<float, DimensionOrder("CWH"), 4, 3, 2> matrixA;
// Matrix<float, DimensionOrder("CWH"), 4, 3, 2> matrixB;
Matrix<float, DimensionOrder("HWC"), 2, 3, 4> matrixB;

constexpr Matrix<float, DimensionOrder("CWH"), 4, 3, 2> const_matrixA;
constexpr Matrix<float, DimensionOrder("HCW"), 2, 4, 3> const_matrixB;

__attribute__((noinline)) void testConcatenation() {
    std::cout << "************************************************************************" << std::endl;
    std::cout << "Matrix concatenation test function called." << std::endl;
    std::cout << "************************************************************************" << std::endl;

    auto concatenatedMatrix = concatenate<2>(matrixA, permute<matrixA.order>(matrixB));

    std::cout << "Concatenated Matrix dimensions: " << concatenatedMatrix.dimensions << std::endl;
    std::cout << "Concatenated Matrix offsets: " << concatenatedMatrix.offsets << std::endl;

    std::cout << "Concatenated Matrix created with type: " << typeid(concatenatedMatrix).name() << std::endl;
    std::cout << "Concatenated Matrix has sizeof: " << sizeof(concatenatedMatrix) << " bytes" << std::endl;
    std::cout << "Concatenated Matrix data size: " << sizeof(concatenatedMatrix.data) << " bytes" << std::endl;

    // concatenatedMatrix.at(0, 0, 0) = 1.0f;
    // std::cout << "Concatenated Matrix at position (0,0,0): " << concatenatedMatrix.at(0, 0, 0) << std::endl;
    // std::cout << "MatrixA data: " << matrixA.data << std::endl;
    // std::cout << "MatrixB data: " << matrixB.data << std::endl;

    // concatenatedMatrix.at(1, 0, 0) = 2.0f;
    // std::cout << "Concatenated Matrix at position (1,0,0): " << concatenatedMatrix.at(1, 0, 0) << std::endl;
    // std::cout << "MatrixA data: " << matrixA.data << std::endl;
    // std::cout << "MatrixB data: " << matrixB.data << std::endl;

    // concatenatedMatrix.at(2, 0, 0) = 3.0f;
    // std::cout << "Concatenated Matrix at position (2,0,0): " << concatenatedMatrix.at(2, 0, 0) << std::endl;
    // std::cout << "MatrixA data: " << matrixA.data << std::endl;
    // std::cout << "MatrixB data: " << matrixB.data << std::endl;

    // concatenatedMatrix.at(3, 0, 0) = 4.0f;
    // std::cout << "Concatenated Matrix at position (3,0,0): " << concatenatedMatrix.at(3, 0, 0) << std::endl;
    // std::cout << "MatrixA data: " << matrixA.data << std::endl;
    // std::cout << "MatrixB data: " << matrixB.data << std::endl;

    for (unsigned long a = 0; a < concatenatedMatrix.dimensions[0]; a++) {
        for (unsigned long b = 0; b < concatenatedMatrix.dimensions[1]; b++) {
            for (unsigned long c = 0; c < concatenatedMatrix.dimensions[2]; c++) {
                concatenatedMatrix.at(a, b, c) = static_cast<float>(a * concatenatedMatrix.dimensions[1] * concatenatedMatrix.dimensions[2] + b * concatenatedMatrix.dimensions[2] + c + 1);
                std::cout << "Concatenated Matrix at (" << a << "," << b << "," << c << "): " << concatenatedMatrix.at(a, b, c) << std::endl;
                std::cout << "MatrixA data: " << matrixA.data << std::endl;
                std::cout << "MatrixB data: " << matrixB.data << std::endl;
            }
        }
    }

    constexpr auto const_concatenatedMatrix = concatenate<2>(const_matrixA, permute<const_matrixA.order>(const_matrixB));
    std::cout << "Const Concatenated Matrix created with type: " << typeid(const_concatenatedMatrix).name() << std::endl;
    std::cout << "Const Concatenated Matrix has sizeof: " << sizeof(const_concatenatedMatrix) << " bytes" << std::endl;
    std::cout << "Const Concatenated Matrix dimensions: " << const_concatenatedMatrix.dimensions << std::endl;
    std::cout << "Const Concatenated Matrix offsets: " << const_concatenatedMatrix.offsets << std::endl;
    std::cout << "Const Concatenated Matrix at position (0,0,0): " << const_concatenatedMatrix.at(0, 0, 0) << std::endl;
    // const_concatenatedMatrix.at(0, 0, 0) = 1.0f; // This line should not compile, as const_concatenatedMatrix is const
}

Matrix<int, DimensionOrder("12"), 5, 6> matrix_int;

void testSclices() {
    std::cout << "************************************************************************" << std::endl;
    std::cout << "Matrix slicing test function called." << std::endl;
    std::cout << "************************************************************************" << std::endl;

    auto slicedMatrix = slice<"2",3>(matrix_int,{1});
    std::cout << "Sliced Matrix created with type: " << typeid(slicedMatrix).name() << std::endl;
    std::cout << "Base Matrix dimensions: " << matrix_int.dimensions << std::endl;
    std::cout << "Sliced Matrix dimensions: " << slicedMatrix.dimensions << std::endl;
    std::cout << "Sliced Matrix slices: " << slicedMatrix.slices << std::endl;
    std::cout << "Sliced Matrix offsets: " << slicedMatrix.offset << std::endl;
    std::cout << "Sliced Matrix has sizeof: " << sizeof(slicedMatrix) << " bytes" << std::endl;

    for (unsigned long a = 0; a < slicedMatrix.dimensions[0]; a++) {
        for (unsigned long b = 0; b < slicedMatrix.dimensions[1]; b++) {
            slicedMatrix.at(a, b) = static_cast<int>(a * slicedMatrix.dimensions[1] + b + 1);
            std::cout << "Sliced Matrix at (" << a << "," << b << "): " << slicedMatrix.at(a, b) << std::endl;
            // std::cout << "Underlying Matrix data: " << matrix_int.data << std::endl;
        }
    }

    for (unsigned long a = 0; a < matrix_int.dimensions[0]; a++) {
        std::cout << "Matrix int row " << a << ": ";
        for (unsigned long b = 0; b < matrix_int.dimensions[1]; b++) {
            std::cout << matrix_int.at(a, b) << ", ";
        }
        std::cout << std::endl;
    }

    auto permutedSlicedMatrix = permute<"21">(slicedMatrix);
    std::cout << "Permuted Sliced Matrix created with type: " << typeid(permutedSlicedMatrix).name() << std::endl;
    std::cout << "Permuted Sliced Matrix dimensions: " << permutedSlicedMatrix.dimensions << std::endl;
    std::cout << "Permuted Sliced Matrix has sizeof: " << sizeof(permutedSlicedMatrix) << " bytes" << std::endl;

    for (unsigned long a = 0; a < permutedSlicedMatrix.dimensions[0]; a++) {
        for (unsigned long b = 0; b < permutedSlicedMatrix.dimensions[1]; b++) {
            permutedSlicedMatrix.at(a, b) = static_cast<int>(a * permutedSlicedMatrix.dimensions[1] + b + 1);
            std::cout << "Permuted Sliced Matrix at (" << a << "," << b << "): " << permutedSlicedMatrix.at(a, b) << std::endl;
            // std::cout << "Underlying Matrix data: " << matrix_int.data << std::endl;
        }
    }

    for (unsigned long a = 0; a < matrix_int.dimensions[0]; a++) {
        std::cout << "Matrix int row " << a << ": ";
        for (unsigned long b = 0; b < matrix_int.dimensions[1]; b++) {
            std::cout << matrix_int.at(a, b) << ", ";
        }
        std::cout << std::endl;
    }

    matrix_int.data               = std::array<int, 30>{}; // illegal reset, but for testing purposes
    auto concatenatedSlicedMatrix = concatenate<0>(slice<"12", 2, 2>(matrix_int), slice<"12",2, 2>(matrix_int,{3,3}));

    std::cout << "Concatenated Sliced Matrix created with type: " << typeid(concatenatedSlicedMatrix).name() << std::endl;
    std::cout << "Concatenated Sliced Matrix dimensions: " << concatenatedSlicedMatrix.dimensions << std::endl;
    std::cout << "Concatenated Sliced Matrix has sizeof: " << sizeof(concatenatedSlicedMatrix) << " bytes" << std::endl;

    for (unsigned long a = 0; a < concatenatedSlicedMatrix.dimensions[0]; a++) {
        for (unsigned long b = 0; b < concatenatedSlicedMatrix.dimensions[1]; b++) {
            concatenatedSlicedMatrix.at(a, b) = static_cast<int>(a * concatenatedSlicedMatrix.dimensions[1] + b + 1);
            std::cout << "Concatenated Sliced Matrix at (" << a << "," << b << "): " << concatenatedSlicedMatrix.at(a, b) << std::endl;
            // std::cout << "Underlying Matrix data: " << matrix_int.data << std::endl;
        }
    }

    for (unsigned long a = 0; a < matrix_int.dimensions[0]; a++) {
        std::cout << "Matrix int row " << a << ": ";
        for (unsigned long b = 0; b < matrix_int.dimensions[1]; b++) {
            std::cout << matrix_int.at(a, b) << ", ";
        }
        std::cout << std::endl;
    }
}

Matrix<int, "C", 10> matrix_epand;

void testExpand() {
    std::cout << "************************************************************************" << std::endl;
    std::cout << "Matrix expand test function called." << std::endl;
    std::cout << "************************************************************************" << std::endl;

    constexpr auto OrderA   = matrix_epand.order;
    constexpr auto neworder = OrderA + DimensionOrder("BS");

    std::cout << "Old Order: " << OrderA << std::endl;
    std::cout << "New order after expansion: " << neworder << std::endl;

    auto expandedMatrix = broadcast<"BS", {2, 3}>(matrix_epand);
    std::cout << "Expanded Matrix created with type: " << typeid(expandedMatrix).name() << std::endl;
    std::cout << "Expanded Matrix dimensions: " << expandedMatrix.dimensions << std::endl;
    std::cout << "Expanded Matrix has sizeof: " << sizeof(expandedMatrix) << " bytes" << std::endl;

    for (unsigned long a = 0; a < expandedMatrix.dimensions[0]; a++) {
        for (unsigned long b = 0; b < expandedMatrix.dimensions[1]; b++) {
            for (unsigned long c = 0; c < expandedMatrix.dimensions[2]; c++) {
                expandedMatrix.at(a, b, c) = static_cast<int>(a * expandedMatrix.dimensions[1] * expandedMatrix.dimensions[2] + b * expandedMatrix.dimensions[2] + c + 1);
                std::cout << "Expanded Matrix at (" << a << "," << b << "," << c << "): " << expandedMatrix.at(a, b, c) << std::endl;
            }
        }
    }
    std::cout << "Expanded Matrix data: " << expandedMatrix << std::endl;

    const auto const_expandedMatrix            = conditionalBroadcast<"BS", {2, 3}>(matrix_epand);
    const auto const_conditionalExpandedMatrix = conditionalBroadcast<"BS", {2, 3}>(const_expandedMatrix);

    std::cout << "Const Expanded Matrix created with type: " << typeid(const_expandedMatrix).name() << std::endl;
    std::cout << "Const Expanded Matrix has sizeof: " << sizeof(const_expandedMatrix) << " bytes" << std::endl;
    std::cout << "Const Expanded Matrix dimensions: " << const_expandedMatrix.dimensions << std::endl;
    std::cout << "Const Expanded Matrix has order: " << const_expandedMatrix.order << std::endl;
    std::cout << "Const Expanded Matrix at position (0,0,0): " << const_expandedMatrix.at(0, 0, 0) << std::endl;
    // const_expandedMatrix.at(0, 0, 0) = 1; // This line should not compile, as const_expandedMatrix is const
    std::cout << "Const Conditional Expanded Matrix created with type: " << typeid(const_conditionalExpandedMatrix).name() << std::endl;
    std::cout << "Const Conditional Expanded Matrix has sizeof: " << sizeof(const_conditionalExpandedMatrix) << " bytes" << std::endl;
    std::cout << "Const Conditional Expanded Matrix dimensions: " << const_conditionalExpandedMatrix.dimensions << std::endl;
    std::cout << "Const Conditional Expanded Matrix has order: " << const_conditionalExpandedMatrix.order << std::endl;
    std::cout << "Const Conditional Expanded Matrix at position (0,0,0): " << const_conditionalExpandedMatrix.at(0, 0, 0) << std::endl;
    // const_conditionalExpanded
}

Matrix<int, "ABC", 2, 3, 4> matrix_replace;

void testReplace() {
    std::cout << "************************************************************************" << std::endl;
    std::cout << "Matrix replace test function called." << std::endl;
    std::cout << "************************************************************************" << std::endl;

    auto replacedMatrix = replace<"AC", "CA">(matrix_replace);
    std::cout << "Replaced Matrix created with type: " << typeid(replacedMatrix).name() << std::endl;
    std::cout << "Replaced Matrix dimensions: " << replacedMatrix.dimensions << std::endl;
    std::cout << "Replaced Matrix has sizeof: " << sizeof(replacedMatrix) << " bytes" << std::endl;

    std::cout << "Base Matrix Order: " << matrix_replace.order << std::endl;
    std::cout << "Replaced Matrix Order: " << replacedMatrix.order << std::endl;

    for (unsigned long a = 0; a < replacedMatrix.dimensions[0]; a++) {
        for (unsigned long b = 0; b < replacedMatrix.dimensions[1]; b++) {
            for (unsigned long c = 0; c < replacedMatrix.dimensions[2]; c++) {
                replacedMatrix.at(a, b, c) = static_cast<int>(a * replacedMatrix.dimensions[1] * replacedMatrix.dimensions[2] + b * replacedMatrix.dimensions[2] + c + 1);
                std::cout << "Replaced Matrix at (" << a << "," << b << "," << c << "): " << replacedMatrix.at(a, b, c) << std::endl;
            }
        }
    }

    std::cout << "Replaced Matrix:" << replacedMatrix << std::endl;
}

void testNegativeMatrix() {
    std::cout << "************************************************************************" << std::endl;
    std::cout << "Negative Matrix test function called." << std::endl;
    std::cout << "************************************************************************" << std::endl;

    auto negativeMatrix = -matrix_replace;
    std::cout << "Negative Matrix created with type: " << typeid(negativeMatrix).name() << std::endl;
    std::cout << "Negative Matrix dimensions: " << negativeMatrix.dimensions << std::endl;
    std::cout << "Negative Matrix has sizeof: " << sizeof(negativeMatrix) << " bytes" << std::endl;

    for (unsigned long a = 0; a < negativeMatrix.dimensions[0]; a++) {
        for (unsigned long b = 0; b < negativeMatrix.dimensions[1]; b++) {
            for (unsigned long c = 0; c < negativeMatrix.dimensions[2]; c++) {
                // negativeMatrix.at(a, b, c) = static_cast<int>(a * negativeMatrix.dimensions[1] * negativeMatrix.dimensions[2] + b * negativeMatrix.dimensions[2] + c + 1);
                std::cout << "Negative Matrix at (" << a << "," << b << "," << c << "): " << negativeMatrix.at(a, b, c) << std::endl;
            }
        }
    }

    std::cout << "Negative Matrix:" << negativeMatrix << std::endl;
}

void testMatrixOperations() {
    std::cout << "************************************************************************" << std::endl;
    std::cout << "Matrix operations test function called." << std::endl;
    std::cout << "************************************************************************" << std::endl;

    auto matrixA = Matrix<int, DimensionOrder("CB"), 2, 2>{1, 2, 3, 4};

    loop([](int &a) { std::cout << a << std::endl; }, permute<"BC">(matrixA));

    std::cout << "Matrix data after loop operation: " << matrixA.data << std::endl;

    auto matrixB = materialize(permute<"BC">(matrixA));
    std::cout << "Materialized Matrix created with type: " << typeid(matrixB).name() << std::endl;
    std::cout << "Materialized Matrix has sizeof: " << sizeof(matrixB) << " bytes" << std::endl;
    std::cout << "Materialized Matrix dimensions: " << matrixB.dimensions << std::endl;
    std::cout << "Materialized Matrix data: " << matrixB.data << std::endl;

    auto result  = Matrix<int, "BO", 2, 4>();
    auto input   = Matrix<int, "BI", 2, 3>{1, 2, 3, 4, 5, 6};
    auto weights = Matrix<int, "IO", 3, 4>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    loop([](int &a, const int &b, const int &c) { a += b * c; }, // MAC
         broadcast<"I", {3}>(result),                            // results
         broadcast<"O", {4}>(input),                             // inputs
         broadcast<"B", {2}>(weights));                          // weights

    std::cout << "Result Matrix after loop operation: " << result.data << std::endl;

    loop(sum<int, int>, slice<"BO",2,3>(result), // results
         replace<"I", "O">(input));                    // inputs

    // loop(sum<int, int>,                 // errores sum operation  should not compile
    //      slice<{0, 2}, {0, 3}>(result), // results
    //      input);                        // inputs

    std::cout << "Result Matrix after sum operation: " << result.data << std::endl;
}

void testSplitMatrix() {
    std::cout << "************************************************************************" << std::endl;
    std::cout << "Matrix split test function called." << std::endl;
    std::cout << "************************************************************************" << std::endl;

    constexpr DimensionOrder order("CWH");
    constexpr auto           InsertedOrder = order.insert('W', "Ww");
    constexpr auto           RemovedOrder  = order.remove('W');
    std::cout << "Inserted Order: " << InsertedOrder << std::endl;
    std::cout << "Removed Order: " << RemovedOrder << std::endl;

    constexpr auto permutationIndexes = (order.remove('W') + "W").permutationOrderComputation<3>(order);
    std::cout << "Permutation indexes from CWH to CHW: " << permutationIndexes << std::endl;

    std::cout << "Start order: " << order.remove('W') + "Ww" << std::endl;
    constexpr auto permutationOrder2 = (InsertedOrder).permutationOrderComputation<4>(order.remove('W') + "Ww");
    std::cout << "Permutation indexes from CHWw to CWwH: " << permutationOrder2 << std::endl;

    auto testMatrix = Matrix<int, "WC", 24, 2>();
    int  counter    = 1;
    loop([&](int &a) { a = counter++; }, testMatrix);
    std::cout << "Test Matrix created with type: " << typeid(testMatrix).name() << std::endl;
    std::cout << "Test Matrix has sizeof: " << sizeof(testMatrix) << " bytes" << std::endl;
    std::cout << "Test Matrix dimensions: " << testMatrix.dimensions << std::endl;

    auto SplitMatrix = split<"W", "Ww1", 2, 3, 4>(testMatrix);
    std::cout << "Split Matrix created with type: " << typeid(SplitMatrix).name() << std::endl;
    // std::cout << "Split Matrix new dimemensions: " << SplitType::new_dimensions << std::endl;
    std::cout << "Split Matrix permutation_order_remove_to_back: " << SplitMatrix.permutation_order_remove_to_back << std::endl;
    std::cout << "Split Matrix permutation_order_new_to_correct: " << SplitMatrix.permutation_order_new_to_correct << std::endl;
    std::cout << "Split Matrix Base dimensions: " << decltype(SplitMatrix)::BaseMatrixTypeNoRef::dimensions << std::endl;

    std::cout << "Split Matrix Order: " << SplitMatrix.order << std::endl;
    std::cout << "Split Matrix dimensions: " << SplitMatrix.dimensions << std::endl;

    for (unsigned long a = 0; a < SplitMatrix.dimensions[0]; a++) {
        for (unsigned long b = 0; b < SplitMatrix.dimensions[1]; b++) {
            for (unsigned long c = 0; c < SplitMatrix.dimensions[2]; c++) {
                for (unsigned long d = 0; d < SplitMatrix.dimensions[3]; d++) {
                    SplitMatrix.at(a, b, c, d) = a + b + c + d + 1; // Just an example, you can set it to any value
                    std::cout << "Split Matrix at (" << a << "," << b << "," << c << "," << d << "): " << SplitMatrix.at(a, b, c, d) << std::endl;
                }
            }
        }
    }
}

void testCollapsedMatrix() {
    std::cout << "************************************************************************" << std::endl;
    std::cout << "Matrix collapsed test function called." << std::endl;
    std::cout << "************************************************************************" << std::endl;

    auto BaseMatrix      = Matrix<int, "CWH", 2, 3, 4>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
    auto CollapsedMatrix = collapse<"HC", "1">(BaseMatrix);
    using collType       = decltype(CollapsedMatrix);
    std::cout << "Collapsed Matrix created with type: " << typeid(collType).name() << std::endl;
    std::cout << "Collapsed Matrix has sizeof: " << sizeof(collType) << " bytes" << std::endl;
    std::cout << "Collapsed Matrix dimensions: " << collType::dimensions << std::endl;
    std::cout << "Collapsed Matrix dimensions_ordered_back: " << collType::dimensions_ordered_back << std::endl;
    std::cout << "Collapsed Matrix removed_dimensions: " << collType::removed_dimensions << std::endl;
    std::cout << "Collapsed Matrix order: " << collType::order << std::endl;
    std::cout << "Collapsed Matrix order_original: " << collType::order_original << std::endl;
    std::cout << "Collapsed Matrix order_original_at_back: " << collType::order_original_at_back << std::endl;
    std::cout << "Collapsed Matrix offsets: " << collType::offsets << std::endl;

    // for (unsigned long a = 0; a < BaseMatrix.dimensions[0]; a++) {
    //     for (unsigned long b = 0; b < BaseMatrix.dimensions[1]; b++) {
    //         for (unsigned long c = 0; c < BaseMatrix.dimensions[2]; c++) {
    //             auto new_b = BaseMatrix.dimensions[2]*b + c;
    //             std::cout << "Collapsed Matrix at (" << a << "," << new_b << "): " << CollapsedMatrix.at(a, new_b) << "  vs.  " << BaseMatrix.at(a, b, c) << std::endl;
    //             // std::cout << "Base Matrix at (" << a << "," << b << "," << c << "): " << BaseMatrix.at(a, b, c) << std::endl;
    //         }
    //     }
    // }
    for (unsigned long a = 0; a < CollapsedMatrix.dimensions[0]; a++) {
        for (unsigned long b = 0; b < CollapsedMatrix.dimensions[1]; b++) {
            CollapsedMatrix.at(a, b) = a * CollapsedMatrix.dimensions[1] + b + 1; // Just an example, you can set it to any value
            std::cout << "Collapsed Matrix at (" << a << "," << b << "): " << CollapsedMatrix.at(a, b) << std::endl;
        }
    }
    std::cout << "Collapsed Matrix data: " << CollapsedMatrix.data.data << std::endl;
}

void testReplication() {
    std::cout << "************************************************************************" << std::endl;
    std::cout << "Matrix replication test function called." << std::endl;
    std::cout << "************************************************************************" << std::endl;

    auto BaseMatrix       = Matrix<int, "CWH", 5, 1, 1>{1, 2, 3, 4, 5};
    auto replicatedMatrix = replicate<"W", {4}>(BaseMatrix);

    std::cout << "Replicated Matrix created with type: " << typeid(replicatedMatrix).name() << std::endl;
    std::cout << "Replicated Matrix has sizeof: " << sizeof(replicatedMatrix) << " bytes" << std::endl;
    std::cout << "Replicated Matrix dimensions: " << replicatedMatrix.dimensions << std::endl;
    std::cout << "Replicated Matrix order: " << replicatedMatrix.order << std::endl;

    for (unsigned long a = 0; a < replicatedMatrix.dimensions[0]; a++) {
        for (unsigned long b = 0; b < replicatedMatrix.dimensions[1]; b++) {
            for (unsigned long c = 0; c < replicatedMatrix.dimensions[2]; c++) {
                std::cout << "Replicated Matrix at (" << a << "," << b << "," << c << "): " << replicatedMatrix.at(a, b, c) << std::endl;
            }
        }
    }

    auto conditionalReplicatedMatrix = conditionalReplicate<"E", {4}>(BaseMatrix);
    std::cout << "Conditional Replicated Matrix created with type: " << typeid(conditionalReplicatedMatrix).name() << std::endl;
    std::cout << "Conditional Replicated Matrix has sizeof: " << sizeof(conditionalReplicatedMatrix) << " bytes" << std::endl;
    std::cout << "Conditional Replicated Matrix dimensions: " << conditionalReplicatedMatrix.dimensions << std::endl;
    std::cout << "Conditional Replicated Matrix order: " << conditionalReplicatedMatrix.order << std::endl;
    for (unsigned long a = 0; a < conditionalReplicatedMatrix.dimensions[0]; a++) {
        for (unsigned long b = 0; b < conditionalReplicatedMatrix.dimensions[1]; b++) {
            for (unsigned long c = 0; c < conditionalReplicatedMatrix.dimensions[2]; c++) {
                std::cout << "Conditional Replicated Matrix at (" << a << "," << b << "," << c << "): " << conditionalReplicatedMatrix.at(a, b, c) << std::endl;
            }
        }
    }
}

void testConst() {
    std::cout << "************************************************************************" << std::endl;
    std::cout << "Matrix const test function called." << std::endl;
    std::cout << "************************************************************************" << std::endl;

    const auto testMatrix = Matrix<int, "BC", 1, 3>();

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
    const auto const_permuted_matrix                 = permute<"CB">(testMatrix);
    const auto const_concatenated_matrix             = concatenate<0>(testMatrix, testMatrix);
    const auto const_sliced_matrix                   = slice<"BC", 1, 2>(testMatrix);
    const auto const_broadcast_matrix                = broadcast<"D", {2}>(testMatrix);
    const auto const_conditional_broadcast_matrix_1  = conditionalBroadcast<"D", {2}>(testMatrix);
    const auto const_conditional_broadcast_matrix_2  = conditionalBroadcast<"C", {2}>(testMatrix); // does nothing
    const auto const_replicated_matrix               = replicate<"B", {2}>(testMatrix);
    const auto const_conditional_replicated_matrix_1 = conditionalReplicate<"B", {2}>(testMatrix);
    const auto const_conditional_replicated_matrix_2 = conditionalReplicate<"E", {2}>(testMatrix); // does nothing
    const auto const_replaced_matrix                 = replace<"B", "D">(testMatrix);
    const auto const_conditional_replaced_matrix_1   = conditionalReplace<"B", "D">(testMatrix);
    const auto const_conditional_replaced_matrix_2   = conditionalReplace<"D", "B">(testMatrix); // does nothing
    const auto const_split_matrix                    = split<"C", "Cw", 1, 3>(testMatrix);
    const auto const_collapsed_matrix                = collapse<"C", "1">(testMatrix);
    const auto const_negative_matrix                 = -testMatrix;
#pragma clang diagnostic pop

    auto testMatrix1D = Matrix<int, "B", 3>();
    auto testMatrix2D = Matrix<int, "BC", 1, 3>();
    auto testMatrix3D = Matrix<int, "BCW", 3, 3, 3>();

    // test all read and writes
    auto permuted_matrix                 = permute<"CB">(testMatrix2D);
    auto concatenated_matrix             = concatenate<0>(testMatrix2D, testMatrix2D);
    auto sliced_matrix                   = slice<"BC", 1, 2>(testMatrix2D);
    auto broadcast_matrix                = broadcast<"D", {2}>(testMatrix1D);
    auto conditional_broadcast_matrix_1  = conditionalBroadcast<"D", {2}>(testMatrix1D);
    auto conditional_broadcast_matrix_2  = conditionalBroadcast<"C", {2}>(testMatrix2D); // does nothing
    auto replicated_matrix               = replicate<"B", {2}>(testMatrix2D);
    auto conditional_replicated_matrix_1 = conditionalReplicate<"B", {2}>(testMatrix2D);
    auto conditional_replicated_matrix_2 = conditionalReplicate<"E", {2}>(testMatrix2D); // does nothing
    auto replaced_matrix                 = replace<"B", "D">(testMatrix2D);
    auto conditional_replaced_matrix_1   = conditionalReplace<"B", "D">(testMatrix2D);
    auto conditional_replaced_matrix_2   = conditionalReplace<"D", "B">(testMatrix2D); // does nothing
    auto split_matrix                    = split<"B", "Cw", 1, 3>(testMatrix1D);
    auto collapsed_matrix                = collapse<"CW", "1">(testMatrix3D);
    auto negative_matrix                 = -testMatrix2D;

    std::size_t index = 10;
    auto        test  = [&](auto &t) {
        std::cout << "Test function called." << std::endl;
        std::cout << "Test function type: " << typeid(t).name() << std::endl;
        t.template at<std::remove_cvref_t<decltype(t)>::order>(0, 0) = index;
        std::cout << "Matrix at (0,0): " << t.at(0, 0) << " vs " << index << std::endl;
        index++;
    };
    test(permuted_matrix);
    test(concatenated_matrix);
    test(sliced_matrix);
    test(broadcast_matrix);
    test(conditional_broadcast_matrix_1);
    test(replicated_matrix);
    test(conditional_replicated_matrix_1);
    test(replaced_matrix);
    test(conditional_replaced_matrix_1);
    test(split_matrix);
    test(collapsed_matrix);
    // test(negative_matrix);
    test(conditional_broadcast_matrix_2);
    test(conditional_replicated_matrix_2);
    test(conditional_replaced_matrix_2);
    
    auto &testMatrix1D_ref = testMatrix1D;
    auto &testMatrix2D_ref = testMatrix2D;
    auto &testMatrix3D_ref = testMatrix3D;

    // test all read and writes
    auto ref_permuted_matrix                 = permute<"CB">(testMatrix2D_ref);
    auto ref_concatenated_matrix             = concatenate<0>(testMatrix2D_ref, testMatrix2D_ref);
    auto ref_sliced_matrix                   = slice<"BC", 1, 2>(testMatrix2D_ref);
    auto ref_broadcast_matrix                = broadcast<"D", {2}>(testMatrix1D_ref);
    auto ref_conditional_broadcast_matrix_1  = conditionalBroadcast<"D", {2}>(testMatrix1D_ref);
    auto ref_conditional_broadcast_matrix_2  = conditionalBroadcast<"C", {2}>(testMatrix2D_ref); // does nothing
    auto ref_replicated_matrix               = replicate<"B", {2}>(testMatrix2D_ref);
    auto ref_conditional_replicated_matrix_1 = conditionalReplicate<"B", {2}>(testMatrix2D_ref);
    auto ref_conditional_replicated_matrix_2 = conditionalReplicate<"E", {2}>(testMatrix2D_ref); // does nothing
    auto ref_replaced_matrix                 = replace<"B", "D">(testMatrix2D_ref);
    auto ref_conditional_replaced_matrix_1   = conditionalReplace<"B", "D">(testMatrix2D_ref);
    auto ref_conditional_replaced_matrix_2   = conditionalReplace<"D", "B">(testMatrix2D_ref); // does nothing
    auto ref_split_matrix                    = split<"B", "Cw", 1, 3>(testMatrix1D_ref);
    auto ref_collapsed_matrix                = collapse<"CW", "1">(testMatrix3D_ref);
    auto ref_negative_matrix                 = -testMatrix2D_ref;

    test(ref_permuted_matrix);
    test(ref_concatenated_matrix);
    test(ref_sliced_matrix);
    test(ref_broadcast_matrix);
    test(ref_conditional_broadcast_matrix_1);
    test(ref_conditional_broadcast_matrix_2);
    test(ref_replicated_matrix);
    test(ref_conditional_replicated_matrix_1);
    test(ref_conditional_replicated_matrix_2);
    test(ref_replaced_matrix);
    test(ref_conditional_replaced_matrix_1);
    test(ref_conditional_replaced_matrix_2);
    test(ref_split_matrix);
    test(ref_collapsed_matrix);
    // test(ref_negative_matrix);

    std::cout << "Testing const matrices:" << std::endl;
    auto non_const = Matrix<int, "BC", 2, 3>();
    const auto con = Matrix<int, "BC", 2,3>();

    auto non_const_permuted = permute<"CB">(non_const);
    auto const_permuted = permute<"CB">(con);

    std::cout << "const_permuted type: " << human_readable_type<decltype(const_permuted)> << std::endl;

    non_const_permuted.at(0, 0) = 1;
    // const_permuted.at(0, 0) = 2; // This line should not compile, as const_permuted is const
    auto tmp = const_permuted.at(0, 0);

    const auto const_non_const_permuted = permute<"CB">(non_const);
    // const_non_const_permuted.at(0, 0) = 3; // This line should not compile, as const_non_const_permuted is const
    auto tmp2 = const_non_const_permuted.at(0, 0);
}

void testZeroSize() {
    std::cout << "************************************************************************" << std::endl;
    std::cout << "Matrix zero size test function called." << std::endl;
    std::cout << "************************************************************************" << std::endl;

    auto zeroSizeMatrix = Matrix<int, "BC", 0, 0>();
    std::cout << "Zero Size Matrix created with type: " << typeid(zeroSizeMatrix).name() << std::endl;
    std::cout << "Zero Size Matrix has sizeof: " << sizeof(zeroSizeMatrix) << " bytes" << std::endl;
    std::cout << "Zero Size Matrix dimensions: " << zeroSizeMatrix.dimensions << std::endl;

    // Attempting to access elements in a zero-size matrix should not compile
    // zeroSizeMatrix.at(0, 0) = 1; // Uncommenting this line should cause a compilation error

    auto testMatrix = Matrix<int, "BC", 1, 3>();
    auto zeroSizeSlicedMatrix = slice<"BC", 0, 2>(testMatrix);
    std::cout << "Zero Size Sliced Matrix created with type: " << typeid(zeroSizeSlicedMatrix).name() << std::endl;
    std::cout << "Zero Size Sliced Matrix has sizeof: " << sizeof(zeroSizeSlicedMatrix) << " bytes" << std::endl;
    std::cout << "Zero Size Sliced Matrix dimensions: " << zeroSizeSlicedMatrix.dimensions << std::endl;

}

void testOverrides(){
    std::cout << "************************************************************************" << std::endl;
    std::cout << "Matrix overrides test function called." << std::endl;
    std::cout << "************************************************************************" << std::endl;

    using BaseMatrixType = Matrix<int, "QBC", 7,2, 3>;
    using OverrittenDimension = OverrideDimensionMatrix<BaseMatrixType, "C", 17>;
    using OverrittenType = OverrideTypeMatrix<BaseMatrixType,float>;
    using OverrittenRemoveDimension = OverrideRemoveDimensionMatrix<BaseMatrixType, "B">;

    std::cout << "Base Matrix Type: " << human_readable_type<BaseMatrixType> << std::endl;
    std::cout << "Overritten Dimension Type: " << human_readable_type<OverrittenDimension> << std::endl;
    std::cout << "Overritten Type Type: " << human_readable_type<OverrittenType> << std::endl;
    std::cout << "Overritten Remove Dimension Type: " << human_readable_type<OverrittenRemoveDimension> << std::endl;
}

int main() {
    std::cout << "Matrix test file is included successfully." << std::endl;

    // testMatrix();
    // testPermutation();
    // testConcatenation();
    // testSclices();
    // testExpand();
    // testReplace();
    // testNegativeMatrix();
    // testMatrixOperations();
    // testSplitMatrix();
    // testCollapsedMatrix();
    // testReplication();
    // testConst();
    // testZeroSize();
    testOverrides();
    return 0;
}