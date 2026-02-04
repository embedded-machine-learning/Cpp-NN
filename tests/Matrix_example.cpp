#include "../include/Matrix.hpp"
#include <iostream>

#include "../include/helpers/human_readable_types.hpp"
#include "../include/helpers/print.hpp"

int main() {

    Matrix<float, DimensionOrder("12"), 1, 2> A;
    Matrix<float, "12", 1, 2>                 B; // same as above, but with string literal
    using AType = decltype(A);

    auto C = permute<"21">(A);           // Permute the matrix A to have the dimensions in the order "21"
    auto D = concatenate<0>(A, B);       // Concatenate the matrices A and B along dimension 0 which is the first dimension TODO: change to named dimensions
    auto E = slice<"1", 1>(A, {0});      // Slice the matrix A along dimension "1" of size 1 with an offset of 0
    auto F = broadcast<"34", {2, 3}>(A); // Broadcast the matrix A to have dimensions "1234" with new dimension lengths 2 and 3
    auto G = replicate<"1", {4}>(A);     // Replicate the matrix A to have dimensions "12" with dimension lengths 4, 2
    auto H = replace<"12", "34">(A);     // Replace the dimensions "12" of matrix A with "34"
    auto I = split<"2", "34", 1, 2>(A);  // Split the dimension "2" of matrix A into two dimensions "3" and "4" with lengths 1 and 2 product of the split dimensions must match the original dimension
    auto J = collapse<"12", "Q">(A);     // Collapse the dimensions "12" of matrix A into a single dimension "Q"
    // Negative of a matrix
    auto K = -A;         // Negate the matrix A, this will return a NegativeMatrix
    auto L = fuse(A, B); // Fuse multiple matrices into a single matrix, the resulting matrix will have a tuple of values at each index

    using RefMatrix = ReferencedMatrixType<AType>; // Doesn't do anything, just used for conditional type changes

    // overrides only usable for type information, not for data

    using OvDim   = OverrideDimensionMatrix<AType, "1", 10>;   // Changes the Dimension at Order position to new size
    using OvType  = OverrideTypeMatrix<AType, int>;            // Changes the value_type of the matrix to OverrideType
    using OvRmDim = OverrideRemoveDimensionMatrix<AType, "2">; // Removes the dimension RemoveOrder from the matrix, if it exists, and returns a new matrix with the remaining dimensions

    std::cout << "Results of Matrix example:\n" << std::endl;
    std::cout << "Matrix A type:                         " << human_readable_type<AType> << std::endl;
    std::cout << "Matrix B type:                         " << human_readable_type<decltype(B)> << std::endl;
    std::cout << "Permuted Matrix C type:                " << human_readable_type<decltype(C)> << std::endl;
    std::cout << "Concatenated Matrix D type:            " << human_readable_type<decltype(D)> << std::endl;
    std::cout << "Sliced Matrix E type:                  " << human_readable_type<decltype(E)> << std::endl;
    std::cout << "Broadcasted Matrix F type:             " << human_readable_type<decltype(F)> << std::endl;
    std::cout << "Replicated Matrix G type:              " << human_readable_type<decltype(G)> << std::endl;
    std::cout << "Replaced Matrix H type:                " << human_readable_type<decltype(H)> << std::endl;
    std::cout << "Split Matrix I type:                   " << human_readable_type<decltype(I)> << std::endl;
    std::cout << "Collapsed Matrix J type:               " << human_readable_type<decltype(J)> << std::endl;
    std::cout << "Negative Matrix K type:                " << human_readable_type<decltype(K)> << std::endl;
    std::cout << "Fused Matrix L type:                   " << human_readable_type<decltype(L)> << std::endl;
    std::cout << "Referenced Matrix type:                " << human_readable_type<RefMatrix> << std::endl;
    std::cout << "Override Dimension Matrix type:        " << human_readable_type<OvDim> << std::endl;
    std::cout << "Override Type Matrix type:             " << human_readable_type<OvType> << std::endl;
    std::cout << "Override Remove Dimension Matrix type: " << human_readable_type<OvRmDim> << std::endl;

    return 0;
}
