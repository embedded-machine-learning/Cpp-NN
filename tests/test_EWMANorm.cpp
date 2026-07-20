#include <cmath>
#include <iostream>

#include "../include/Matrix.hpp"
#include "../include/MatrixOperations.hpp"
#include "../include/helpers/extended_matrix_ops.hpp"
#include "../include/helpers/print.hpp"
#include "../include/helpers/human_readable_types.hpp"
#include "../include/layers/EWMAGlobalNorm.hpp"
#include "../include/layers/Sequence.hpp"

Matrix<char,"E",10000> buffer;
Matrix<char,"E",10000> permanent;


void testEWMAGlobalNorm() {
    std::cout << "************************************************************************" << std::endl;
    std::cout << "EWMAGlobalNorm test function called." << std::endl;
    std::cout << "************************************************************************" << std::endl;

    using Type                      = float;
    constexpr Dim_size_t input_channels     = 5;
    constexpr Dim_size_t batch_size = 2;
    constexpr Dim_size_t sequence_length = 10;


    Matrix<Type, "BSC", batch_size,sequence_length, input_channels>  input_matrix;
    Matrix<Type, "E", 1>     weight_matrix{0.8f};
    decltype(input_matrix) output_matrix;
    
    matrixSet(input_matrix, 1.0f);
    input_matrix.at(0, 0, 0) = 10.0f;
    input_matrix.at(0, 0, 1) = -20.0f;

    auto NormLayer = layers::EWMAGlobalNorm<"S", "C", Type>(weight_matrix,
                                          [](auto &ret) { std::cout << "Input Reset Lambda Called" << std::endl; ret = 0; },
                                          [](auto &ret, const auto &input) {  std::cout << "Input Lambda Called" << std::endl; ret = std::max(ret, std::abs(input)); },
                                          [](auto &ret, const auto &input, const auto &weights) {  std::cout << "Smoothing Lambda Called" << std::endl; ret = ret * weights + (static_cast<decltype(weights)>(1) - weights) * input; },
                                          [](auto &ret, const auto &permanent, [[maybe_unused]] const auto &input) { std::cout << "Output Lambda Called" << std::endl; ret = input / (permanent + 1e-6); });

    using StateMatrixType = decltype(NormLayer)::template StateMatrixType<decltype(input_matrix)>;
    auto &state_matrix = *reinterpret_cast<StateMatrixType *>(&permanent.data[0]);
        state_matrix.at(0,0) = 1.0f; // Initialize the state matrix to 1.0f
    // matrixSet(state_matrix, 1.0f);


    NormLayer(input_matrix, output_matrix, buffer, permanent);

    std::cout << "Input Matrix: " << input_matrix << std::endl;
    std::cout << "Output Matrix: " << output_matrix << std::endl;

    printNDMatrix(input_matrix);
    printNDMatrix(output_matrix);
}

int main() {
    testEWMAGlobalNorm();
    return 0;
}