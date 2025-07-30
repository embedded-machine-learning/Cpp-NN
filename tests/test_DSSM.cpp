#include <cmath>
#include <iostream>

#include "../include/Matrix.hpp"
#include "../include/helpers/print.hpp"
#include "../include/helpers/extended_matrix_ops.hpp"

#include "../include/layers/DSSM.hpp"

void test_DSSM() {
    using namespace layers;

    constexpr std::size_t batch_size        = 1;
    constexpr std::size_t sequence_length   = 2;
    constexpr std::size_t input_channels    = 4;
    constexpr std::size_t hidden_channels   = 5;
    constexpr std::size_t output_channels   = 6;

    using AMatrixType     = Matrix<Complex<float>, "C", hidden_channels>;
    using BMatrixType     = Matrix<Complex<float>, "IO", input_channels, hidden_channels>;
    using BBiasMatrixType = Matrix<Complex<float>, "C", hidden_channels>;
    using CMatrixType     = Matrix<Complex<float>, "IO", hidden_channels, output_channels>;
    using CBiasMatrixType = Matrix<float, "C", output_channels>;
    using DMatrixType     = Matrix<float, "IO", 0, 0>; // Empty matrix, not used
    using SkipMatrixType  = Matrix<float, "IO", input_channels, output_channels>; // Trainable skip connection+
    // using SkipMatrixType  = Matrix<float, "E", 1>; // Trainable skip connection+

    using InputMatrixType  = Matrix<float, "BCS",batch_size, input_channels, sequence_length>;
    using OutputMatrixType = Matrix<float, "BCS",batch_size, output_channels, sequence_length>;


    AMatrixType amatrix;
    BMatrixType bmatrix;
    BBiasMatrixType bbias;  
    CMatrixType cmatrix;
    CBiasMatrixType cbias;
    DMatrixType dmatrix; // Not used, but required for the type
    SkipMatrixType skip_matrix; 

    // Initialize matrices with random values
    randomize(amatrix);
    randomize(bmatrix); 
    randomize(bbias);
    randomize(cmatrix);
    randomize(cbias);
    randomize(dmatrix); // Not used, but required for the type
    randomize(skip_matrix); // Trainable skip connection    

    auto bmatrix_split = functions::linear::weightSubBio<3,3>(bmatrix);
    auto cmatrix_split = functions::linear::weightSubBio<3,3>(cmatrix);
    auto skip_matrix_split = functions::linear::weightSubBio<3,3>(skip_matrix);


    // auto dssm = DSSM(
    //     amatrix, bmatrix, bbias, cmatrix, cbias, dmatrix, skip_matrix,
    //     [](const auto &x) { return std::max(x,0.f); } // Example activation function
    // );
    auto dssm = DSSM(
        amatrix, bmatrix_split, bbias, cmatrix_split, cbias, dmatrix, skip_matrix_split,
        [](const auto &x) { return std::max(x,0.f); } // Example activation function
    );

    using DSSMType = decltype(dssm);

    Matrix<char, "E", DSSMType::memory_buffer<InputMatrixType>> buffer;
    Matrix<char, "E", DSSMType::memory_permanent<InputMatrixType>> permanent;

    std::cout << "DSSM size:" << sizeof(dssm) << " bytes\n";
    std::cout << "DSSM Memory Requirements:\n";
    std::cout << "Buffer Memory: " << buffer << "\n";
    std::cout << "Permanent Memory: " << permanent << "\n";

    InputMatrixType input;
    OutputMatrixType output;

    randomize(input);

    // Forward pass
    dssm(input, output, buffer, permanent);
    // Print the output
    std::cout << "Output Matrix:\n";
    auto output_collapsed = collapse<"BS","B">(output);
    print2DMatrix(output_collapsed);
    // print2DMatrix(output);

   
}

int main() {
    test_DSSM();
    return 0;
}