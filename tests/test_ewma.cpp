#include <cmath>
#include <iostream>

#include "../include/Matrix.hpp"
#include "../include/helpers/print.hpp"
#include "../include/helpers/extended_matrix_ops.hpp"

#include "../include/layers/EWMAReduction.hpp"



int main() {
    using namespace layers;

    constexpr std::size_t batch_size        = 1;
    constexpr std::size_t sequence_length   = 20;
    constexpr std::size_t input_channels    = 4;
    constexpr std::size_t output_channels   = 4;

    using InputMatrixType  = Matrix<float, "BCS",batch_size, input_channels, sequence_length>;
    using OutputMatrixType = Matrix<float, "BCS",batch_size, output_channels, sequence_length>;

    using WeightMatrixType = Matrix<float, "E", 1>;

    InputMatrixType input_matrix;
    WeightMatrixType weight_matrix{1.0f}; // Example weight for EMA
    auto ewma_layer = EWMAReduction<"S">(weight_matrix);

    auto permanent_buffer = Matrix<char, "E", ewma_layer.template memory_permanent<InputMatrixType>>{};
    auto temporary_buffer = Matrix<char, "E", ewma_layer.template memory_buffer<InputMatrixType>>{};

    // Initialize input matrix with random values
    randomize(input_matrix);
    printNDMatrix(input_matrix);

    OutputMatrixType output_matrix;

    // Run the EWMA reduction layer
    ewma_layer(input_matrix, output_matrix, temporary_buffer, permanent_buffer);

    // Print the output
    printNDMatrix(output_matrix);

    return 0;
}