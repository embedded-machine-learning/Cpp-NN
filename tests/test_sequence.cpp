#include <iostream>

#include "../include/helpers/human_readable_types.hpp"
#include "../include/helpers/print.hpp"
#include "../include/layers/Sequence.hpp"
#include "../include/Matrix.hpp"
#include "../include/helpers/extended_matrix_ops.hpp"
#include "../include/layers/BaseLayer.hpp"
#include "../include/layers/Linear.hpp"

void testSequence() {
    using namespace layers;

    // Example usage of Sequence with BaseLayer
    using Seq = Sequence<BaseLayer, BaseLayer>;
    Seq seq(BaseLayer{}, BaseLayer{});

    Matrix<float, "IO", 10, 20> weights1;
    Matrix<float, "OI", 10, 20> weights2;
    Matrix<float, "E", 1>       bias;

    randomize(weights1);
    randomize(weights2);
    randomize(bias);

    auto weights2_split = functions::linear::weightSubBio<3, 3>(weights2);

    auto seq2 = Sequence( // For Linebreak
            Linear<float, 4>(weights1, bias), 
            Linear<float, 3>(weights2_split, bias),
            Linear<float, 4>(weights1, bias), 
            Linear<float, 3>(weights2_split, bias),
            Linear<float, 4>(weights1, bias), 
            Linear<float, 3>(weights2_split, bias)
        );
    // auto seq2  = Sequence(Linear(weights1, bias));
    using Seq2 = decltype(seq2);

    // Example input, output, buffers,  matrices
    using InputMatrixType  = Matrix<float, "BC", 5, 10>;
    using OutputMatrixType = Seq2::OutputMatrix<InputMatrixType>;

    constexpr auto buffer_size    = Seq2::memory_buffer<InputMatrixType>;
    constexpr auto permanent_size = Seq2::memory_permanent<InputMatrixType>;
    std::cout << "Buffer size: " << buffer_size << std::endl;
    std::cout << "Permanent size: " << permanent_size << std::endl;

    Matrix<char, "E", buffer_size>    buffer;
    Matrix<char, "E", permanent_size> permanent;

    auto buffer_tmp = permute<"E">(buffer);

    InputMatrixType  &input  = *(seq2.getInputMatrix<InputMatrixType>(buffer));
    OutputMatrixType &output = *(seq2.getOutputMatrix<InputMatrixType>(buffer));
    // print2DMatrix(input);
    // print2DMatrix(output);
    randomize(input);
    // print2DMatrix(input);
    // print2DMatrix(output);  // should change aswell on even numbers of layers

    // Call the sequence layer
    seq2(input, output, buffer, permanent);
    // seq2(input, output, buffer_tmp, permanent); // Error
    std::cout << "Sequence layer executed successfully." << std::endl;
    std::cout << "Output Matrix: " << output << std::endl;
    print2DMatrix(output);

    using MemoryPlaning                   = Seq2::CurrentMemoryPlaning<decltype(input)>;
    constexpr auto memory_minimal         = MemoryPlaning::total_memory_minimal;
    constexpr auto memory_index_locations = MemoryPlaning::template memory_index_locations<memory_minimal, 0>;

    std::cout << "Memory minimal size: " << memory_minimal << std::endl;
    std::cout << "Memory index locations: " << human_readable_type<decltype(memory_index_locations)> << std::endl;
    std::cout << memory_index_locations << std::endl;
}

int main() {
    std::cout << "Sequence test file is included successfully." << std::endl;
    testSequence();
    return 0;
}