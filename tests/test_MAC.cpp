#include <complex>
#include <iostream>

#include "../include/Matrix.hpp"
#include "../include/MatrixOperations.hpp"
#include "../include/helpers/extended_matrix_ops.hpp"
#include "../include/helpers/human_readable_types.hpp"
#include "../include/helpers/print.hpp"
#include "../include/MAC.hpp"
#include "../include/hardware/AVX2.hpp"

void testMAC() {
    std::cout << "************************************************************************" << std::endl;
    std::cout << "MAC test function called." << std::endl;
    std::cout << "************************************************************************" << std::endl;
    using Type    = float;
    using MACType = DefaultMACOperation<Type, Type, Type>;

    constexpr Dim_size_t batch_channels  = 3;
    constexpr Dim_size_t input_channels  = 4;
    constexpr Dim_size_t output_channels = 13;

    using InputMatrixType  = Matrix<Type, "bi", batch_channels, input_channels>;
    using OutputMatrixType = Matrix<Type, "bo", batch_channels, output_channels>;
    using WeightMatrixType = Matrix<Type, "ibo", input_channels, batch_channels, output_channels>;

    InputMatrixType  input;
    OutputMatrixType output;
    WeightMatrixType weights;

    randomize(input);
    randomize(weights);

    OutputMatrixType comparison_output;
    loop(MACType::lambda, broadcast<"i", {input_channels}>(comparison_output), broadcast<"o", {output_channels}>(input), weights);

    std::cout << "Running MAC operation..." << std::endl;
    MACType::op(output, input, weights);
    std::cout << "Done with MAC operation." << std::endl;

    auto                     result_cmp = matrixCmp(output, comparison_output);
    Matrix<long, "bo", 1, 1> result_cmp_long;
    matrixSum(replicate<"bo", {batch_channels, output_channels}>(result_cmp_long), result_cmp);
    std::cout << "Comparison result: " << result_cmp_long.at(0, 0) << " should be: " << OutputMatrixType::dimensions[0] * OutputMatrixType::dimensions[1] << std::endl << std::endl;
    print2DMatrix(result_cmp);
}

void testComplexMAC() {
    std::cout << "************************************************************************" << std::endl;
    std::cout << "Complex MAC test function called." << std::endl;
    std::cout << "************************************************************************" << std::endl;
    using Type    = std::complex<float>;
    using MACType = DefaultMACOperation<Type, Type, Type>;

    constexpr Dim_size_t batch_channels  = 1;
    constexpr Dim_size_t input_channels  = 1;
    constexpr Dim_size_t output_channels = 4;

    using InputMatrixType  = Matrix<Type, "bi", batch_channels, input_channels>;
    using OutputMatrixType = Matrix<Type, "bo", batch_channels, output_channels>;
    using WeightMatrixType = Matrix<Type, "ibo", input_channels, batch_channels, output_channels>;

    InputMatrixType  input;
    OutputMatrixType output;
    WeightMatrixType weights;

    randomize(input);
    randomize(weights);

    OutputMatrixType comparison_output;
    loop(MACType::lambda, broadcast<"i", {input_channels}>(comparison_output), broadcast<"o", {output_channels}>(input), weights);

    std::cout << "Running MAC operation..." << std::endl;
    MACType::op(output, input, weights);
    std::cout << "Done with MAC operation." << std::endl;

    auto                     result_cmp = matrixCmp(output, comparison_output);
    Matrix<long, "bo", 1, 1> result_cmp_long;
    matrixSum(replicate<"bo", {batch_channels, output_channels}>(result_cmp_long), result_cmp);
    std::cout << "Comparison result: " << result_cmp_long.at(0, 0) << " should be: " << OutputMatrixType::dimensions[0] * OutputMatrixType::dimensions[1] << std::endl << std::endl;
    print2DMatrix(result_cmp);
    // print2DMatrix(output);
}

void testRealingComplexMAC() {
    std::cout << "************************************************************************" << std::endl;
    std::cout << "Realing Complex MAC test function called." << std::endl;
    std::cout << "************************************************************************" << std::endl;
    using Type    = Complex<float>;
    using MACType = RealResultMACOperation<Type, Type, Type::value_type>;

    constexpr Dim_size_t batch_channels  = 2;
    constexpr Dim_size_t input_channels  = 1;
    constexpr Dim_size_t output_channels = 4*5+2+1;

    using InputMatrixType  = Matrix<Type, "bi", batch_channels, input_channels>;
    using OutputMatrixType = Matrix<Type::value_type, "bo", batch_channels, output_channels>;
    using WeightMatrixType = Matrix<Type, "ibo", input_channels, batch_channels, output_channels>;

    InputMatrixType  input;
    OutputMatrixType output;
    WeightMatrixType weights;

    randomize(input);
    randomize(weights);
    randomize(output); // random bias

    auto accumulation = materialize(output, MACType::pre_processing);
    std::cout << "Accumulation Type is: " <<  human_readable_type<typename decltype(accumulation)::value_type> << std::endl;

    auto comparison_accumulation = materialize(accumulation);
    loop(MACType::lambda, broadcast<"i", {input_channels}>(comparison_accumulation), broadcast<"o", {output_channels}>(input), weights);
    auto comparison_output = materialize(comparison_accumulation, MACType::post_processing);

    std::cout << "Running MAC operation..." << std::endl;
    MACType::op(accumulation, input, weights);
    loop([](auto &a, const auto &b) { a = MACType::post_processing(b); }, output, accumulation);
    std::cout << "Done with MAC operation." << std::endl;

    auto                     result_cmp = matrixCmp(output, comparison_output);
    Matrix<long, "bo", 1, 1> result_cmp_long;
    matrixSum(replicate<"bo", {batch_channels, output_channels}>(result_cmp_long), result_cmp);
    std::cout << "Comparison result: " << result_cmp_long.at(0, 0) << " should be: " << OutputMatrixType::dimensions[0] * OutputMatrixType::dimensions[1] << std::endl << std::endl;
    print2DMatrix(result_cmp);
}

int main() {
    // testMAC();
    // testComplexMAC();
    testRealingComplexMAC();
    return 0;
}