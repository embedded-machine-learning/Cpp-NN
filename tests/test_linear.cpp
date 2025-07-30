#include <cmath>
#include <iostream>

#include "../include/Matrix.hpp"
#include "../include/MatrixOperations.hpp"
#include "../include/helpers/extended_matrix_ops.hpp"
#include "../include/helpers/print.hpp"
#include "../include/functions/linear.hpp"
#include "../include/layers/BaseLayer.hpp"
#include "../include/layers/Linear.hpp"

// #include "../hardware/AVX2.hpp"

// template <typename T>
// constexpr auto cmp = [](bool &ret, const T &a, const T &b) { ret = 2 * std::fabs(a - b) / (std::fabs(a) + std::fabs(b)) < 1e-6; };

// template <>
// constexpr auto cmp<bool> = [](bool &ret, const bool &a, const bool &b) { ret = a == b; };

// template <>
// constexpr auto cmp<int> = [](bool &ret, const int &a, const int &b) { ret = a == b; };

void testDefaultBeh() {
    std::cout << "************************************************************************" << std::endl;
    std::cout << "Default Matrix test function called." << std::endl;
    std::cout << "************************************************************************" << std::endl;

    using Type                      = int;
    constexpr Dim_size_t inputs     = 5;
    constexpr Dim_size_t outputs    = 11;
    constexpr Dim_size_t batch_size = 7;

    Matrix<Type, "BC", batch_size, inputs>  input_matrix;
    Matrix<Type, "IO", inputs, outputs>     weight_matrix;
    Matrix<Type, "BC", batch_size, outputs> output_matrix;
    Matrix<Type, "BC", batch_size, outputs> bias_matrix;

    randomize(input_matrix);
    randomize(weight_matrix);
    // randomize(output_matrix);
    randomize(bias_matrix);

    // int counter=0;
    // loop([&](auto& a){a=counter++;},input_matrix);
    // loop([&](auto& a){a=counter++;},weight_matrix);
    // loop([&](auto& a){a=counter++;},bias_matrix);
    // loop([&](auto& a){a=counter++;},output_matrix);

    auto act = [](const Type &a) { return a; };

    functions::linear::Linear(input_matrix, output_matrix, weight_matrix, bias_matrix, act);

    std::cout << "Input Matrix: " << input_matrix << std::endl;
    std::cout << "Weight Matrix: " << weight_matrix << std::endl;
    std::cout << "Output Matrix: " << output_matrix << std::endl;
    std::cout << "Bias Matrix: " << bias_matrix << std::endl;

    // comparison
    decltype(output_matrix) second_result{}; // initialize with zeros
    auto                    used_second_output = replace<"C", "O">(second_result);
    auto                    used_input         = replace<"C", "I">(input_matrix);

    loop([](Type &a, const Type b, const Type c) { a += b * c; },      // MAC
         permute<"BOI">(broadcast<"I", {inputs}>(used_second_output)), // results
         broadcast<"O", {outputs}>(used_input),                        // inputs
         broadcast<"B", {batch_size}>(weight_matrix));                 // weights
    matrixSum(second_result, bias_matrix);

    std::cout << "Second Result Matrix: " << second_result << std::endl;
    //     std::cout << "{";
    //     loop([](auto a) { std::cout << a << ", "; }, second_result);
    print2DMatrix(second_result);
    //     std::cout << "}" << std::endl;

    Matrix<bool, DimensionOrder("BC"), output_matrix.dimensions[0], output_matrix.dimensions[1]> comparison_matrix;
    loop([](bool &a, const Type &b, const Type &c) { a = (b == c); }, // comparison
         comparison_matrix, second_result, output_matrix);            // results, inputs
    Matrix<bool, DimensionOrder("BC"), 1, 1> total_comparison_matrix{true};
    loop([](bool &a, const bool &b) { a = a && b; },                                                                               // logical AND
         replicate<"BC", {output_matrix.dimensions[0], output_matrix.dimensions[1]}>(total_comparison_matrix), comparison_matrix); // results, inputs

    std::cout << "Comparison Matrix: " << comparison_matrix << std::endl;
    std::cout << "{";
    //     loop([](auto a) { std::cout << a << ", "; }, comparison_matrix);
    print2DMatrix(comparison_matrix);
    std::cout << "}" << std::endl;
    std::cout << "Total Comparison Matrix: " << total_comparison_matrix << std::endl;
    std::cout << "{";
    //     loop([](auto a) { std::cout << a << ", "; }, total_comparison_matrix);
    print2DMatrix(total_comparison_matrix);
    std::cout << "}" << std::endl;
    std::cout << "************************************************************************" << std::endl;
    constexpr Dim_size_t split_size = 8 * 3;
    //     constexpr Dim_size_t actual_split                = ((outputs > split_size && outputs % split_size == 0) ? split_size : outputs);
    constexpr Dim_size_t    actual_split                = 2;
    const auto              weight_split_representation = permute<"IOio">(split<"O", "Oo", outputs / actual_split, actual_split>(split<"I", "Ii", inputs, 1>(weight_matrix)));
    auto                    weight_split                = functions::linear::weightSubBio<2, actual_split>(weight_matrix);
    decltype(output_matrix) output_matrix_split{};
    //     std::cout << "Weight Split Representation: " << weight_split_representation.dimensions << std::endl;
    // weight_split.at(0,0,0,0) = 10; // just to make sure it is not empty

    //     std::cout << "Weight Split Matrix: " << weight_split << std::endl;
    functions::linear::Linear<3>(input_matrix, output_matrix_split, weight_split, bias_matrix, act);

    std::cout << "Output Matrix after split: " << output_matrix_split << std::endl;
    std::cout << "{";
    //     loop([](auto a) { std::cout << a << ", "; }, output_matrix_split);
    print2DMatrix(output_matrix_split);
    std::cout << "}" << std::endl;

    Matrix<bool, DimensionOrder("BC"), output_matrix_split.dimensions[0], output_matrix_split.dimensions[1]> comparison_matrix_2;
    loop(cmp<Type>,                                                // comparison
         comparison_matrix_2, second_result, output_matrix_split); // results, inputs
    Matrix<bool, DimensionOrder("BC"), 1, 1> total_comparison_matrix_2{true};
    loop([](bool &a, const bool &b) { a = a && b; },                                                                                               // logical AND
         replicate<"BC", {output_matrix_split.dimensions[0], output_matrix_split.dimensions[1]}>(total_comparison_matrix_2), comparison_matrix_2); // results, inputs

    std::cout << "Comparison Matrix: " << comparison_matrix_2 << std::endl;
    std::cout << "{";
    //     loop([](auto a) { std::cout << a << ", "; }, comparison_matrix_2);
    print2DMatrix(comparison_matrix_2);
    std::cout << "}" << std::endl;
    std::cout << "Total Comparison Matrix: " << total_comparison_matrix_2 << std::endl;
    std::cout << "{";
    //     loop([](auto a) { std::cout << a << ", "; }, total_comparison_matrix_2);
    print2DMatrix(total_comparison_matrix_2);
    std::cout << "}" << std::endl;
    // auto tmp = makeAlignedMatrixCollection<8>(weight_split_representation, weight_split);

    // std ::cout << std::get<0>(tmp) << std::endl;
    // std ::cout << std::get<1>(tmp) << std::endl;

    std::cout << "************************************************************************" << std::endl;

    auto optimized_weight_matrix = functions::linear::weightSubBio<1, 8 * 3>(weight_matrix);

    auto inversed_optimized_weight_matrix = functions::linear::inverseWeightSubBio(optimized_weight_matrix);

    std::cout << "Optimized Weight Matrix: " << optimized_weight_matrix << std::endl;
    std::cout << "Base Weight Matrix: " << weight_matrix << std::endl;
    std::cout << "{";
    //     loop([](auto a) { std::cout << a << ", "; }, weight_matrix);
    print2DMatrix(weight_matrix);
    std::cout << "}" << std::endl;
    std::cout << "Inversed Optimized Weight Matrix: " << inversed_optimized_weight_matrix << std::endl;
    std::cout << "{";
    //     loop([](auto a) { std::cout << a << ", "; }, inversed_optimized_weight_matrix);
    print2DMatrix(inversed_optimized_weight_matrix);
    std::cout << "}" << std::endl;

    Matrix<bool, DimensionOrder("IO"), weight_matrix.dimensions[0], weight_matrix.dimensions[1]> comparison_matrix_3;
    loop([](bool &a, const Type &b, const Type &c) { a = b == c; },             // comparison
         comparison_matrix_3, weight_matrix, inversed_optimized_weight_matrix); // results, inputs

    std::cout << "Comparison Matrix: " << comparison_matrix_3 << std::endl;
    std::cout << "{";
    //     loop([](auto a) { std::cout << a << ", "; }, comparison_matrix_3);
    print2DMatrix(comparison_matrix_3);
    std::cout << "}" << std::endl;
}

void testLayer() {
    layers::BaseLayer layer;
    using LayerType = decltype(layer);

    using Type                        = float;
    constexpr Dim_size_t inputs       = 100;
    constexpr Dim_size_t outputs      = 200;
    constexpr Dim_size_t batch_size   = 70;

    constexpr Dim_size_t output_split = 8 * 5;    
    constexpr Dim_size_t input_split  = 1;        // AVX2 requires input split to be 1 not anything otherwise suboptimal
    constexpr Dim_size_t batch_split  = 2;


    Matrix<Type, "BC", batch_size, inputs>  input_matrix;
    Matrix<Type, "IO", inputs, outputs>     weight_matrix;
    Matrix<Type, "BC", batch_size, outputs> output_matrix;
//     Matrix<Type, "BC", batch_size, outputs> bias_matrix;
    Matrix<Type, "C", outputs> bias_matrix;

    randomize(input_matrix);
    randomize(weight_matrix);
    randomize(bias_matrix);

    using tmp  = typename functions::linear::InverseWeightSubBioMatrixType<decltype(weight_matrix)>;
    using tmp2 = MaterializedMatrix<PermutedMatrix<"OI", tmp>>;

    std::cout << "Weight Matrix   : " << typeid(weight_matrix).name() << std::endl;
    std::cout << "tmp             : " << typeid(tmp).name() << std::endl;
    std::cout << "tmp2            : " << typeid(tmp2).name() << std::endl;
    std::cout << "tmp2 order      : " << tmp2::order << std::endl;
    std::cout << "tmp2 dimensions : " << tmp2::dimensions << std::endl;
    std::cout << "tmp2 dimensions : " << tmp2::dimensions[0] << std::endl;
    std::cout << "tmp2 dimensions : " << tmp2::dimensions[1] << std::endl;

    auto linearLayer = layers::Linear<Type>(weight_matrix, bias_matrix, [](const auto a) { return a; });

    linearLayer(input_matrix, output_matrix);

    std::cout << "Input Matrix: " << input_matrix << std::endl;
    std::cout << "Weight Matrix: " << weight_matrix << std::endl;
    std::cout << "Output Matrix: " << output_matrix << std::endl;
    //     std::cout << "Output Matrix: " << output_matrix.data << std::endl;
//     print2DMatrix(output_matrix);

    // layer(std::declval<layers::get_ExpectedInputMatrix<LayerType>>(), std::declval<layers::get_ExpectedOutputMatrix<LayerType>&>(), std::declval<Matrix<char, "E",
    // layers::get_memory_buffer_size<LayerType>>&>(),
    //          std::declval<Matrix<char, "E", layers::get_memory_permanent_size<LayerType>>&>());

    auto weight_matrix_split = functions::linear::weightSubBio<input_split,output_split>(weight_matrix);
    auto output_matrix_split = Matrix<Type, "BC", batch_size, outputs>{}; // initialize with zeros
    auto linearlayer_split   = layers::Linear<Type, batch_split>(weight_matrix_split, bias_matrix, [](const auto a) { return a; });

    linearlayer_split(input_matrix, output_matrix_split);
//     std::cout << "Output Matrix after split: " << output_matrix_split << std::endl;
//     print2DMatrix(output_matrix_split);

//     Matrix<bool, DimensionOrder("BC"), output_matrix_split.dimensions[0], output_matrix_split.dimensions[1]> comparison_matrix;
    auto comparison_matrix = matrixCmp(output_matrix_split,output_matrix);
    
    Matrix<bool, DimensionOrder("BC"), 1, 1> total_comparison_matrix{true};
    loop([](bool &a, const bool &b) { a = a && b; },                                                                                           // logical AND
         replicate<"BC", {output_matrix_split.dimensions[0], output_matrix_split.dimensions[1]}>(total_comparison_matrix), comparison_matrix); // results, inputs
    std::cout << "Comparison Matrix: " << comparison_matrix << std::endl;
//     print2DMatrix(comparison_matrix);
    std::cout << "Total Comparison Matrix: " << total_comparison_matrix << std::endl;
    print2DMatrix(total_comparison_matrix);
}

int main() {
    std::cout << "Matrix test file is included successfully." << std::endl;
    testDefaultBeh();
    testLayer();
    return 0;
}