#include <cstddef>
#include <iostream>
#include <tuple>
#include <utility>

#include "../include/MAC.hpp"
#include "../include/MatrixOperations.hpp"
#include "../include/functions/linear.hpp"
#include "../include/helpers/extended_matrix_ops.hpp"
#include "../include/helpers/human_readable_types.hpp"
#include "../include/helpers/print.hpp"

template <IsMatrixType FusedMatrixType>
void at(FusedMatrixType &mat, std::size_t i, std::size_t j) {
    std::cout << "At function called with i:    " << i << " j: " << j << std::endl;
    std::cout << "Typename of value_tupe:       " << human_readable_type<typename FusedMatrixType::value_type> << std::endl;
    std::cout << "Typename of data:             " << human_readable_type<decltype(mat.data)> << std::endl;
    std::cout << "Typename of data at 0:        " << human_readable_type<decltype(std::get<0>(mat.data))> << std::endl;
    std::cout << "Typename of data at 0 at i,j: " << human_readable_type<decltype(std::get<0>(mat.data).at(i, j))> << std::endl;
    std::cout << "Typename of data at 1 at i,j: " << human_readable_type<decltype(std::get<1>(mat.data).at(i, j))> << std::endl;
    std::cout << "Value at data 0 at (" << i << "," << j << "): " << std::get<0>(mat.data).at(i, j) << std::endl;
    std::cout << "Typename at i,j: " << human_readable_type<decltype(std::make_tuple(std::get<0>(mat.data).at(i, j), std::get<1>(mat.data).at(i, j)))> << std::endl;

    auto val = mat.at(i, j);
    std::cout << "Value at (" << i << "," << j << "): " << val << std::endl;
    std::cout << "Typename of val: " << human_readable_type<decltype(val)> << std::endl;
}

void test_Fused_matrix() {
    Matrix<int, "BC", 2, 3> A = {.data = {1, 2, 3, 4, 5, 6}};

    Matrix<float, "BC", 2, 3> B = {.data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}};

    FusedMatrixType<std::index_sequence<0, 1>, Matrix<int, "BC", 2, 3>, Matrix<float, "BC", 2, 3>> FusedMatrix(A, B);

    // at(FusedMatrix,1,2);
    std::cout << (FusedMatrix.at(1, 2)) << std::endl; // should print (6,6.0)

    FusedMatrix.at(0, 0) = std::make_tuple(10, 10.0f);
    std::cout << (FusedMatrix.at(0, 0)) << std::endl; // should print (10,10.0)

    printNDMatrix(A);
    printNDMatrix(B);

    auto FusedC = permute<"CB">(FusedMatrix);
    std::cout << "After Permutation to CB:" << std::endl;
    std::cout << human_readable_type<decltype(FusedC.at(0, 0))> << " : " << (FusedC.at(0, 0)) << std::endl; // should print (10,10.0)

    const auto permutedFusedC_const = FusedMatrixType<std::index_sequence<0, 1>, Matrix<int, "BC", 2, 3>, Matrix<float, "BC", 2, 3>>(A, B);
    std::cout << "After Permutation to CB (Const):" << std::endl;
    std::cout << human_readable_type<decltype((permutedFusedC_const.at(0, 0)))> << " : " << (permutedFusedC_const.at(0, 0)) << std::endl; // should print (10,10.0)

    std::cout << "FusedC at return type: " << human_readable_type<decltype(FusedC.at(0, 0))> << std::endl;
    std::cout << "     A at return type: " << human_readable_type<decltype(A.at(0, 0))> << std::endl;

    auto A_permute = permute<"CB">(A);
    auto B_permute = permute<"CB">(B);
    auto FusedC2   = fuse(A_permute, B_permute);
    std::cout << "After Permutation to CB (FusedC2):" << std::endl;
    std::cout << human_readable_type<decltype(FusedC2.at(0, 0))> << " : " << (FusedC2.at(0, 0)) << std::endl; // should print (10,10.0)
}

void test_unfuzed_matrix() {
    Matrix<int, "BC", 2, 3> A = {.data = {1, 2, 3, 4, 5, 6}};

    Matrix<float, "BC", 2, 3> B = {.data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}};

    printNDMatrix(A);
    printNDMatrix(B);

    auto fused_matrix = fuse(A, B);

    printNDMatrix(fused_matrix);
    std::cout << "Fused matrix at (1,2): " << fused_matrix.at(1, 2) << " type: " << human_readable_type<decltype(fused_matrix.at(1, 2))> << std::endl; // should print (6,6.0)

    auto SelectedA = selectFused<0>(fused_matrix);
    auto SelectedB = selectFused<1>(fused_matrix);

    printNDMatrix(SelectedA);
    printNDMatrix(SelectedB);

    std::cout << "Type of fuse(A,B): " << human_readable_type<decltype(fuse(A, B))> << std::endl;

    auto tmp = permute<"CB">(fuse(A, B));

    std::cout << "Type of permute<\"BC\">(fuse(A,B)): " << human_readable_type<decltype(permute<"BC">(fuse(A, B)))> << std::endl;
    std::cout << "Type of permute<\"BC\">(fuse(A,B)): " << human_readable_type<decltype(tmp)> << std::endl;

    auto tmp2            = selectFused<0>(tmp);
    auto tmp3            = FusedMatrix<decltype(A), decltype(B)>(A, B);
    auto SelectedA_combo = selectFused<0>(fuse(A, B));

    std::cout << "tmp2 at (1,2): " << tmp2.at(2, 1) << " type: " << human_readable_type<decltype(tmp2.at(1, 2))> << std::endl;

    std::cout << "SelectedA at (1,2): " << SelectedA.at(1, 2) << std::endl;             // should print 6
    std::cout << "SelectedB at (1,2): " << SelectedB.at(1, 2) << std::endl;             // should print 6.0
    std::cout << "SelectedA_combo at (1,2): " << SelectedA_combo.at(1, 2) << std::endl; // should print 6

    auto materialized_fusion = materialize(fused_matrix);
    printNDMatrix(materialized_fusion);
    std::cout << "Materialized fusion at (1,2): " << materialized_fusion.at(1, 2) << " type: " << human_readable_type<decltype(materialized_fusion.at(1, 2))> << std::endl; // should print (6,6.0)
    auto unfused_materialized = selectFused<0>(materialized_fusion);
    printNDMatrix(unfused_materialized);
    std::cout << "Unfused materialized at (1,2): " << unfused_materialized.at(1, 2) << " type: " << human_readable_type<decltype(unfused_materialized.at(1, 2))> << std::endl; // should print 6
}

template <typename InputType, typename WeightType, typename BiasType>
using MACType1 = DefaultMACOperation<float, float, float>;

template <typename InputType, typename WeightType, typename BiasType>
using MACType2 = DefaultMACOperation<InputType, WeightType, BiasType>;

template <typename InputType, typename WeightType, typename BiasType>
using FusedMac = MACOperationTuple<MACType1, MACType2>::FusedMACOperation<InputType, WeightType, BiasType>;

void test_mac() {

    constexpr std::size_t batch_channels  = 20;
    constexpr std::size_t input_channels  = 234;
    constexpr std::size_t output_channels = 415;

    Matrix<float, "BC", batch_channels, input_channels>  input{};
    Matrix<float, "IO", input_channels, output_channels> weight1{};
    Matrix<float, "IO", input_channels, output_channels> weight2{};

    Matrix<float, "C", output_channels> bias1{};
    Matrix<float, "C", output_channels> bias2{};

    randomize(input);
    randomize(weight1);
    randomize(bias1);
    randomize(weight2);
    randomize(bias2);

    Matrix<float, "BC", batch_channels, output_channels> output1{};
    Matrix<float, "BC", batch_channels, output_channels> output2{};
    Matrix<float, "BC", batch_channels, output_channels> fused_output_split_1{};
    Matrix<float, "BC", batch_channels, output_channels> fused_output_split_2{};

    randomize(output1);              // random init to test actual writeout
    randomize(output2);              // random init to test actual writeout
    randomize(fused_output_split_1); // random init to test actual writeout
    randomize(fused_output_split_2); // random init to test actual writeout

    std::cout << "Running first MAC operation..." << std::endl;
    functions::linear::Linear<1, MACType1>(input, output1, weight1, bias1, [](const auto &x) { return x; });
    std::cout << "Done with first MAC operation." << std::endl;

    std::cout << "Running second MAC operation..." << std::endl;
    functions::linear::Linear<1, MACType2>(input, output2, weight2, bias2, [](const auto &x) { return x; });
    std::cout << "Done with second MAC operation." << std::endl;
    // printNDMatrix(output1);
    // printNDMatrix(Output2);

    auto fused_input  = fuse<0>(input, input);
    auto fused_weight = fuse(weight1, weight2);
    auto fused_bias   = fuse(bias1, bias2);
    auto fused_output = fuse(fused_output_split_1, fused_output_split_2);
    std::cout << "Running fused MAC operation..." << std::endl;
    functions::linear::Linear<1, FusedMac>(fused_input, fused_output, fused_weight, fused_bias, [](const auto &x) { return x; });
    std::cout << "Done with fused MAC operation." << std::endl;
    // printNDMatrix(fused_output_split_1);
    // printNDMatrix(fused_output_split_2);

    auto                     result_cmp = matrixCmp(output1, fused_output_split_1);
    Matrix<long, "BC", 1, 1> result_cmp_long;
    matrixSum(replicate<"BC", {batch_channels, output_channels}>(result_cmp_long), result_cmp);
    std::cout << "Comparison result for first fused MAC operation: " << result_cmp_long.at(0, 0) << " : " << batch_channels * output_channels << std::endl;

    result_cmp               = matrixCmp(output2, fused_output_split_2);
    result_cmp_long.at(0, 0) = 0; // reset
    matrixSum(replicate<"BC", {batch_channels, output_channels}>(result_cmp_long), result_cmp);
    std::cout << "Comparison result for second fused MAC operation: " << result_cmp_long.at(0, 0) << " : " << batch_channels * output_channels << std::endl;

    auto weights_unrolled_1 = functions::linear::weightSubBio<5, 3>(weight1);
    auto weights_unrolled_2 = functions::linear::weightSubBio<5, 3>(weight2);

    std::cout << "Running first unrolled MAC operation..." << std::endl;
    functions::linear::Linear<1, MACType1>(input, output1, weights_unrolled_1, bias1, [](const auto &x) { return x; });
    std::cout << "Done with first unrolled MAC operation." << std::endl;

    std::cout << "Running second unrolled MAC operation..." << std::endl;
    functions::linear::Linear<1, MACType2>(input, output2, weights_unrolled_2, bias2, [](const auto &x) { return x; });
    std::cout << "Done with second unrolled MAC operation." << std::endl;
    // printNDMatrix(Output1);
    // printNDMatrix(Output2);

    result_cmp               = matrixCmp(output1, fused_output_split_1);
    result_cmp_long.at(0, 0) = 0; // reset
    matrixSum(replicate<"BC", {batch_channels, output_channels}>(result_cmp_long), result_cmp);
    std::cout << "Comparison result for first unrolled MAC operation: " << result_cmp_long.at(0, 0) << " : " << batch_channels * output_channels << std::endl;

    result_cmp               = matrixCmp(output2, fused_output_split_2);
    result_cmp_long.at(0, 0) = 0; // reset
    matrixSum(replicate<"BC", {batch_channels, output_channels}>(result_cmp_long), result_cmp);
    std::cout << "Comparison result for second unrolled MAC operation: " << result_cmp_long.at(0, 0) << " : " << batch_channels * output_channels << std::endl;

    auto weights_unrolled_fused = functions::linear::weightSubBio<5, 3>(fuse(weight1, weight2));
    std::cout << "Running fused unrolled MAC operation..." << std::endl;
    functions::linear::Linear<1, FusedMac>(fused_input, fused_output, weights_unrolled_fused, fused_bias, [](const auto &x) { return x; });
    std::cout << "Done with fused unrolled MAC operation." << std::endl;
    // printNDMatrix(fused_output_split_1);
    // printNDMatrix(fused_output_split_2);
    result_cmp               = matrixCmp(output1, fused_output_split_1);
    result_cmp_long.at(0, 0) = 0; // reset
    matrixSum(replicate<"BC", {batch_channels, output_channels}>(result_cmp_long), result_cmp);
    std::cout << "Comparison result for first fused unrolled MAC operation: " << result_cmp_long.at(0, 0) << " : " << batch_channels * output_channels << std::endl;
    result_cmp               = matrixCmp(output2, fused_output_split_2);
    result_cmp_long.at(0, 0) = 0; // reset
    matrixSum(replicate<"BC", {batch_channels, output_channels}>(result_cmp_long), result_cmp);
    std::cout << "Comparison result for second fused unrolled MAC operation: " << result_cmp_long.at(0, 0) << " : " << batch_channels * output_channels << std::endl;

    Matrix<float, "BC", batch_channels, output_channels> output_combined{};
    std::cout << "Running combined MAC operation with custom lambda..." << std::endl;
    functions::linear::Linear<1, FusedMac>(fused_input, output_combined, fused_weight, fused_bias, [](const std::tuple<float, float> &x) { return std::get<0>(x) * std::get<1>(x); });

    std::cout << "Done with combined MAC operation with custom lambda." << std::endl;
    // printNDMatrix(output_combined);

    Matrix<float, "BC", batch_channels, output_channels> output_combined_reference{};
    loop([](auto& res, const auto& out1, const auto& out2){
        res = out1 * out2;
    }, output_combined_reference, output1, output2);
    result_cmp               = matrixCmp(output_combined, output_combined_reference);
    result_cmp_long.at(0, 0) = 0; // reset
    matrixSum(replicate<"BC", {batch_channels, output_channels}>(result_cmp_long), result_cmp);
    std::cout << "Comparison result for combined MAC operation with custom lambda: " << result_cmp_long.at(0, 0) << " : " << batch_channels * output_channels << std::endl;

    std::cout << "Type of fused weight unrolled:        " << human_readable_type<decltype(weights_unrolled_fused)> << std::endl;
    auto test_pack = functions::linear::pack(weights_unrolled_1, weights_unrolled_2);
    std::cout << "Type of packed fused weight unrolled: " << human_readable_type<decltype(test_pack)> << std::endl;

    std::cout << "Size of fused weight unrolled:        " << sizeof(weights_unrolled_fused) << " bytes" << std::endl;
    std::cout << "Size of packed fused weight unrolled: " << sizeof(test_pack) << " bytes" << std::endl;

    std::cout << "size of individual unrolled weight 1: " << sizeof(weights_unrolled_1) << " bytes" << std::endl;
    std::cout << "size of individual unrolled weight 2: " << sizeof(weights_unrolled_2) << " bytes" << std::endl;
    std::cout << "size of individual sum:               " << sizeof(weights_unrolled_1) + sizeof(weights_unrolled_2) << " bytes" << std::endl;
    std::cout << " Should be identical to packed fused weight unrolled" << std::endl;
    
    std::cout << "Size of weight1:                      " << sizeof(weight1) << " bytes" << std::endl;
    std::cout << "Size of weight2:                      " << sizeof(weight2) << " bytes" << std::endl;
    std::cout << "Size of fused weight:                 " << sizeof(fused_weight) << " bytes" << std::endl;
    std::cout << "Should NOT be identical to sum of weight1 and weight2" << std::endl;
}

void test_early_late_fusion_mac(){
    constexpr std::size_t batch_channels  = 20;
    constexpr std::size_t input_channels  = 234;
    constexpr std::size_t output_channels = 415;

    Matrix<float, "BC", batch_channels, input_channels>  input{};
    Matrix<float, "IO", input_channels, output_channels> weight1{};
    Matrix<float, "IO", input_channels, output_channels> weight2{};

    Matrix<float, "C", output_channels> bias1{};
    Matrix<float, "C", output_channels> bias2{};

    randomize(input);
    randomize(weight1);
    randomize(bias1);
    randomize(weight2);
    randomize(bias2);

    Matrix<float, "BC", batch_channels, output_channels> output1{};
    Matrix<float, "BC", batch_channels, output_channels> output2{};

    randomize(output1);              // random init to test actual writeout
    randomize(output2);              // random init to test actual writeout

    auto weights_fused_early = functions::linear::weightSubBioEarlyFusion<5,3>(weight1, weight2);
    auto weights_fused_late  = functions::linear::weightSubBioLateFusion<5,3>(weight1, weight2);

    std::cout << "early fusion weights type: " << human_readable_type<decltype(weights_fused_early)> << std::endl;
    std::cout << "late fusion weights type:  " << human_readable_type<decltype(weights_fused_late)> << std::endl;

    std::cout << "Size of early fused weights: " << sizeof(weights_fused_early) << " bytes" << std::endl;
    std::cout << "Size of late fused weights:  " << sizeof(weights_fused_late) << " bytes" << std::endl;

    auto reversed_weights_early = functions::linear::inverseWeightSubBio(weights_fused_early);
    auto reversed_weights_late  = functions::linear::inverseWeightSubBio(weights_fused_late);

    std::cout << "reversed early fusion weights type: " << human_readable_type<decltype(reversed_weights_early)> << std::endl;
    std::cout << "reversed late fusion weights type:  " << human_readable_type<decltype(reversed_weights_late)> << std::endl;

    std::cout << "Size of reversed early fused weights: " << sizeof(reversed_weights_early) << " bytes" << std::endl;
    std::cout << "Size of reversed late fused weights:  " << sizeof(reversed_weights_late) << " bytes" << std::endl;    
}

int main() {
    test_Fused_matrix();
    test_unfuzed_matrix();
    test_mac();
    test_early_late_fusion_mac();
    return 0;
}