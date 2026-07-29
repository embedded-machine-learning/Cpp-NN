#include <cmath>
#include <iostream>
#include <tuple>

#include "../include/Matrix.hpp"
#include "../include/MatrixOperations.hpp"
#include "../include/helpers/extended_matrix_ops.hpp"
#include "../include/helpers/print.hpp"

#include "../include/layers/LowRank.hpp"

constexpr Dim_size_t dimension_k = 32;
constexpr Dim_size_t dimension_v = 48;
constexpr Dim_size_t N           = 700;
constexpr Dim_size_t d_model     = 64;
constexpr Dim_size_t r           = 10;

Matrix<char,"E",5000000> buffer;
Matrix<char,"E",0> permBuffer;

using Type = double;

template <typename MatrixType>
MatrixType initialize_matrix(auto value) {
    MatrixType mat;
    loop([&value](auto &a) { a = value; }, mat);
    return mat;
}

template <typename MatrixType>
MatrixType randomize_matrix() {
    MatrixType mat;
    randomize(mat);
    // loop([](auto &a) { a = (a / 50.0) - 1.0;}, mat); // normalize to range [-1, 1] to prevent overflow in exp and to make the test more stable
    Matrix<Type, MatrixType::order[1], MatrixType::dimensions[1]> sum;

    loop([](auto &a, const auto b) { a += b * b; }, broadcast<DimensionOrder(MatrixType::order[0]), {MatrixType::dimensions[0]}>(sum), mat);

    loop([](auto &a) { a = sqrt(a+1e-3); }, sum); 
    // printNDMatrix(mat);
    // printNDMatrix(sum);
    
    loop([](auto &a, const auto b) { a /= b; }, mat, broadcast<DimensionOrder(MatrixType::order[0]), {MatrixType::dimensions[0]}>(sum));
    
    
    // printNDMatrix(mat);
    return mat;
}

// Basic attention (Q = X@W_Q, K = X@W_K, V = X@W_V)

template <typename MatQ, typename MatK, typename MatV>
struct QKV_Results {
    MatQ Q;
    MatK K;
    MatV V;
};

template <typename MatWQ, typename MatWK, typename MatWV>
struct QKV_Weights {
    MatWQ Weight_Q;
    MatWK Weight_K;
    MatWV Weight_V;
};

// Low-rank attention (Q = (X@A_Q)@VTr_Q, same pattern for K, V)

template <typename MatAQ, typename MatVTrQ, typename MatAK, typename MatVTrK, typename MatAV, typename MatVTrV>
struct LowRankWeights {
    MatAQ   A_Q;
    MatVTrQ VTr_Q;
    MatAK   A_K;
    MatVTrK VTr_K;
    MatAV   A_V;
    MatVTrV VTr_V;
};

auto generateLowRankWeights() {
    auto A_Q   = randomize_matrix<Matrix<Type, "mr", d_model, r>>();
    auto VTr_Q = randomize_matrix<Matrix<Type, "rk", r, dimension_k>>();

    auto A_K   = randomize_matrix<Matrix<Type, "mr", d_model, r>>();
    auto VTr_K = randomize_matrix<Matrix<Type, "rk", r, dimension_k>>();

    auto A_V   = randomize_matrix<Matrix<Type, "mr", d_model, r>>();
    auto VTr_V = randomize_matrix<Matrix<Type, "rv", r, dimension_v>>();

    return LowRankWeights{A_Q, VTr_Q, A_K, VTr_K, A_V, VTr_V};
}

auto lowRank(auto input, auto weights) {
     std::cout << "Test5" << std::endl;
     auto Q = initialize_matrix<Matrix<Type, "Nk", N, dimension_k>>(0.0);
     auto K = initialize_matrix<Matrix<Type, "ak", N, dimension_k>>(0.0);
     auto V = initialize_matrix<Matrix<Type, "av", N, dimension_v>>(0.0);
     
     auto Q_low = initialize_matrix<Matrix<Type, "Nr", N, r>>(0.0);
     auto K_low = initialize_matrix<Matrix<Type, "Nr", N, r>>(0.0);
     auto V_low = initialize_matrix<Matrix<Type, "Nr", N, r>>(0.0);
     std::cout << "Test6" << std::endl;

    // Q_low = X @ AQ
    loop([](Type &a, const Type b, const Type c) { a += b * c; }, // MAC
         broadcast<"m", {d_model}>(Q_low),                        // results
         broadcast<"r", {r}>(input),                              // inputs
         broadcast<"N", {N}>(weights.A_Q));                       // weights

    // Q = Q_low @ VTr_Q
    loop([](Type &a, const Type b, const Type c) { a += b * c; }, // MAC
         broadcast<"r", {r}>(Q),                                  // results
         broadcast<"k", {dimension_k}>(Q_low),                    // inputs
         broadcast<"N", {N}>(weights.VTr_Q));                     // weights

    // K_low = X @ AK
    loop([](Type &a, const Type b, const Type c) { a += b * c; }, // MAC
         broadcast<"m", {d_model}>(K_low),                        // results
         broadcast<"r", {r}>(input),                              // inputs
         broadcast<"N", {N}>(weights.A_K));                       // weights

    // K = K_low @ VTr_K
    loop([](Type &a, const Type b, const Type c) { a += b * c; }, // MAC
         broadcast<"r", {r}>(replace<"a", "N">(K)),               // results
         broadcast<"k", {dimension_k}>(K_low),                    // inputs
         broadcast<"N", {N}>(weights.VTr_K));                     // weights

    // V_low = X @ AV
    loop([](Type &a, const Type b, const Type c) { a += b * c; }, // MAC
         broadcast<"m", {d_model}>(V_low),                        // results
         broadcast<"r", {r}>(input),                              // inputs
         broadcast<"N", {N}>(weights.A_V));                       // weights

    // V = V_low @ VTr_V
    loop([](Type &a, const Type b, const Type c) { a += b * c; }, // MAC
         broadcast<"r", {r}>(replace<"a", "N">(V)),               // results
         broadcast<"v", {dimension_v}>(V_low),                    // inputs
         broadcast<"N", {N}>(weights.VTr_V));                     // weights

    auto results = QKV_Results{Q, K, V};
    return results;
}

auto softmax(auto Q, auto K) {

    auto QKT          = initialize_matrix<Matrix<Type, "Na", N, N>>(0.0);
    auto SUM          = initialize_matrix<Matrix<Type, "N", N>>(0.0);
    auto expanded_SUM = broadcast<"a", {N}>(SUM); // expand by diemension to be reduced

    auto row_max          = initialize_matrix<Matrix<Type, "N", N>>(-INFINITY);
    auto expanded_row_max = broadcast<"a", {N}>(row_max); // expand by diemension to be reduced

    auto K_T = permute<"ka">(K);
    loop([](Type &a, const Type b, const Type c) { a += b * c / sqrt(dimension_k); }, // MAC
         broadcast<"k", {dimension_k}>(QKT),                                          // results
         broadcast<"a", {N}>(Q),                                                      // inputs
         broadcast<"N", {N}>(K_T));

    // // For each row, find the maximum value
    // loop([](Type &a, const Type b) { a = std::max(a, b); },
    //     expanded_row_max,
    //         QKT);
    // // Subtract the row max from each element in the row to prevent overflow in exp
    // loop([](Type &a, const Type b) { a -= b; },
    //     QKT,
    //     expanded_row_max);

    // Apply the exponential function to each element in Q_KT

    loop([](Type &a) { a = exp(a); }, // MAC

         QKT);

    // Calculate the sum of each row in Q_KT
    loop([](Type &a, const Type b) { a += b; }, // MAC
         expanded_SUM,                          // results
         (QKT)                                  // inputs
    );

    // std::cout << "expanded_SUM: " << expanded_SUM << std::endl;
    // std::cout << "Q_KT: " << Q_KT << std::endl;
    // printNDMatrix_(Q_KT);
    // Divide each element in Q_KT by the sum of its row to calculate the softmax
    loop([](Type &a, const Type b) { a /= b; }, // Divide the matrix
         QKT, expanded_SUM);

    auto softmax_result = QKT;
    return softmax_result;
}


auto LowRankCalculation() {
    std::cout << "Test1" << std::endl;

     auto input = randomize_matrix<Matrix<Type, "Nm", N, d_model>>(); // input matrix

     auto result   = initialize_matrix<Matrix<Type, "Nv", N, dimension_v>>(0.0);
     

     auto LayerOutput   = initialize_matrix<Matrix<Type, "Nv", N, dimension_v>>(0.0);
    std::cout << "Test2" << std::endl;
    
    auto weights = generateLowRankWeights();
    std::cout << "Test3" << std::endl;
    auto QKV = lowRank(input, weights);
    std::cout << "Test4" << std::endl;
    
//     auto LowRank = layers::LowRank<Type,dimension_k,dimension_v,d_model, r>(weights.A_Q, weights.VTr_Q,
//          weights.A_K, weights.VTr_K, weights.A_V, weights.VTr_V);
    
    std::cout << "Test4" << std::endl;
     auto renamed_output = replace<"v", "C">(LayerOutput);
     // LowRank(replace<"m", "C">(input),renamed_output,buffer, permBuffer);
    
    // A = softmax(Q_KT) * V
    auto softmax_result = softmax(QKV.Q, QKV.K);
    loop([](Type &a, const Type b, const Type c) { a += b * c; }, // MAC
         broadcast<"a", {N}>(result),                                  // results
         broadcast<"v", {dimension_v}>(softmax_result),           // inputs
         broadcast<"N", {N}>(QKV.V));                             // weights

    return std::make_tuple(result,LayerOutput);
}

int main() {
    std::cout << "Low rank Test" << std::endl;
    auto [lowRank,lowRank_Layer]  = LowRankCalculation();
    printNDMatrix_(lowRank);
    printNDMatrix_(lowRank_Layer);

    auto lowRank_cmp = matrixCmp(lowRank,lowRank_Layer);
    bool equal = true;
    loop([&equal](bool val){equal = equal && val;}, lowRank_cmp);
    std::cout << "Low rank Test Comparison: " << (equal ? "Equal" : "Not Equal") << std::endl;

    return 0;
}