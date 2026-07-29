#include <cmath>
#include <iostream>
#include <tuple>

#include "../include/Matrix.hpp"
#include "../include/MatrixOperations.hpp"
#include "../include/helpers/extended_matrix_ops.hpp"
#include "../include/helpers/print.hpp"

#include "../include/layers/BasicAttention.hpp"

constexpr Dim_size_t dimension_k = 32;
constexpr Dim_size_t dimension_v = 48;
constexpr Dim_size_t N           = 700;
constexpr Dim_size_t d_model     = 64;
constexpr Dim_size_t r           = 2;

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

auto generateQKVWeights() {
    auto Weight_Q = randomize_matrix<Matrix<Type, "mk", d_model, dimension_k>>();
    auto Weight_K = randomize_matrix<Matrix<Type, "mk", d_model, dimension_k>>();
    auto Weight_V = randomize_matrix<Matrix<Type, "mv", d_model, dimension_v>>();

    return QKV_Weights{Weight_Q, Weight_K, Weight_V};
}

auto qkvCalculation(auto input, auto weights) {

    auto Q = initialize_matrix<Matrix<Type, "Nk", N, dimension_k>>(0.0);
    auto K = initialize_matrix<Matrix<Type, "ak", N, dimension_k>>(0.0);
    auto V = initialize_matrix<Matrix<Type, "av", N, dimension_v>>(0.0);

    // Q = input @ Weight_Q
    loop([](Type &a, const Type b, const Type c) { a += b * c; }, // MAC
         broadcast<"m", {d_model}>(Q),                            // results
         broadcast<"k", {dimension_k}>(input),                    // inputs
         broadcast<"N", {N}>(weights.Weight_Q));                  // weights

    // K = input @ Weight_K
    loop([](Type &a, const Type b, const Type c) { a += b * c; }, // MAC
         broadcast<"m", {d_model}>(replace<"a", "N">(K)),         // results
         broadcast<"k", {dimension_k}>(input),                    // inputs
         broadcast<"N", {N}>(weights.Weight_K));                  // weights

    // V = input @ Weight_V
    loop([](Type &a, const Type b, const Type c) { a += b * c; }, // MAC
         broadcast<"m", {d_model}>(replace<"a", "N">(V)),         // results
         broadcast<"v", {dimension_v}>(input),                    // inputs
         broadcast<"N", {N}>(weights.Weight_V));                  // weights

    return QKV_Results{Q, K, V};
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

// Full attention: A = softmax(Q @ K^T) @ V
// use_low_rank selects between the basic and low-rank QKV paths.

auto attentionCalculation() {

     auto input = randomize_matrix<Matrix<Type, "Nm", N, d_model>>(); // input matrix     
     auto attention   = initialize_matrix<Matrix<Type, "Nv", N, dimension_v>>(0.0);
\
     auto LayerOutput   = initialize_matrix<Matrix<Type, "Nv", N, dimension_v>>(0.0);
     auto weights = generateQKVWeights();
     auto QKV = qkvCalculation(input, weights);  
     auto BasicAttention = layers::BasicAttention<Type,dimension_k,dimension_v,d_model>(weights.Weight_Q,weights.Weight_K,weights.Weight_V);
     auto renamed_output = replace<"v", "C">(LayerOutput);
     BasicAttention(replace<"m", "C">(input),renamed_output,buffer, permBuffer);
     // A = softmax(Q_KT) * V
     auto softmax_result = softmax(QKV.Q, QKV.K);
     loop([](Type &a, const Type b, const Type c) { a += b * c; }, // MAC
          broadcast<"a", {N}>(attention),                                  // results
          broadcast<"v", {dimension_v}>(softmax_result),           // inputs
          broadcast<"N", {N}>(QKV.V));                             // weights   
     return std::make_tuple(attention,LayerOutput);
}



int main() {
    std::cout << "Basic Attention Test" << std::endl;
    auto [basic_attention,basic_attention_Layer]  = attentionCalculation();
    printNDMatrix_(basic_attention);
    printNDMatrix_(basic_attention_Layer);

    auto basic_cmp = matrixCmp(basic_attention,basic_attention_Layer);
    bool equal = true;
    loop([&equal](bool val){equal = equal && val;}, basic_cmp);
    std::cout << "Basic Attention Test Comparison: " << (equal ? "Equal" : "Not Equal") << std::endl;

    return 0;
}