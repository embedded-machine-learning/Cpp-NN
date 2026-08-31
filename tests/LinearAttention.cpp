#include <cmath>
#include <iostream>
#include <tuple>

#include "../include/Matrix.hpp"
#include "../include/MatrixOperations.hpp"
#include "../include/helpers/extended_matrix_ops.hpp"
#include "../include/helpers/print.hpp"

#include "../include/layers/LinearAttention.hpp"

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



void elu(auto &mat) {
    loop([](Type &a) { a = (a >= 0) ? (a + 1) : exp(a); }, mat);
}

auto linearAttentionCalculation() {
     auto input = randomize_matrix<Matrix<Type, "Nm", N, d_model>>(); // input matrix
     auto weights = generateQKVWeights();
     auto QKV   = qkvCalculation(input, weights);

     auto LayerOutput   = initialize_matrix<Matrix<Type, "Nv", N, dimension_v>>(0.0);
     auto LinearAttention = layers::LinearAttention<Type, dimension_k, dimension_v, d_model>(weights.Weight_Q, weights.Weight_K, weights.Weight_V);
     auto renamed_output = replace<"v", "C">(LayerOutput);
     LinearAttention(replace<"m", "C">(input),renamed_output,buffer, permBuffer);


     auto phi_Q = QKV.Q;
     elu(phi_Q);
     auto phi_K = QKV.K;
     elu(phi_K);

     auto phi_KT_V    = initialize_matrix<Matrix<Type, "kv", dimension_k, dimension_v>>(0.0);
     auto numerator   = initialize_matrix<Matrix<Type, "Nv", N, dimension_v>>(0.0);
     auto denominator = initialize_matrix<Matrix<Type, "N", N>>(0.0);

     auto SUM          = initialize_matrix<Matrix<Type, "k", dimension_k>>(0.0);
     auto expanded_SUM = broadcast<"a", {N}>(SUM); // expand by diemension to be reduced

     // Calculate the sum of each row in Q_KT
     loop([](Type &a, const Type b) { a += b; }, // MAC
          expanded_SUM,                          // results
          phi_K                                  // inputs
     );
     loop([](Type &a, const Type b, const Type c) { a += b * c; }, // MAC
          broadcast<"k", {dimension_k}>(denominator),              // results
          (phi_Q),                                                 // inputs
          (replace<"a", "N">(expanded_SUM)));

     // phi_K.T @ V
     auto phi_KT = permute<"ka">(phi_K);
     loop([](Type &a, const Type b, const Type c) { a += b * c; }, // MAC
          broadcast<"a", {N}>(phi_KT_V),                           // results
          broadcast<"v", {dimension_v}>(phi_KT),                   // inputs
          broadcast<"k", {dimension_k}>(QKV.V));

     loop([](Type &a, const Type b, const Type c) { a += b * c; }, // MAC
          broadcast<"k", {dimension_k}>(numerator),                // results
          broadcast<"v", {dimension_v}>(phi_Q),                    // inputs
          broadcast<"N", {N}>(phi_KT_V));

     loop([](Type &a, const Type b) { a /= b; }, // Divide the matrix
          numerator, broadcast<"v", {dimension_v}>(denominator));
     return std::make_tuple(numerator,LayerOutput);
}



int main() {
    std::cout << "Linear Attention Test" << std::endl;
    auto [linear_attention, linear_attention_Layer]  = linearAttentionCalculation();
    printNDMatrix_(linear_attention);
    printNDMatrix_(linear_attention_Layer);

    auto linear_cmp = matrixCmp(linear_attention,linear_attention_Layer);
    bool equal = true;
    loop([&equal](bool val){equal = equal && val;}, linear_cmp);
    std::cout << "Linear Attention Test Comparison: " << (equal ? "Equal" : "Not Equal") << std::endl;

    return 0;
}