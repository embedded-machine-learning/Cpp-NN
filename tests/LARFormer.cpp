#include <cmath>
#include <iostream>
#include <tuple>

#include "../include/Matrix.hpp"
#include "../include/MatrixOperations.hpp"
#include "../include/helpers/extended_matrix_ops.hpp"
#include "../include/helpers/print.hpp"

#include "../include/layers/LARFormer.hpp"

constexpr Dim_size_t dimension_k = 4;
constexpr Dim_size_t dimension_v = 5;
constexpr Dim_size_t N           = 7;
constexpr Dim_size_t d_model     = 3;
constexpr Dim_size_t r           = 2;

Matrix<char,"E",100000> buffer;
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




void relu(auto &mat) {

    loop([](Type &a) { a = a >= 0 ? a : 0; }, mat);
}

void rmsNorm(auto &mat) {
    auto power_two = mat;
    auto sum_sq    = initialize_matrix<Matrix<Type, "k", N>>(0.0);
    loop([](Type &a) { a *= a; }, power_two);
    loop([](Type &a, const Type b) { a += b; }, broadcast<"o", {d_model}>(sum_sq), power_two);
    loop([](Type &a) { a = sqrt((a / static_cast<Type>(d_model)) + 1e-8); }, sum_sq);
    loop([](Type &a, const Type b) { a /= b; }, mat, broadcast<"o", {d_model}>(sum_sq));
}

template <typename MatWI, typename MatWO, typename MatWK, typename MatWV>
struct LARFormerWeights {
    MatWI W_I;
    MatWO W_O;
    MatWK W_K;
    MatWV W_V;
};

auto generateLARFormerWeights() {
    auto W_I = randomize_matrix<Matrix<Type, "i1", d_model, 1>>();
    auto W_O = randomize_matrix<Matrix<Type, "co", d_model, d_model>>();
    auto W_K = randomize_matrix<Matrix<Type, "ic", d_model, d_model>>();
    auto W_V = randomize_matrix<Matrix<Type, "iv", d_model, d_model>>();

    return LARFormerWeights{W_I, W_O, W_K, W_V};
}

auto LARFormercalculation() {

    auto input  =  randomize_matrix<Matrix<Type, "ki", N, d_model>>();
    auto weights= generateLARFormerWeights();
    auto cs     = initialize_matrix<Matrix<Type, "k1", N, 1>>(0.0);
    auto cv     = initialize_matrix<Matrix<Type, "1c", 1, d_model>>(0.0);
    auto ck     = initialize_matrix<Matrix<Type, "kc", N, d_model>>(0.0);
    auto xv     = initialize_matrix<Matrix<Type, "kv", N, d_model>>(0.0);
    auto cv_xv  = initialize_matrix<Matrix<Type, "kc", N, d_model>>(0.0);
    auto result = initialize_matrix<Matrix<Type, "ko", N, d_model>>(0.0);

    auto LayerOutput   = initialize_matrix<Matrix<Type, "Nv", N, d_model>>(0.0);
    auto LARFormer = layers::LARFormer<Type, d_model>(weights.W_I, weights.W_K, weights.W_V, weights.W_O);
    auto renamed_output = replace<"v", "C">(LayerOutput);
    LARFormer(replace<"ki", "NC">(input),renamed_output,buffer, permBuffer);


    // cs = ReLU(X @ W_I)
    loop([](Type &a, const Type b, const Type c) { a += b * c; }, // MAC
         broadcast<"i", {d_model}>(cs),                           // results
         broadcast<"1", {1}>(input),                              // inputs
         broadcast<"k", {N}>(weights.W_I));                       // weights

    relu(cs);

    // ck = X @ W_K
    loop([](Type &a, const Type b, const Type c) { a += b * c; }, // MAC
         broadcast<"i", {d_model}>(ck),                           // results
         broadcast<"c", {d_model}>(input),                        // inputs
         broadcast<"k", {N}>(weights.W_K));                       // weights

    // cv = Σ cs * ck
    loop([](Type &a, const Type b, const Type c) { a += b * c; }, // MAC
         broadcast<"k", {N}>(cv),                                 // results
         broadcast<"c", {d_model}>(cs),                           // inputs
         broadcast<"1", {1}>(ck));                                // weights

    // xv = ReLU(X @ W_V)
    loop([](Type &a, const Type b, const Type c) { a += b * c; }, // MAC
         broadcast<"i", {d_model}>(xv),                           // results
         broadcast<"v", {d_model}>(input),                        // inputs
         broadcast<"k", {N}>(weights.W_V));                       // weights
    relu(xv);

    // cv_xv = cv ⊙ xv
    loop([](Type &a, const Type b, const Type c) { a = b * c; }, broadcast<"1", {1}>(cv_xv), broadcast<"k", {N}>(cv), broadcast<"1", {1}>(replace<"v", "c">(xv)));

    // result = cv_xv @ W_O
    loop([](Type &a, const Type b, const Type c) { a += b * c; }, // MAC
         broadcast<"c", {d_model}>(result),                       // results
         broadcast<"o", {d_model}>(cv_xv),                        // inputs
         broadcast<"k", {N}>(weights.W_O));                       // weights

    rmsNorm(result);
    return std::make_tuple(result, LayerOutput);
}

int main() {
    std::cout << "LARFormer Test" << std::endl;
    auto [larFormer,larFormer_Layer]  = LARFormercalculation();
    printNDMatrix_(larFormer);
    printNDMatrix_(larFormer_Layer);

    auto larFormer_cmp = matrixCmp(replace<"ko","Nv">(larFormer),larFormer_Layer);
    bool equal = true;
    loop([&equal](bool val){equal = equal && val;}, larFormer_cmp);
    std::cout << "LarFormer Test Comparison: " << (equal ? "Equal" : "Not Equal") << std::endl;

    return 0;
}