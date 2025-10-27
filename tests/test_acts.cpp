#include <cmath>
#include <iostream>

#include "../include/types/Complex.hpp"
#include "../include/Matrix.hpp"
#include "../include/MatrixOperations.hpp"

#include "../include/helpers/print.hpp"


#include "../include/functions/activations.hpp"


void test_InvSQRT(){
    Matrix<float,"C",100> a;
    float low=0.001,high=1000;
    int index=0;
    loop([&](auto &v){
        v=low + (high - low)/(a.dimensions[0]-1)*index;
        index++;
    },a);
    
    std::cout << "Input:\n";
    printNDMatrix(a);

    Matrix<float,"C",100> inv_sqrt;
    Matrix<float,"C",100> inv_sqrt_fast;
    loop([](const auto &v, auto &out1, auto &out2){
        out1=InvertSQRT<>(v);
        out2=FastInvertSQRT<>(v);
    },a,inv_sqrt,inv_sqrt_fast);

    std::cout << "Output InvertSQRT:\n";
    printNDMatrix(inv_sqrt);
    std::cout << "Output FastInvertSQRT:\n";
    printNDMatrix(inv_sqrt_fast);

    Matrix<float,"C",100> diff;
    loop([](const auto &v1, const auto &v2, auto &out){
        out=std::abs(v1 - v2)/std::abs(v1);
    },inv_sqrt,inv_sqrt_fast,diff);
    std::cout << "Relative Difference:\n";
    printNDMatrix(diff);
}

void test_Norm(){
    Matrix<Complex<float>,"C",100> a;
    float low=0.001,high=5;
    int index=0;
    loop([&](auto &v){
        v=Complex<float>(low + (high - low)/(a.dimensions[0]-1)*index, (low + (high - low)/(a.dimensions[0]-1)*index));
        index++;
    },a);
    std::cout << "Input:\n";
    printNDMatrix(a);

    Matrix<float,"C",100> norm;
    loop([](const auto &v, auto &out){
        out=Norm<>(v);
    },a,norm);
    std::cout << "Output Norm:\n";
    printNDMatrix(norm);
    
    Matrix<float,"C",100> fastnorm;
    loop([](const auto &v, auto &out){
        out= FastNorm<1,float>(v);
    },a,fastnorm);
    std::cout << "Output Fast Norm:\n";
    printNDMatrix(fastnorm);


    Matrix<float,"C",100> diff;
    loop([](const auto &v1, const auto &v2, auto &out){
        out=std::abs(v1 - v2)/std::abs(v1);
    },norm,fastnorm,diff);
    std::cout << "Relative Difference:\n";
    printNDMatrix(diff);
}



int main(){
    test_InvSQRT();
    test_Norm();
    return 0;
}