#include <cmath>
#include <iostream>

#include "../include/Matrix.hpp"
#include "../include/MatrixOperations.hpp"
#include "../include/layers/Convolution.hpp"

#include "../include/helpers/print.hpp"

void testIm2Col() {
    std::cout << "************************************************************************" << std::endl;
    std::cout << "Testing Im2Col" << std::endl;
    std::cout << "************************************************************************" << std::endl;

    Matrix<float, "SC", 5, 5> input_matrix{1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 25};
    std::cout << "Input Matrix: " << input_matrix << std::endl;
    // print2DMatrix(input_matrix);

    using im2col = Image2ColMatrix<decltype(input_matrix), "S", "s", {3}, {1}, {1}>;
    std::cout << "Im2Col Data Type" << MaterializedMatrix<im2col>() << std::endl;

    Matrix<float, "BSC", 1, 5, 5> input_matrix2{1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 25};
    using im2col2 = Image2ColMatrix<decltype(input_matrix2), "SC", "sc", {3, 3}, {1, 1}, {1, 1}>;
    std::cout << "Im2Col2 number_of_dimensions              :" << im2col2::number_of_dimensions << std::endl;
    std::cout << "Im2Col2 order                             :" << im2col2::order << std::endl;
    std::cout << "Im2Col2 original_number_of_dimensions     :" << im2col2::original_number_of_dimensions << std::endl;
    std::cout << "Im2Col2 original_order                    :" << im2col2::original_dimensions << std::endl;
    std::cout << "Im2Col2 original_dimensions               :" << im2col2::original_order << std::endl;
    std::cout << "Im2Col2 traverse_sizes                    :" << im2col2::traverse_sizes << std::endl;
    std::cout << "Im2Col2 order_traverse_ez_at_back         :" << im2col2::order_traverse_ez_at_back << std::endl;
    std::cout << "Im2Col2 permutation_order                 :" << im2col2::permutation_order << std::endl;
    std::cout << "Im2Col2 permutation_order_inverse         :" << im2col2::permutation_order_inverse << std::endl;
    std::cout << "Im2Col2 tmp_dimensions                    :" << im2col2::tmp_dimensions << std::endl;
    std::cout << "Im2Col2 dimensions                        :" << im2col2::dimensions << std::endl;
    std::cout << "im2Col2 Type                              :" << MaterializedMatrix<im2col2>() << std::endl;
}

template <IsMatrixType MatrixType>
void dirty_print(const MatrixType &mat) {
    loop([](auto a) { std::cout << a << ", "; }, mat);
}

void testIm2Col_fnc() {
    std::cout << "************************************************************************" << std::endl;
    std::cout << "Testing Im2Col_fnc" << std::endl;
    std::cout << "************************************************************************" << std::endl;
    Matrix<long, "SC", 5, 5> TestMatrix{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};
    std::cout << "Input Matrix: " << TestMatrix << std::endl;
    print2DMatrix(TestMatrix);

    auto im2col_matrix         = im2col<"S", "s">(TestMatrix);
    print2DMatrix(collapse<"sC","C">(im2col_matrix));
    std::cout << "Im2Col Matrix: " << im2col_matrix << std::endl;

    auto im2col_matrix_2 = im2col<"S", "s",{2}>(TestMatrix);
    std::cout << "im2col_matrix_2 Matrix: " << im2col_matrix_2 << std::endl;
    print2DMatrix(collapse<"sC","C">(im2col_matrix_2));
    
    auto im2col_matrix_3 = im2col<"SC", "sc",{2,2}>(TestMatrix);
    std::cout << "im2col_matrix_3 Matrix: " << im2col_matrix_3 << std::endl;
    
    auto im2col_matrix_stride = im2col<"S", "s",{5},{2}>(TestMatrix);
    std::cout << "im2col_matrix_stride Matrix: " << im2col_matrix_stride << std::endl;
    print2DMatrix(collapse<"sC","C">(im2col_matrix_stride));

    auto im2col_matrix_dilation = im2col<"S", "s",{2},{1},{2}>(TestMatrix);
    std::cout << "im2col_matrix_dilation Matrix: " << im2col_matrix_dilation << std::endl;
    print2DMatrix(collapse<"sC","C">(im2col_matrix_dilation));  
}

void testConv(){
    Matrix<int, "SOI",2,3,4> Weights{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
    Matrix<int, "SC",5,4> Input{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
    Matrix<int, "C",3> Bias{1,2, 3};
    auto Conv = layers::Convolution<int,"S",{2},{1},{1}>(Weights,Bias);

    auto Output = Matrix<int, "SC",4,3>();

    auto Buffer = Matrix<int, "E",0>();
    Conv(Input,Output,Buffer,Buffer);

    print2DMatrix(Output);
}

void testConv2d(){
    Matrix<int, "WHOI",3,3,2,3> Weights;
    Matrix<int, "WHC",5,6,3> Input;
    Matrix<int, "C",2> Bias{1,2};

    int counter=1;
    loop([&](auto &w){ w=counter++; }, Weights);
    counter=1;
    loop([&](auto &w){ w=counter++; }, Input);

    auto Conv = layers::Convolution<int,"WH",{3,3},{1,1},{1,1}>(Weights,Bias);

    auto Output = Matrix<int, "WHC",3,4,2>();

    auto Buffer = Matrix<int, "E",0>();
    Conv(Input,Output,Buffer,Buffer);

    // printNDMatrix(Weights);
    printNDMatrix(permute<"CWH">(Output));
}

void testConv2d_stride(){
    Matrix<int, "WHOI",3,3,2,3> Weights;
    Matrix<int, "WHC",5,6,3> Input;
    Matrix<int, "C",2> Bias{1,2};

    int counter=1;
    loop([&](auto &w){ w=counter++; }, Weights);
    counter=1;
    loop([&](auto &w){ w=counter++; }, Input);

    auto Conv = layers::Convolution<int,"WH",{3,3},{2,1},{1,1}>(Weights,Bias);

    auto Output = Matrix<int, "WHC",2,4,2>();

    auto Buffer = Matrix<int, "E",0>();
    Conv(Input,Output,Buffer,Buffer);

    // printNDMatrix(Weights);
    printNDMatrix(permute<"CWH">(Output));
}
void testConv2d_dilation(){
    Matrix<int, "WHOI",3,3,2,3> Weights;
    Matrix<int, "WHC",7,6,3> Input;
    Matrix<int, "C",2> Bias{1,2};

    int counter=1;
    loop([&](auto &w){ w=counter++; }, Weights);
    counter=1;
    loop([&](auto &w){ w=counter++; }, Input);

    auto Conv = layers::Convolution<int,"WH",{3,3},{1,1},{2,1}>(Weights,Bias);

    auto Output = Matrix<int, "WHC",3,4,2>();

    auto Buffer = Matrix<int, "E",0>();
    Conv(Input,Output,Buffer,Buffer);

    // printNDMatrix(Weights);
    printNDMatrix(permute<"CWH">(Output));
}

int main() {
    std::cout << "Matrix test file is included successfully." << std::endl;
    testIm2Col();
    testIm2Col_fnc();
    testConv();
    testConv2d();
    testConv2d_stride();
    testConv2d_dilation();
    return 0;
}