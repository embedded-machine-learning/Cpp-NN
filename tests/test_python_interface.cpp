#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "../Matrix.hpp"
#include "../functions/linear.hpp"
#include "../pybind_interface.hpp"

#include "../helpers/print.hpp"

auto print(pybind11::array_t<float> input) {
    auto matrix = convertToBaseMatrix<Matrix<float, "BC", 10, 20>>(input);

    print2DMatrix(matrix);

    return convertToNumpyArray(matrix);
}

auto matrixMult(pybind11::array_t<float> input, pybind11::array_t<float> weights, pybind11::array_t<float> bias) {
    auto inputMatrix  = convertToBaseMatrix<Matrix<float, "BC", 10, 20>>(input);
    auto weightMatrix = convertToBaseMatrix<Matrix<float, "IO", 20, 30>>(weights);
    auto biasMatrix   = convertToBaseMatrix<Matrix<float, "C", 30>>(bias);

    Matrix<float, "BC", 10, 30> output;

    functions::linear::Linear(inputMatrix, output, weightMatrix, biasMatrix, [](const auto &x) { return x; });

    return convertToNumpyArray(output);
}

auto matrixMultSplit(pybind11::array_t<float> input, pybind11::array_t<float> weights, pybind11::array_t<float> bias) {
    auto inputMatrix  = convertToBaseMatrix<Matrix<float, "BC", 10, 20>>(input);
    auto weightMatrix = convertToBaseMatrix<Matrix<float, "IO", 20, 30>>(weights);
    auto biasMatrix   = convertToBaseMatrix<Matrix<float, "C", 30>>(bias);

    auto weightMatrixSplit = functions::linear::weightSubBio<3, 4>(weightMatrix);

    Matrix<float, "BC", 10, 30> output;

    functions::linear::Linear<6>(inputMatrix, output, weightMatrixSplit, biasMatrix, [](const auto &x) { return x; });

    return convertToNumpyArray(output);
}

PYBIND11_MODULE(TestPython, m) {
    m.doc() = "Testst The python interface";

    m.def("print", &print, "Runs the print function, requires a (10,10) matrix of floats", pybind11::arg("input"));
    m.def("matrix_mult", &matrixMult, "Runs the matrix multiplication function, requires a (10,20) input matrix, a (20,30) weight matrix and a (30,) bias vector", pybind11::arg("input"),
          pybind11::arg("weights"), pybind11::arg("bias"));
    m.def("matrix_mult_split", &matrixMultSplit,
          "Runs the matrix multiplication function with split weights, requires a (10,20) input matrix, a (20,30) weight matrix and a (30,) bias vector",
          pybind11::arg("input"), pybind11::arg("weights"), pybind11::arg("bias"));
}