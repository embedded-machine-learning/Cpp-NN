#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "./Matrix.hpp"
#include "./helpers/Complex.hpp"

namespace py = pybind11;

/* ==================== Conversion Functions ==================== */
/* To Matrix Conversion Simple*/

template <typename MatrixType, typename PythonType, DimensionOrder Order, Dim_size_t... dims>
struct array_to_Matrix_helper;

template <typename MatrixType, typename PythonType, DimensionOrder Order, Dim_size_t M1>
struct array_to_Matrix_helper<MatrixType, PythonType, Order, M1> {
    static auto convert(py::array_t<PythonType> A, Matrix<MatrixType, Order, M1> *out) {
        py::buffer_info A_buf      = A.request();
        PythonType     *A_ptr      = static_cast<PythonType *>(A_buf.ptr);
        long            max_length = A_buf.size;
        // Matrix<Type, Order, M1> out;
        for (Dim_size_t i = 0; i < M1; i++) {
            if (i >= max_length) {
                throw std::invalid_argument("Array is too short");
            }
            out->data[i] = static_cast<MatrixType>(A_ptr[i]);
        }
    }
};

template <typename MatrixType, typename PythonType, DimensionOrder Order, Dim_size_t M1, Dim_size_t M2>
struct array_to_Matrix_helper<MatrixType, PythonType, Order, M1, M2> {
    static auto convert(py::array_t<PythonType> A, Matrix<MatrixType, Order, M1, M2> *out) {
        py::buffer_info A_buf = A.request();
        PythonType     *A_ptr = static_cast<PythonType *>(A_buf.ptr);
        // Matrix<Type, Order, M1, M2> out;
        for (Dim_size_t i = 0; i < M1; i++)
            for (Dim_size_t j = 0; j < M2; j++) {
                out->data[i][j] = static_cast<MatrixType>(A_ptr[i * M2 + j]);
                // std::cout << "pos: " << i * M2 + j << "\n";
            }
    }
};

template <typename MatrixType, typename PythonType, DimensionOrder Order, Dim_size_t M1, Dim_size_t M2, Dim_size_t M3>
struct array_to_Matrix_helper<MatrixType, PythonType, Order, M1, M2, M3> {
    static auto convert(py::array_t<PythonType> A, Matrix<MatrixType, Order, M1, M2, M3> *out) {
        py::buffer_info A_buf      = A.request();
        PythonType     *A_ptr      = static_cast<PythonType *>(A_buf.ptr);
        long            max_length = A_buf.size;

        // Matrix<Type, Order, M1, M2, M3> out;
        for (Dim_size_t i = 0; i < M1; i++)
            for (Dim_size_t j = 0; j < M2; j++)
                for (Dim_size_t k = 0; k < M3; k++) {
                    if (i * M2 * M3 + j * M3 + k >= max_length) {
                        throw std::invalid_argument("Array is too short");
                    }
                    out->data[i][j][k] = static_cast<MatrixType>(A_ptr[i * M2 * M3 + j * M3 + k]);
                    // std::cout << "pos: " << i * M2 * M3 + j * M3 + k << "\n";
                }
    }
};

template <typename MatrixType, typename PythonType, DimensionOrder Order, Dim_size_t M1, Dim_size_t M2, Dim_size_t M3, Dim_size_t M4>
struct array_to_Matrix_helper<MatrixType, PythonType, Order, M1, M2, M3, M4> {
    static auto convert(py::array_t<PythonType> A, Matrix<MatrixType, Order, M1, M2, M3, M4> *out) {
        py::buffer_info A_buf = A.request();
        PythonType     *A_ptr = static_cast<PythonType *>(A_buf.ptr);
        // Matrix<Type, Order, M1, M2, M3, M4> out;
        for (Dim_size_t i = 0; i < M1; i++)
            for (Dim_size_t j = 0; j < M2; j++)
                for (Dim_size_t k = 0; k < M3; k++)
                    for (Dim_size_t l = 0; l < M4; l++) {
                        out->data[i][j][k][l] = static_cast<MatrixType>(A_ptr[i * M2 * M3 * M4 + j * M3 * M4 + k * M4 + l]);
                        // std::cout << "pos: " << i * M2 * M3 + j * M3 + k << "\n";
                    }
    }
};

/* To Matrix Conversion Complex*/
template <typename MatrixType, typename PythonType, DimensionOrder Order, Dim_size_t M1>
struct array_to_Matrix_helper<Complex<MatrixType>, PythonType, Order, M1> {
    static auto convert(py::array_t<PythonType> A, Matrix<Complex<MatrixType>, Order, M1> *out) {
        py::buffer_info A_buf = A.request();
        PythonType     *A_ptr = static_cast<PythonType *>(A_buf.ptr);
        // Matrix<Complex<Type>, Order, M1> out;
        for (Dim_size_t i = 0; i < M1; i++)
            out->data[i] = Complex<MatrixType>(A_ptr[i * 2 + 0], A_ptr[i * 2 + 1]);
    }
};

template <typename MatrixType, typename PythonType, DimensionOrder Order, Dim_size_t M1, Dim_size_t M2>
struct array_to_Matrix_helper<Complex<MatrixType>, PythonType, Order, M1, M2> {
    static auto convert(py::array_t<PythonType> A, Matrix<Complex<MatrixType>, Order, M1, M2> *out) {
        py::buffer_info A_buf = A.request();
        PythonType     *A_ptr = static_cast<PythonType *>(A_buf.ptr);
        // Matrix<Complex<Type>, Order, M1, M2> out;
        for (Dim_size_t i = 0; i < M1; i++)
            for (Dim_size_t j = 0; j < M2; j++)
                out->data[i][j] = Complex<MatrixType>(A_ptr[i * M2 * 2 + j * 2 + 0], A_ptr[i * M2 * 2 + j * 2 + 1]);
    }
};

/* From Matrix to Numpy Simple*/
template <typename PythonType, typename MatrixType, DimensionOrder Order, Dim_size_t...>
struct Matrix_to_array_helper;

template <typename PythonType, typename MatrixType, DimensionOrder Order, Dim_size_t M1>
struct Matrix_to_array_helper<PythonType, MatrixType, Order, M1> {
    static py::array_t<PythonType> convert(Matrix<MatrixType, Order, M1> A) {

        auto            out     = py::array_t<PythonType>(M1);
        py::buffer_info out_buf = out.request();
        PythonType     *out_ptr = static_cast<PythonType *>(out_buf.ptr);

        for (Dim_size_t i = 0; i < M1; i++)
            out_ptr[i] = static_cast<PythonType>(A.data[i]);
        return out;
    }
};

template <typename PythonType, typename MatrixType, DimensionOrder Order, Dim_size_t M1, Dim_size_t M2>
struct Matrix_to_array_helper<PythonType, MatrixType, Order, M1, M2> {
    static py::array_t<PythonType> convert(Matrix<MatrixType, Order, M1, M2> A) {
        auto            out     = py::array_t<PythonType>({M1, M2});
        py::buffer_info out_buf = out.request();
        PythonType     *out_ptr = static_cast<PythonType *>(out_buf.ptr);

        for (Dim_size_t i = 0; i < M1; i++)
            for (Dim_size_t j = 0; j < M2; j++)
                out_ptr[i * M2 + j] = static_cast<PythonType>(A.data[i][j]);
        return out;
    }
};

template <typename PythonType, typename MatrixType, DimensionOrder Order, Dim_size_t M1, Dim_size_t M2, Dim_size_t M3>
struct Matrix_to_array_helper<PythonType, MatrixType, Order, M1, M2, M3> {
    static py::array_t<PythonType> convert(Matrix<MatrixType, Order, M1, M2, M3> A) {
        auto            out     = py::array_t<PythonType>({M1, M2, M3});
        py::buffer_info out_buf = out.request();
        PythonType     *out_ptr = static_cast<PythonType *>(out_buf.ptr);

        for (Dim_size_t i = 0; i < M1; i++)
            for (Dim_size_t j = 0; j < M2; j++)
                for (Dim_size_t k = 0; k < M3; k++)
                    out_ptr[i * M2 * M3 + j * M3 + k] = static_cast<PythonType>(A.data[i][j][k]);
        return out;
    }
};

template <typename PythonType, typename MatrixType, DimensionOrder Order, Dim_size_t M1, Dim_size_t M2, Dim_size_t M3, Dim_size_t M4>
struct Matrix_to_array_helper<PythonType, MatrixType, Order, M1, M2, M3, M4> {
    static py::array_t<PythonType> convert(Matrix<MatrixType, Order, M1, M2, M3, M4> A) {
        auto            out     = py::array_t<PythonType>({M1, M2, M3, M4});
        py::buffer_info out_buf = out.request();
        PythonType     *out_ptr = static_cast<PythonType *>(out_buf.ptr);

        for (Dim_size_t i = 0; i < M1; i++)
            for (Dim_size_t j = 0; j < M2; j++)
                for (Dim_size_t k = 0; k < M3; k++)
                    for (Dim_size_t l = 0; l < M4; l++)
                        out_ptr[i * M2 * M3 * M4 + j * M3 * M4 + k * M4 + l] = static_cast<PythonType>(A.data[i][j][k][l]);
        return out;
    }
};

/* From Matrix to Numpy Complex*/

template <typename PythonType, typename MatrixType, DimensionOrder Order, Dim_size_t M1>
struct Matrix_to_array_helper<PythonType, Complex<MatrixType>, Order, M1> {
    static py::array_t<PythonType> convert(Matrix<Complex<MatrixType>, Order, M1> A) {
        auto            out     = py::array_t<PythonType>({M1, (Dim_size_t)2});
        py::buffer_info out_buf = out.request();
        PythonType     *out_ptr = static_cast<PythonType *>(out_buf.ptr);

        for (Dim_size_t i = 0; i < M1; i++) {
            out_ptr[i * 2 + 0] = static_cast<PythonType>(A.data[i].real());
            out_ptr[i * 2 + 1] = static_cast<PythonType>(A.data[i].imag());
        }
        return out;
    }
};

template <typename PythonType, typename MatrixType, DimensionOrder Order, Dim_size_t M1, Dim_size_t M2>
struct Matrix_to_array_helper<PythonType, Complex<MatrixType>, Order, M1, M2> {
    static py::array_t<PythonType> convert(Matrix<Complex<MatrixType>, Order, M1, M2> A) {
        auto            out     = py::array_t<PythonType>({M1, M2, (Dim_size_t)2});
        py::buffer_info out_buf = out.request();
        PythonType     *out_ptr = static_cast<PythonType *>(out_buf.ptr);

        for (Dim_size_t i = 0; i < M1; i++)
            for (Dim_size_t j = 0; j < M2; j++) {
                out_ptr[i * M2 * 2 + j * 2 + 0] = static_cast<PythonType>(A.data[i][j].real());
                out_ptr[i * M2 * 2 + j * 2 + 1] = static_cast<PythonType>(A.data[i][j].imag());
            }
        return out;
    }
};

/* Simplified Function Calls*/
template <typename PythonType, DimensionOrder Order, Dim_size_t... dims>
Matrix<PythonType, Order, dims...> array_to_Matrix(py::array_t<PythonType> A) {
    typename array_to_Matrix_helper<PythonType, PythonType, Order, dims...>::MatrixType out;
    array_to_Matrix_helper<PythonType, PythonType, Order, dims...>::convert(A, &out);
    return out;
}

template <typename MatrixType, typename PythonType, DimensionOrder Order, Dim_size_t... dims>
void array_to_Matrix(py::array_t<PythonType> A, Matrix<MatrixType, Order, dims...> &out) {
    array_to_Matrix_helper<MatrixType, PythonType, Order, dims...>::convert(A, &out);
}

template <typename MatrixType, typename PythonType, DimensionOrder Order, Dim_size_t... dims>
void array_to_Matrix(py::array_t<PythonType> A, Matrix<MatrixType, Order, dims...> *out) {
    array_to_Matrix_helper<MatrixType, PythonType, Order, dims...>::convert(A, out);
}

// template <typename Type, template <typename> class TargetType, DimensionOrder Order, Dim_size_t... dims>
// Matrix<TargetType<Type>, Order, dims...> array_to_Matrix(py::array_t<Type> A) {
//     typename array_to_Matrix_helper<TargetType<Type>, Order, dims...>::MatrixType out;
//     array_to_Matrix_helper<TargetType<Type>, Order, dims...>::convert(A, &out);
//     return out;
// }

// template <typename Type, template <typename> class TargetType, DimensionOrder Order, Dim_size_t... dims>
// void array_to_Matrix(py::array_t<Type> A, typename array_to_Matrix_helper<TargetType<Type>, Order, dims...>::MatrixType &out) {
//     array_to_Matrix_helper<TargetType<Type>, Order, dims...>::convert(A, &out);
// }

// template <typename Type, template <typename> class TargetType, DimensionOrder Order, Dim_size_t... dims>
// void array_to_Matrix(py::array_t<Type> A, typename array_to_Matrix_helper<TargetType<Type>, Order, dims...>::MatrixType *out) {
//     array_to_Matrix_helper<TargetType<Type>, Order, dims...>::convert(A, *out);
// }

template <typename PythonType, typename MatrixType, DimensionOrder Order, Dim_size_t... dims>
py::array_t<PythonType> Matrix_to_array(Matrix<MatrixType, Order, dims...> A) {
    return Matrix_to_array_helper<PythonType, MatrixType, Order, dims...>::convert(A);
}

// template <typename Type, template <typename> class TargetType, DimensionOrder Order, Dim_size_t... dims>
// py::array_t<Type> Matrix_to_array(Matrix<TargetType<Type>, Order, dims...> A) {
//     return Matrix_to_array_helper<TargetType<Type>, Order, dims...>::convert(A);
// }