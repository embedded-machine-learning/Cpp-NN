#pragma once
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "./Matrix.hpp"
#include "./MatrixOperations.hpp"

template <IsBaseMatrixType TargetMatrixType>
TargetMatrixType convertToBaseMatrix(pybind11::array_t<typename TargetMatrixType::value_type> array) {
    pybind11::buffer_info info = array.request();
    if (info.ndim != TargetMatrixType::number_of_dimensions) {
        throw std::runtime_error("Array dimensions do not match the target matrix dimensions");
    }

    for (std::size_t i = 0; i < TargetMatrixType::number_of_dimensions; ++i) {
        if ((unsigned)info.shape[i] != TargetMatrixType::dimensions[i]) {
            std::cout << "TargetMatrixType::dimensions[" << i << "]: " << TargetMatrixType::dimensions[i] << ", info.shape[" << i << "]: " << info.shape[i] << std::endl;
            throw std::runtime_error("Array shape does not match the target matrix dimensions");
        }
    }
    
    TargetMatrixType result;
    auto *data_ptr = static_cast<typename TargetMatrixType::value_type *>(info.ptr);
    std::copy(data_ptr, data_ptr + info.size, result.data.begin());

    return std::move(result);
}

template <IsBaseMatrixType TargetMatrixType, typename NumpyType=typename TargetMatrixType::value_type>
void convertToBaseMatrix(pybind11::array_t<NumpyType> array, TargetMatrixType &result) {
    pybind11::buffer_info info = array.request();
    if (info.ndim != TargetMatrixType::number_of_dimensions) {
        throw std::runtime_error("Array dimensions do not match the target matrix dimensions");
    }

    for (std::size_t i = 0; i < TargetMatrixType::number_of_dimensions; ++i) {
        if ((unsigned)info.shape[i] != TargetMatrixType::dimensions[i]) {
            std::cout << "TargetMatrixType::dimensions[" << i << "]: " << TargetMatrixType::dimensions[i] << ", info.shape[" << i << "]: " << info.shape[i] << std::endl;
            throw std::runtime_error("Array shape does not match the target matrix dimensions");
        }
    }
    
    auto *data_ptr = reinterpret_cast<typename TargetMatrixType::value_type *>(info.ptr);
    std::copy(data_ptr, data_ptr + info.size, result.data.begin());
}

template <IsMatrixType MatrixType, typename NumpyType=typename MatrixType::value_type>
pybind11::array_t<NumpyType> convertToNumpyArray(const MatrixType &matrix) {
    using MaterializedMatrixType = MaterializedMatrix<MatrixType>;

    pybind11::array_t<NumpyType> array(matrix.dimensions);
    pybind11::buffer_info                              info = array.request();

    MaterializedMatrixType *data_ptr = reinterpret_cast<MaterializedMatrixType *>(info.ptr);

    loop([](auto &ret, const auto b) { ret = b; }, *data_ptr, matrix);

    return array;
}