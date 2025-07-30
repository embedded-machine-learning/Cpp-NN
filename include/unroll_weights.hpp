#include <complex>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include "./Matrix.hpp"
#include "./MatrixOperations.hpp"
#include "./helpers/human_readable_types.hpp"
#include "functions/linear.hpp"

#include "./helpers/print.hpp"

std::string file_location = "./weights_unrolled.hpp";

#define TellAboutZeroDimension false

template <typename T>
constexpr auto toString = [](const T &value) {
    std::ostringstream oss;
    oss << value;
    return oss.str();
};

template <>
constexpr auto toString<float> = [](const float &value) {
    std::ostringstream oss;
    oss << value;
    auto str = oss.str();
    if (str.find('.') != std::string::npos) {
        // If the string contains a decimal point, append 'f' to indicate float type
        str += 'f';
    } else {
        // If the string does not contain a decimal point, append '.0f' to indicate float type
        str += ".0f";
    }
    return str;
};

template <>
constexpr auto toString<Complex<float>> = [](const Complex<float> &value) {
    std::ostringstream oss;
    oss << "Complex<float>(" << toString<float>(value.real()) << ", " << toString<float>(value.imag()) << ")";
    return oss.str();
};

template <typename WeightMatrixType>
void writeUnrolledWeightsToFile(std::ostream &os, const WeightMatrixType &weights) {
    if constexpr (WeightMatrixType::k_has_zero_dimension) {
        os << "/* Empty Matrix */ {}";
        return;
    }
    os << "{{ ";
    loop([&](const WeightMatrixType::value_type &weight) { os << toString<typename WeightMatrixType::value_type>(weight) << ", "; }, weights);
    os << "\b\b}}";
}

template <std::size_t N>
std::string sanitiseArray(std::array<char, N> arr) {
    std::string result;
    for (const auto &c : arr) {
        if (c == '\0')
            continue;
        else if (c == '\b')
            result.pop_back(); // Remove the last character if it's a backspace
        else
            result += c; // Append the character to the result
    }
    return result;
}

std::string sanitiseString(std::string arr) {
    std::string result;
    for (const auto &c : arr) {
        if (c == '\0')
            continue;
        else if (c == '\b')
            result.pop_back(); // Remove the last character if it's a backspace
        else
            result += c; // Append the character to the result
    }
    return result;
}

template <typename MatrixTuple, std::size_t... Indexes>
void hiddenWriteAlignedMatrices(std::ostream &os, const MatrixTuple &aligned_matrices, const std::string name, std::index_sequence<Indexes...>) {
    std::ostringstream oss;

    if (std::get<1>(aligned_matrices).k_has_zero_dimension && std::get<2>(aligned_matrices).k_has_zero_dimension && std::get<3>(aligned_matrices).k_has_zero_dimension) {
        std::cout << name << " has a zero dimension" << std::endl;
    }

    oss << "constexpr " << human_readable_type<std::remove_cvref_t<MatrixTuple>> << name << " = {";
    ((writeUnrolledWeightsToFile(oss, std::get<Indexes>(aligned_matrices)), oss << ", "), ...);
    oss << "\b\b};\n";

    os << sanitiseString(oss.str());
}

template <typename WeightType>
void WriteMatrix(std::ostream &os, const WeightType &aligned_matrices, const std::string name) {
    std::ostringstream oss;

    oss << "constexpr " << human_readable_type<WeightType>;
    oss << name << " = ";
    writeUnrolledWeightsToFile(oss, aligned_matrices);
    oss << ";\n";

    os << sanitiseString(oss.str());
}

template <typename MatrixTuple>
void writeAlignedMatrices(std::ostream &os, const MatrixTuple &aligned_matrices, const std::string name) {
    hiddenWriteAlignedMatrices(os, aligned_matrices, name, std::make_index_sequence<std::tuple_size_v<MatrixTuple>>{});
}

template <IsMatrixType WeightType>
void writeAlignedMatrices(std::ostream &os, const WeightType &aligned_matrices, const std::string name) {
    WriteMatrix(os, aligned_matrices, name);
}

template <Dim_size_t SuggestedSubInputChannels, Dim_size_t SuggestedSubOutputChannels, typename WeightMatrixType>
void writeAndUnrollWeight(std::ostream &os, const WeightMatrixType &weight, const std::string name) {
    writeAlignedMatrices(os, functions::linear::weightSubBio<SuggestedSubInputChannels, SuggestedSubOutputChannels>(weight),
                         name + "_" + std::to_string(SuggestedSubInputChannels) + "_" + std::to_string(SuggestedSubOutputChannels));
}

#define WRITE_MATRICES(os, name, SSInput, SSOutput) writeAndUnrollWeight<SSInput, SSOutput>(os, name, #name)

template <Dim_size_t SuggestedSubInputChannels, typename WeightMatrixType, Dim_size_t... SuggestedSubOutputChannels>
void hiddenWriteMultiUnrolledWeights(std::ostream &os, const WeightMatrixType &weight, const std::string &name, std::index_sequence<SuggestedSubOutputChannels...>) {
    ((writeAndUnrollWeight<SuggestedSubInputChannels, SuggestedSubOutputChannels + 1>(os, weight, name)), ...);
}

template <Dim_size_t SuggestedSubInputChannels, Dim_size_t SuggestedSubOutputChannelsUpperBound, typename WeightMatrixType>
void writeMultiUnrolledWeights(std::ostream &os, const WeightMatrixType &weight, const std::string &name) {
    hiddenWriteMultiUnrolledWeights<SuggestedSubInputChannels>(os, weight, name, std::make_index_sequence<SuggestedSubOutputChannelsUpperBound>{});
}

#define WRITE_MULTI_UNROLLED_WEIGHTS(os, name, SSInput, SSOutputUpperBound) writeMultiUnrolledWeights<SSInput, SSOutputUpperBound>(os, name, #name)