#include <fstream>
#include <iostream>
#include <string>
#include <tuple>

#include "include/Matrix.hpp"
#include "include/helpers/Complex.hpp"
#include "include/functions/inference/Linear.hpp"
#include "include/helpers/TestHelpers.hpp"

// #include "weights/weights_60_60.hpp"
#include "weights.hpp"

// #include "weights_unrolled.hpp"

std::string file_location = "./weights_unrolled.hpp";

template <typename Type>
std::ostream &operator<<(std::ostream &os, const Complex<Type> &dt) {
    auto        tmp = TypeName<Type>;
    std::string type(std::begin(tmp), std::end(tmp) - 1);
    os << "Complex<" << type << ">"
       << "{" << dt.real() << ", " << dt.imag() << "}";
    return os;
}

template <Dim_size_t KernelParallelSuggested, Dim_size_t UnrolledSuggested, typename WeightMatrixType>
constexpr auto Linear_unrolling(const WeightMatrixType &Weights) {
    return functions::linear::template WeightUnrollParallel<KernelParallelSuggested, UnrolledSuggested>(Weights);
}

template <typename IndexSequence, size_t unrolling>
struct helper;

template <size_t... I, size_t unrolling>
struct helper<std::index_sequence<I...>, unrolling> {
    template <typename WeightMatrixType>
    static constexpr auto unroll_pointers = std::make_tuple(Linear_unrolling<I + 1, unrolling, WeightMatrixType>...);
};

// const auto    tuple_of_weights =
// std::make_tuple(weights_0,weights_1,weights_2,weights_3,weights_4,weights_5,weights_6,weights_7,weights_8,weights_9,weights_0_int,weights_1_int,weights_2_int,weights_3_int,weights_4_int,weights_5_int,weights_6_int,weights_7_int,weights_8_int,weights_9_int,weights_0_int8,weights_1_int8,weights_2_int8,weights_3_int8,weights_4_int8,weights_5_int8,weights_6_int8,weights_7_int8,weights_8_int8,weights_9_int8);
// const std::string names[]          = {
//         "weights_0",      "weights_1",      "weights_2",      "weights_3",      "weights_4",      "weights_5",      "weights_6",      "weights_7",      "weights_8",      "weights_9",
//         "weights_0_int",  "weights_1_int",  "weights_2_int",  "weights_3_int",  "weights_4_int",  "weights_5_int",  "weights_6_int",  "weights_7_int",  "weights_8_int",  "weights_9_int",
//         "weights_0_int8", "weights_1_int8", "weights_2_int8", "weights_3_int8", "weights_4_int8", "weights_5_int8", "weights_6_int8", "weights_7_int8", "weights_8_int8", "weights_9_int8",
// };

// const auto tuple_of_weights = std::make_tuple(
//     weights_60_60);

// const auto    tuple_of_weights = std::make_tuple(
//     B0,C0,SkipLayer0_weights,
//     B1,C1,SkipLayer1_weights,
//     B2,C2,SkipLayer2_weights
//     ,B3,C3,SkipLayer3_weights,
//     B4,C4,SkipLayer4_weights,
//     B5,C5,SkipLayer5_weights
// );

// const std::string names[]={
//     "B0","C0","SkipLayer0_weights",
//     "B1","C1","SkipLayer1_weights",
//     "B2","C2","SkipLayer2_weights"
//     ,"B3","C3","SkipLayer3_weights",
//     "B4","C4","SkipLayer4_weights",
//     "B5","C5" ,"SkipLayer5_weights"
// };
const auto    tuple_of_weights = std::make_tuple(
    B0,C0,
    B1,C1,
    B2,C2
    ,B3,C3,
    B4,C4,
    B5,C5);

const std::string names[]={
    "B0","C0",
    "B1","C1",
    "B2","C2"
    ,"B3","C3",
    "B4","C4",
    "B5","C5" 
};

// const std::string names[] = {"weights_60_60"};

template <size_t unrolling, typename IndexSequenceWeights, typename IndexSequenceFunctions>
struct helper2;

template <size_t unrolling, size_t... WI, size_t... FI>
struct helper2<unrolling, std::index_sequence<WI...>, std::index_sequence<FI...>> {
    template <typename WeightTuple>
    static constexpr void WeightLoop(const WeightTuple &weights) {
        ((FunctionLoop(names[WI], std::get<WI>(weights))), ...);
    }

    template <typename WeightMatrixType>
    static void FunctionLoop(const std::string Name, const WeightMatrixType &weight) {
        constexpr auto max               = std::get<sizeof...(FI) - 1>(std::make_tuple(FI...));
        constexpr auto unrolled_pointers = helper<std::make_index_sequence<max + 1>, unrolling>::template unroll_pointers<WeightMatrixType>;
        // ((std::cout << Name << " : " << std::get<FI>(unrolled_pointers)(weight) << std::endl), ...);
        ((append_weight_maxtrix(Name, std::get<FI>(unrolled_pointers)(weight), FI, unrolling)), ...);
        // std::cout << max << std::endl;
    }
};

template <typename WeightMatrixType>
void write_weights(const WeightMatrixType &weights, std::ofstream &file) {
    std::string type = "";
    for (auto c : TypeName<WeightMatrixType>) {
        if (c == 0)
            continue;
        if (c == '\b')
            type.pop_back();
        else
            type += c;
    }

    if (weights.dim3 == 0) {
        file << type << "()";
    } else if (weights.dim4 == 0) {
        file << type << "()";
    } else {
        file << "{{";
        for (Dim_size_t dim1 = 0; dim1 < weights.dim1; dim1++) {
            file << "{";
            for (Dim_size_t dim2 = 0; dim2 < weights.dim2; dim2++) {
                file << "{";
                for (Dim_size_t dim3 = 0; dim3 < weights.dim3; dim3++) {
                    file << "{";
                    for (Dim_size_t dim4 = 0; dim4 < weights.dim4; dim4++) {
                        // file << (int)weights.at(dim1, dim2, dim3, dim4);
                        file << weights.at(dim1, dim2, dim3, dim4);
                        if (dim4 != weights.dim4 - 1)
                            file << ",";
                    }
                    file << "}";
                    if (dim3 != weights.dim3 - 1)
                        file << ",";
                }
                file << "}";
                if (dim2 != weights.dim2 - 1)
                    file << ",";
            }
            file << "}";
            if (dim1 != weights.dim1 - 1)
                file << ",";
        }
        file << "}}";
    }
}

template <typename WeightMatrixType>
void append_weight_maxtrix(const std::string &name, const WeightMatrixType &weight, Dim_size_t parallel, Dim_size_t unrolling) {
    std::ofstream file(file_location, std::ios::out | std::ios::app);
    std::string   type = "";
    for (auto c : TypeName<WeightMatrixType>) {
        if (c == 0)
            continue;
        if (c == '\b')
            type.pop_back();
        else
            type += c;
    }
    file << "constexpr " << type << " " << name << "_" << parallel + 1 << "_" << unrolling << " = {";
    // std::cout << name << "_" << parallel + 1 << "_" << unrolling << std::endl;
    const auto &weight_matrix         = std::get<0>(weight);
    const auto &weight_spilled_matrix = std::get<1>(weight);
    
    // const auto &WeightsMainUnrolled    = weight.matrix0;
    // const auto &WeightsUnrolledSpilled = weight.matrix1;
    // const auto &WeightsInputSpilled    = weight.matrix2;
    // const auto &WeightsCombinedSpilled = weight.matrix3;
    
    // if (sizeof(WeightsUnrolledSpilled) == 0 && sizeof(WeightsInputSpilled) == 0 && sizeof(WeightsCombinedSpilled) == 0) {
    //     std::cout << name << "_" << parallel + 1 << "_" << unrolling << std::endl;
    // }
    file << "{{";
    for (Dim_size_t dim1 = 0; dim1 < weight_matrix.dim1; dim1++) {
        file << "{";
        for (Dim_size_t dim2 = 0; dim2 < weight_matrix.dim2; dim2++) {
            file << "{";
            for (Dim_size_t dim3 = 0; dim3 < weight_matrix.dim3; dim3++) {
                file << "{";
                for (Dim_size_t dim4 = 0; dim4 < weight_matrix.dim4; dim4++) {
                    file << weight_matrix.at(dim1, dim2, dim3, dim4);
                    if (dim4 != weight_matrix.dim4 - 1)
                        file << ",";
                }
                file << "}";
                if (dim3 != weight_matrix.dim3 - 1)
                    file << ",";
            }
            file << "}";
            if (dim2 != weight_matrix.dim2 - 1)
                file << ",";
        }
        file << "}";
        if (dim1 != weight_matrix.dim1 - 1)
            file << ",";
    }
    file << "}},";
    if constexpr (remove_cvref_t<decltype(weight_spilled_matrix)>::dim3 == 0) {
        file << "{}};" << std::endl;
        file.close();
        std::cout << name << "_" << parallel + 1 << "_" << unrolling << " has no spilled matrix" << std::endl;
        return;
    } else {
        file << "{{";
        for (Dim_size_t dim1 = 0; dim1 < weight_spilled_matrix.dim1; dim1++) {
            file << "{";
            for (Dim_size_t dim2 = 0; dim2 < weight_spilled_matrix.dim2; dim2++) {
                file << "{";
                for (Dim_size_t dim3 = 0; dim3 < weight_spilled_matrix.dim3; dim3++) {
                    file << "{";
                    for (Dim_size_t dim4 = 0; dim4 < weight_spilled_matrix.dim4; dim4++) {
                        file << weight_spilled_matrix.at(dim1, dim2, dim3, dim4);
                        if (dim4 != weight_spilled_matrix.dim4 - 1)
                            file << ",";
                    }
                    file << "}";
                    if (dim3 != weight_spilled_matrix.dim3 - 1)
                        file << ",";
                }
                file << "}";
                if (dim2 != weight_spilled_matrix.dim2 - 1)
                    file << ",";
            }
            file << "}";
            if (dim1 != weight_spilled_matrix.dim1 - 1)
                file << ",";
        }

        file << "}}};" << std::endl;
    }
    // write_weights(WeightsMainUnrolled, file);
    // file << ",";
    // write_weights(WeightsUnrolledSpilled, file);
    // file << ",";
    // write_weights(WeightsInputSpilled, file);
    // file << ",";
    // write_weights(WeightsCombinedSpilled, file);
    // file << "};" << std::endl;
    file.close();
}

int main() {
    std::ofstream file(file_location, std::ios::out);
    file << "#pragma once\n";
    file << "#include \"./include/Matrix.hpp\"\n";
    file.close();

    helper2<1,std::make_index_sequence<std::tuple_size<decltype(tuple_of_weights)>::value>, std::make_index_sequence<32>>::WeightLoop(tuple_of_weights);
    // helper2<1, std::make_index_sequence<std::tuple_size<decltype(tuple_of_weights)>::value>, std::index_sequence<1>>::WeightLoop(tuple_of_weights);
    // helper2<2,std::make_index_sequence<std::tuple_size<decltype(tuple_of_weights)>::value>, std::make_index_sequence<32>>::WeightLoop(tuple_of_weights);
    // helper2<3,std::make_index_sequence<std::tuple_size<decltype(tuple_of_weights)>::value>, std::make_index_sequence<32>>::WeightLoop(tuple_of_weights);
    // helper2<4,std::make_index_sequence<std::tuple_size<decltype(tuple_of_weights)>::value>, std::make_index_sequence<32>>::WeightLoop(tuple_of_weights);
    // helper2<5,std::make_index_sequence<std::tuple_size<decltype(tuple_of_weights)>::value>, std::make_index_sequence<32>>::WeightLoop(tuple_of_weights);
    // helper2<6,std::make_index_sequence<std::tuple_size<decltype(tuple_of_weights)>::value>, std::make_index_sequence<32>>::WeightLoop(tuple_of_weights);
    // helper2<7,std::make_index_sequence<std::tuple_size<decltype(tuple_of_weights)>::value>, std::make_index_sequence<32>>::WeightLoop(tuple_of_weights);
    // helper2<8,std::make_index_sequence<std::tuple_size<decltype(tuple_of_weights)>::value>, std::make_index_sequence<32>>::WeightLoop(tuple_of_weights);
    // helper2<9,std::make_index_sequence<std::tuple_size<decltype(tuple_of_weights)>::value>, std::make_index_sequence<32>>::WeightLoop(tuple_of_weights);
    // helper2<10,std::make_index_sequence<std::tuple_size<decltype(tuple_of_weights)>::value>, std::make_index_sequence<32>>::WeightLoop(tuple_of_weights);

    return 0;
}