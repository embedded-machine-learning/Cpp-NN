#pragma once

#include <cmath>
#include <concepts>
#include <cstddef>
#include <stddef.h>
#include <type_traits>

#include "BaseLayer.hpp"

#include "../Matrix.hpp"

#include "../functions/activations.hpp"
#include "../functions/linear.hpp"

namespace layers {

template < // Forced Linebreak
        typename DataType      = double,
        Dim_size_t dimension_k = 4,
        Dim_size_t dimension_v = 5,
        Dim_size_t d_model     = 3>
class BasicAttention {
    template <IsMatrixType InputMatrix>
    static constexpr Dim_size_t NDim = InputMatrix::dimensions[InputMatrix::order.indexOf('N')];

    template <IsMatrixType InputMatrix>
    using QMatrixType = Matrix<DataType, "Nk", NDim<InputMatrix>, dimension_k>;
    template <IsMatrixType InputMatrix>
    using KMatrixType = Matrix<DataType, "ak", NDim<InputMatrix>, dimension_k>;
    template <IsMatrixType InputMatrix>
    using VMatrixType = Matrix<DataType, "av", NDim<InputMatrix>, dimension_v>;
    template <IsMatrixType InputMatrix>
    using SoftmaxMatrixType = Matrix<DataType, "Na", NDim<InputMatrix>, NDim<InputMatrix>>;
    template <IsMatrixType InputMatrix>
    using SummationMatrixType = Matrix<DataType, "B", NDim<InputMatrix>>;

    
public:

    using QWeightMatrixType_ = Matrix<DataType, "mk", d_model, dimension_k>;
    using KWeightMatrixType_ = Matrix<DataType, "mk", d_model, dimension_k>;
    using VWeightMatrixType_ = Matrix<DataType, "mv", d_model, dimension_v>;

    QWeightMatrixType_ QWeightMatrix;
    KWeightMatrixType_ KWeightMatrix;
    VWeightMatrixType_ VWeightMatrix;

  public:
    // Permanent Memory which is required for some layers, if used in a time series model
    template <IsMatrixType InputMatrix>
    static constexpr std::size_t memory_permanent = 0;

    using ExampleInputMatrix = Matrix<DataType, "NC", 2, d_model>;
    template <typename InputMatrix>
    using OutputMatrix = OverrideDimensionMatrix<InputMatrix, "C", dimension_v>;

    // the amount of Memory required for temporary storage
    template <IsMatrixType InputMatrix>
    static constexpr std::size_t memory_buffer = sizeof(QMatrixType<InputMatrix>) + sizeof(KMatrixType<InputMatrix>) + sizeof(VMatrixType<InputMatrix>) + sizeof(SoftmaxMatrixType<InputMatrix>) + sizeof(SummationMatrixType<InputMatrix>);
    // Memory Requirements of forward pass
    template <IsMatrixType InputMatrix>
    static constexpr std::size_t memory_minimal = sizeof(InputMatrix) + sizeof(OutputMatrix<InputMatrix>) + memory_buffer<InputMatrix>;
    // Can it reuse the input Memory region?
    static constexpr bool memory_inlined = false;

    // Required Buffer size

    // Constructor
    constexpr BasicAttention(QWeightMatrixType_ &&QWeightMatrix, KWeightMatrixType_ &&KWeightMatrix, VWeightMatrixType_ &&VWeightMatrix)
            : QWeightMatrix(std::forward<QWeightMatrixType_>(QWeightMatrix)), KWeightMatrix(std::forward<KWeightMatrixType_>(KWeightMatrix)),
              VWeightMatrix(std::forward<VWeightMatrixType_>(VWeightMatrix)) {
    }

    // Constructor
    constexpr BasicAttention(QWeightMatrixType_ QWeightMatrix, KWeightMatrixType_ KWeightMatrix, VWeightMatrixType_ VWeightMatrix)
            : QWeightMatrix(QWeightMatrix), KWeightMatrix(KWeightMatrix), VWeightMatrix(VWeightMatrix) {
    }

    // Constructor
    constexpr BasicAttention() = default;

    template <IsMatrixType InputMatrixType, IsMatrixType OutputMatrixType, IsBaseMatrixType BufferMatrixType, IsBaseMatrixType PermanentMatrixType>
    __attribute__((always_inline)) inline void operator()(const InputMatrixType &Input, OutputMatrixType &Out, BufferMatrixType &buffer, [[maybe_unused]]PermanentMatrixType &permanent) const noexcept {
        static_assert(InputMatrixType::order.remove("NC").length() == 0, "Input may only use the dimensions 'NC' (Channel, Sequence), rest not implemented");
        static_assert(OutputMatrixType::order.remove("NC").length() == 0, "Output may only use the dimensions 'NC' (Channel, Sequence), rest not implemented");

        static_assert(InputMatrixType::order.containsAll("NC"), "Input must contain 'NC' (Channel, Sequence), rest not implemented");
        static_assert(OutputMatrixType::order.containsAll("NC"), "Output must contain  'NC' (Channel, Sequence), rest not implemented");

        const auto input_renamed = replace<"C", "m">(Input);
        auto       out_renamed   = replace<"C", "v">(Out); // A Matrix

        constexpr std::size_t MemoryIndexQ       = 0;
        constexpr std::size_t MemoryIndexK       = sizeof(QMatrixType<InputMatrixType>);
        constexpr std::size_t MemoryIndexV       = MemoryIndexK + sizeof(KMatrixType<InputMatrixType>);
        constexpr std::size_t MemoryIndexSoftmax = MemoryIndexV + sizeof(VMatrixType<InputMatrixType>);
        constexpr std::size_t MemoryIndexSummation = MemoryIndexSoftmax + sizeof(SoftmaxMatrixType<InputMatrixType>);


        static_assert(sizeof(BufferMatrixType) >=
                              sizeof(QMatrixType<InputMatrixType>) + sizeof(KMatrixType<InputMatrixType>) + sizeof(VMatrixType<InputMatrixType>) + sizeof(SoftmaxMatrixType<InputMatrixType>),
                      "Buffersize is insufficient");

        auto &Q       = *reinterpret_cast<QMatrixType<InputMatrixType> *>(&buffer.data[MemoryIndexQ]);
        auto &K       = *reinterpret_cast<KMatrixType<InputMatrixType> *>(&buffer.data[MemoryIndexK]);
        auto &V       = *reinterpret_cast<VMatrixType<InputMatrixType> *>(&buffer.data[MemoryIndexV]);
        auto &Softmax = *reinterpret_cast<SoftmaxMatrixType<InputMatrixType> *>(&buffer.data[MemoryIndexSoftmax]);
        auto& Summation = *reinterpret_cast<SummationMatrixType<InputMatrixType>*>(&buffer.data[MemoryIndexSummation]); 
        loop([](DataType &a) { a = static_cast<DataType>(0); }, Summation); // Initialize Summation to zero


        // Q = input @ Weight_Q
        auto input_renamed_renamed = replace<"Nm", "BC">(input_renamed);
        auto Q_renamed             = replace<"Nk", "BC">(Q);
        auto QWeightMatrix_renamed = replace<"mk", "IO">(QWeightMatrix);

        functions::linear::Linear(input_renamed_renamed, Q_renamed, QWeightMatrix_renamed, Matrix<DataType, "E", 1>{0}, PassThrough<DataType>);
        // loop([](Type &a, const Type b, const Type c) { a += b * c; },      // MAC
        //     broadcast<"m", {d_model}>(Q), // results
        //     broadcast<"k", {dimension_k}>(input_renamed),                        // inputs
        //     broadcast<"N", {N}>(QWeightMatrix));                 // weights

        // K = input_renamed @ Weight_K
        auto k_renamed             = replace<"ak", "BC">(K);
        auto KWeightMatrix_renamed = replace<"mk", "IO">(KWeightMatrix);
        functions::linear::Linear(input_renamed_renamed, k_renamed, KWeightMatrix_renamed, Matrix<DataType, "E", 1>{0}, PassThrough<DataType>);
        // loop([](Type &a, const Type b, const Type c) { a += b * c; },      // MAC
        //     broadcast<"m", {d_model}>(replace<"a", "N">(K)), // results
        //     broadcast<"k", {dimension_k}>(input_renamed),                        // inputs
        //     broadcast<"N", {N}>(KWeightMatrix));                 // weights

        // V = input_renamed @ Weight_V
        auto v_renamed             = replace<"av", "BC">(V);
        auto VWeightMatrix_renamed = replace<"mv", "IO">(VWeightMatrix);
        functions::linear::Linear(input_renamed_renamed, v_renamed, VWeightMatrix_renamed, Matrix<DataType, "E", 1>{0}, PassThrough<DataType>);
        // loop([](Type &a, const Type b, const Type c) { a += b * c; },      // MAC
        //     broadcast<"m", {d_model}>(replace<"a", "N">(V)), // results
        //     broadcast<"v", {dimension_v}>(input_renamed),                        // inputs
        //     broadcast<"N", {N}>(VWeightMatrix)
        // );                 // weights

        // Attention + Softmax
        // B = N
        // IC = k
        // OC = a
        const DataType scaler = static_cast<DataType>(1.0 / sqrt(dimension_k));

        auto Summation_br = broadcast<"C", {NDim<InputMatrixType>}>(Summation);

        auto Output_fuzed = fuse(replace<"Na", "BC">(Softmax), Summation_br);
        // Looks like a Matrix<tuple<OutputData,Summation>, "BC",....>

        functions::linear::Linear(
                replace<"Nk", "BC">(Q), Output_fuzed, replace<"ak", "OI">(K), Matrix<DataType, "E", 1>{0},
                [scaler](DataType QKT, DataType Summatuion) -> std::tuple<DataType, DataType> {
                    QKT = QKT * scaler;
                    QKT = exp(QKT);
                    Summatuion += QKT;
                    return {QKT, Summatuion};
                },
                Summation_br);

        loop([](DataType &a, const DataType b) { a /= b; }, // Divide the matrix
             replace<"Na", "BC">(Softmax), Summation_br);

        // Softmax * V
        auto out_renamed_renamed = replace<"Nv", "BC">(out_renamed);
        functions::linear::Linear(replace<"Na", "BC">(Softmax), out_renamed_renamed, replace<"av", "IO">(V), Matrix<DataType, "E", 1>{0}, PassThrough<DataType>);

        // loop([](Type &a, const Type b, const Type c) { a += b * c;},      // MAC
        // broadcast<"a", {N}>(A), // results
        // broadcast<"v", {dimension_v}>(softmax_result),                        // inputs
        // broadcast<"N", {N}>(QKV.V));                 // weights
    }
};

static_assert(IsValidLayer<BasicAttention<>>, "BasicAttention does not meet the requirements of a valid layer");

} // namespace layers