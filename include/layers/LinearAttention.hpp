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
class LinearAttention {
    template <IsMatrixType InputMatrix>
    static constexpr Dim_size_t NDim = InputMatrix::dimensions[InputMatrix::order.indexOf('N')];

    template <IsMatrixType InputMatrix>
    using QMatrixType = Matrix<DataType, "Nk", NDim<InputMatrix>, dimension_k>;
    template <IsMatrixType InputMatrix>
    using KMatrixType = Matrix<DataType, "ak", NDim<InputMatrix>, dimension_k>;
    template <IsMatrixType InputMatrix>
    using VMatrixType = Matrix<DataType, "av", NDim<InputMatrix>, dimension_v>;
    template <IsMatrixType InputMatrix>
    using KTVMatrixType = Matrix<DataType, "kv", dimension_k, dimension_v>;
    template <IsMatrixType InputMatrix>
    using SummationMatrixType = Matrix<DataType, "k", dimension_k>;
    template <IsMatrixType InputMatrix>
    using DenominatorMatrixType = Matrix<DataType, "N", NDim<InputMatrix>>;

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
    static constexpr std::size_t memory_buffer = sizeof(QMatrixType<InputMatrix>) + sizeof(KMatrixType<InputMatrix>) + sizeof(VMatrixType<InputMatrix>) + sizeof(KTVMatrixType<InputMatrix>) +
                                                 sizeof(SummationMatrixType<InputMatrix>) + sizeof(DenominatorMatrixType<InputMatrix>);
    // Memory Requirements of forward pass
    template <IsMatrixType InputMatrix>
    static constexpr std::size_t memory_minimal = sizeof(InputMatrix) + sizeof(OutputMatrix<InputMatrix>) + memory_buffer<InputMatrix>;
    // Can it reuse the input Memory region?
    static constexpr bool memory_inlined = false;

    // Required Buffer size

    // Constructor
    constexpr LinearAttention(QWeightMatrixType_ &&QWeightMatrix, KWeightMatrixType_ &&KWeightMatrix, VWeightMatrixType_ &&VWeightMatrix)
            : QWeightMatrix(std::forward<QWeightMatrixType_>(QWeightMatrix)), KWeightMatrix(std::forward<KWeightMatrixType_>(KWeightMatrix)),
              VWeightMatrix(std::forward<VWeightMatrixType_>(VWeightMatrix)) {
    }

    // Constructor
    constexpr LinearAttention(QWeightMatrixType_ QWeightMatrix, KWeightMatrixType_ KWeightMatrix, VWeightMatrixType_ VWeightMatrix)
            : QWeightMatrix(QWeightMatrix), KWeightMatrix(KWeightMatrix), VWeightMatrix(VWeightMatrix) {
    }

    // Constructor
    constexpr LinearAttention() = default;

    template <IsMatrixType InputMatrixType, IsMatrixType OutputMatrixType, IsBaseMatrixType BufferMatrixType, IsBaseMatrixType PermanentMatrixType>
    __attribute__((always_inline)) inline void operator()(const InputMatrixType &Input, OutputMatrixType &Out, BufferMatrixType &buffer, PermanentMatrixType &permanent) const noexcept {
        static_assert(InputMatrixType::order.remove("NC").length() == 0, "Input may only use the dimensions 'NC' (Channel, Sequence), rest not implemented");
        static_assert(OutputMatrixType::order.remove("NC").length() == 0, "Output may only use the dimensions 'NC' (Channel, Sequence), rest not implemented");

        static_assert(InputMatrixType::order.containsAll("NC"), "Input must contain 'NC' (Channel, Sequence), rest not implemented");
        static_assert(OutputMatrixType::order.containsAll("NC"), "Output must contain  'NC' (Channel, Sequence), rest not implemented");

        const auto input_renamed = replace<"C", "m">(Input);
        auto       out_renamed   = replace<"C", "v">(Out); // A Matrix

        constexpr std::size_t MemoryIndexQ           = 0;
        constexpr std::size_t MemoryIndexK           = sizeof(QMatrixType<InputMatrixType>);
        constexpr std::size_t MemoryIndexV           = MemoryIndexK + sizeof(KMatrixType<InputMatrixType>);
        constexpr std::size_t MemoryIndexKTV         = MemoryIndexV + sizeof(VMatrixType<InputMatrixType>);
        constexpr std::size_t MemoryIndexSummation   = MemoryIndexKTV + sizeof(KTVMatrixType<InputMatrixType>);
        constexpr std::size_t MemoryIndexDenominator = MemoryIndexSummation + sizeof(SummationMatrixType<InputMatrixType>);

        static_assert(sizeof(BufferMatrixType) >= sizeof(QMatrixType<InputMatrixType>) + sizeof(KMatrixType<InputMatrixType>) + sizeof(VMatrixType<InputMatrixType>) +
                                                          sizeof(KTVMatrixType<InputMatrixType>) + sizeof(SummationMatrixType<InputMatrixType>) + sizeof(DenominatorMatrixType<InputMatrixType>),
                      "Buffersize is insufficient");

        auto &Q           = *reinterpret_cast<QMatrixType<InputMatrixType> *>(&buffer.data[MemoryIndexQ]);
        auto &K           = *reinterpret_cast<KMatrixType<InputMatrixType> *>(&buffer.data[MemoryIndexK]);
        auto &V           = *reinterpret_cast<VMatrixType<InputMatrixType> *>(&buffer.data[MemoryIndexV]);
        auto &phi_KT_V    = *reinterpret_cast<KTVMatrixType<InputMatrixType> *>(&buffer.data[MemoryIndexKTV]);
        auto &Summation   = *reinterpret_cast<SummationMatrixType<InputMatrixType> *>(&buffer.data[MemoryIndexSummation]);
        auto &denominator = *reinterpret_cast<DenominatorMatrixType<InputMatrixType> *>(&buffer.data[MemoryIndexDenominator]);

        loop([](DataType &a) { a = DataType(0); }, Summation);
        loop([](DataType &a) { a = DataType(0); }, denominator);

        // Q = input @ Weight_Q, phi_Q
        auto input_renamed_renamed = replace<"Nm", "BC">(input_renamed);
        auto Q_renamed             = replace<"Nk", "BC">(Q);
        auto QWeightMatrix_renamed = replace<"mk", "IO">(QWeightMatrix);

        functions::linear::Linear(input_renamed_renamed, Q_renamed, QWeightMatrix_renamed, Matrix<DataType, "E", 1>{0}, [](DataType a) -> DataType {
            if (a >= static_cast<DataType>(0))
                return a + DataType(1);
            else
                return exp(a);
        });

        // K = input_renamed @ Weight_K, phi_k
        auto K_renamed             = replace<"ak", "BC">(K);
        auto KWeightMatrix_renamed = replace<"mk", "IO">(KWeightMatrix);
        functions::linear::Linear(input_renamed_renamed, K_renamed, KWeightMatrix_renamed, Matrix<DataType, "E", 1>{0}, [](DataType a) -> DataType {
            if (a >= static_cast<DataType>(0))
                return a + DataType(1);
            else
                return exp(a);
        });

        // V = input_renamed @ Weight_V
        auto V_renamed             = replace<"av", "BC">(V);
        auto VWeightMatrix_renamed = replace<"mv", "IO">(VWeightMatrix);
        functions::linear::Linear(input_renamed_renamed, V_renamed, VWeightMatrix_renamed, Matrix<DataType, "E", 1>{0}, PassThrough<DataType>);

        auto phi_KT_V_renamed = replace<"kv", "BC">(phi_KT_V);
        functions::linear::Linear(replace<"ak", "CB">(K), phi_KT_V_renamed, replace<"av", "IO">(V), Matrix<DataType, "E", 1>{0}, PassThrough<DataType>);

        // SUM_K = sum over N of phi_K   (shape [dimension_k])
        auto summation_br = broadcast<"a", {NDim<InputMatrixType>}>(Summation);
        loop([](DataType &a, const DataType b) { a += b; }, summation_br, K);

        // denominator[N] = phi_Q[N,:] . SUM_K
        loop([](DataType &a, const DataType b, const DataType c) { a += b * c; }, // MAC
             broadcast<"k", {dimension_k}>(denominator),                          // results
             (Q),                                                                 // inputs
             (replace<"a", "N">(summation_br)));

        // output = (phi_Q @ phi_KT_V) / denominator   (fused numerator + division)
        auto denominator_br = broadcast<"C", {dimension_v}>(replace<"N", "B">(denominator));
        // auto Output_fused   = fuse(replace<"Nv", "BC">(out_renamed), denominator_br);
        auto output_renamed = replace<"Nv", "BC">(out_renamed);

        // functions::linear::Linear(
        //         replace<"Nk", "BC">(Q), Output_fused, replace<"kv", "IO">(phi_KT_V), Matrix<DataType, "E", 1>{0},
        //         [](DataType numerator, DataType denom) -> std::tuple<DataType, DataType> { return {numerator / denom, denom}; }, denominator_br);

        functions::linear::Linear(
                replace<"Nk", "BC">(Q), output_renamed, replace<"kv", "IO">(phi_KT_V), Matrix<DataType, "E", 1>{0},
                [](DataType numerator, DataType denom) -> DataType { return {numerator / denom}; }, denominator_br);
    }
};

static_assert(IsValidLayer<LinearAttention<>>, "TransformerLayer does not meet the requirements of a valid layer");

} // namespace layers