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
        typename DataType  = double,
        Dim_size_t d_model = 3>
class LARFormer {
    template <IsMatrixType InputMatrix>
    static constexpr Dim_size_t NDim = InputMatrix::dimensions[InputMatrix::order.indexOf('N')];

    template <IsMatrixType InputMatrix>
    using CSMatrixType = Matrix<DataType, "k1", NDim<InputMatrix>, 1>;
    template <IsMatrixType InputMatrix>
    using CVMatrixType = Matrix<DataType, "1c", 1, d_model>;
    template <IsMatrixType InputMatrix>
    using CKMatrixType = Matrix<DataType, "kc", NDim<InputMatrix>, d_model>;
    template <IsMatrixType InputMatrix>
    using XVMatrixType = Matrix<DataType, "kv", NDim<InputMatrix>, d_model>;
    template <IsMatrixType InputMatrix>
    using CVXVMatrixType = Matrix<DataType, "kc", NDim<InputMatrix>, d_model>;
    template <IsMatrixType InputMatrix>
    using SumSQMatrixType = Matrix<DataType, "N", NDim<InputMatrix>>;

  public:
    using IWeightMatrixType_ = Matrix<DataType, "i1", d_model, 1>;
    using OWeightMatrixType_ = Matrix<DataType, "co", d_model, d_model>;
    using KWeightMatrixType_ = Matrix<DataType, "ic", d_model, d_model>;
    using VWeightMatrixType_ = Matrix<DataType, "iv", d_model, d_model>;

    IWeightMatrixType_ IWeightMatrix;
    OWeightMatrixType_ OWeightMatrix;
    KWeightMatrixType_ KWeightMatrix;
    VWeightMatrixType_ VWeightMatrix;

  public:
    // Permanent Memory which is required for some layers, if used in a time series model
    template <IsMatrixType InputMatrix>
    static constexpr std::size_t memory_permanent = 0;

    using ExampleInputMatrix = Matrix<DataType, "NC", 2, d_model>;
    template <typename InputMatrix>
    using OutputMatrix = OverrideDimensionMatrix<InputMatrix, "C", d_model>;

    // the amount of Memory required for temporary storage
    template <IsMatrixType InputMatrix>
    static constexpr std::size_t memory_buffer = sizeof(CSMatrixType<InputMatrix>) + sizeof(CVMatrixType<InputMatrix>) + sizeof(CKMatrixType<InputMatrix>) + sizeof(XVMatrixType<InputMatrix>) + sizeof(SumSQMatrixType<InputMatrix>) + sizeof(CVXVMatrixType<InputMatrix>)
            ;
    // Memory Requirements of forward pass
    template <IsMatrixType InputMatrix>
    static constexpr std::size_t memory_minimal = sizeof(InputMatrix) + sizeof(OutputMatrix<InputMatrix>) + memory_buffer<InputMatrix>;
    // Can it reuse the input Memory region?
    static constexpr bool memory_inlined = false;

    // Required Buffer size

    // Constructor
    constexpr LARFormer(IWeightMatrixType_ &&IWeightMatrix, KWeightMatrixType_ &&KWeightMatrix, VWeightMatrixType_ &&VWeightMatrix, OWeightMatrixType_ &&OWeightMatrix)
            : IWeightMatrix(std::forward<IWeightMatrixType_>(IWeightMatrix)), KWeightMatrix(std::forward<KWeightMatrixType_>(KWeightMatrix)),
              VWeightMatrix(std::forward<VWeightMatrixType_>(VWeightMatrix)), OWeightMatrix(std::forward<OWeightMatrixType_>(OWeightMatrix)) {
    }

    // Constructor
    constexpr LARFormer(IWeightMatrixType_ IWeightMatrix, KWeightMatrixType_ KWeightMatrix, VWeightMatrixType_ VWeightMatrix, OWeightMatrixType_ OWeightMatrix)
            : IWeightMatrix(IWeightMatrix), KWeightMatrix(KWeightMatrix), VWeightMatrix(VWeightMatrix), OWeightMatrix(OWeightMatrix) {
    }

    // Constructor
    constexpr LARFormer() = default;

    template <IsMatrixType InputMatrixType, IsMatrixType OutputMatrixType, IsBaseMatrixType BufferMatrixType, IsBaseMatrixType PermanentMatrixType>
    __attribute__((always_inline)) inline void operator()(const InputMatrixType &Input, OutputMatrixType &Out, BufferMatrixType &buffer, PermanentMatrixType &permanent) const noexcept {
        static_assert(InputMatrixType::order.remove("NC").length() == 0, "Input may only use the dimensions 'NC' (Channel, Sequence), rest not implemented");
        static_assert(OutputMatrixType::order.remove("NC").length() == 0, "Output may only use the dimensions 'NC' (Channel, Sequence), rest not implemented");

        static_assert(InputMatrixType::order.containsAll("NC"), "Input must contain 'NC' (Channel, Sequence), rest not implemented");
        static_assert(OutputMatrixType::order.containsAll("NC"), "Output must contain  'NC' (Channel, Sequence), rest not implemented");

        const auto input_renamed = replace<"C", "m">(Input);
        auto       out_renamed   = replace<"C", "v">(Out); // A Matrix

        constexpr std::size_t MemoryIndexCS    = 0;
        constexpr std::size_t MemoryIndexCV    = sizeof(CSMatrixType<InputMatrixType>);
        constexpr std::size_t MemoryIndexCK    = MemoryIndexCV + sizeof(CVMatrixType<InputMatrixType>);
        constexpr std::size_t MemoryIndexXV    = MemoryIndexCK + sizeof(CKMatrixType<InputMatrixType>);
        constexpr std::size_t MemoryIndexCVXV  = MemoryIndexXV + sizeof(XVMatrixType<InputMatrixType>);
        constexpr std::size_t MemoryIndexSumSQ = MemoryIndexCVXV + sizeof(CVXVMatrixType<InputMatrixType>);

        static_assert(sizeof(BufferMatrixType) >=
                              sizeof(CSMatrixType<InputMatrixType>) + sizeof(CVMatrixType<InputMatrixType>) + sizeof(CKMatrixType<InputMatrixType>) + sizeof(XVMatrixType<InputMatrixType>)+sizeof(CVXVMatrixType<InputMatrixType>)+sizeof(SumSQMatrixType<InputMatrixType>)
                              ,
                      "Buffersize is insufficient");

        auto &CS     = *reinterpret_cast<CSMatrixType<InputMatrixType> *>(&buffer.data[MemoryIndexCS]);
        auto &CV     = *reinterpret_cast<CVMatrixType<InputMatrixType> *>(&buffer.data[MemoryIndexCV]);
        auto &CK     = *reinterpret_cast<CKMatrixType<InputMatrixType> *>(&buffer.data[MemoryIndexCK]);
        auto &XV     = *reinterpret_cast<XVMatrixType<InputMatrixType> *>(&buffer.data[MemoryIndexXV]);
        auto &cv_xv  = *reinterpret_cast<CVXVMatrixType<InputMatrixType> *>(&buffer.data[MemoryIndexCVXV]);
        auto &sum_sq = *reinterpret_cast<SumSQMatrixType<InputMatrixType> *>(&buffer.data[MemoryIndexSumSQ]);

        loop([](DataType &a) { a = static_cast<DataType>(0); }, CV); // Initialize Summation to zero
        loop([](DataType &a) { a = static_cast<DataType>(0); }, sum_sq); // Initialize Summation to zero

        // CVXVMatrixType<InputMatrixType> cv_xv;
        // Matrix<DataType, "Nv", NDim<InputMatrixType>, d_model> power_two;
        // SumSQMatrixType<InputMatrixType> sum_sq;

        // cs = ReLU(X @ W_I)
        auto input_renamed_renamed = replace<"Nm", "BC">(input_renamed);
        auto CS_renamed            = replace<"k1", "BC">(CS);
        auto IWeightMatrix_renamed = replace<"i1", "IO">(IWeightMatrix);

        functions::linear::Linear(input_renamed_renamed, CS_renamed, IWeightMatrix_renamed, Matrix<DataType, "E", 1>{0}, [](DataType a) {
            if (a >= static_cast<DataType>(0)) {
                return a;
            }
            return DataType(0);
        });

        // ck = X @ W_K
        auto CK_renamed            = replace<"kc", "BC">(CK);
        auto KWeightMatrix_renamed = replace<"ic", "IO">(KWeightMatrix);
        functions::linear::Linear(input_renamed_renamed, CK_renamed, KWeightMatrix_renamed, Matrix<DataType, "E", 1>{0}, PassThrough<DataType>);

        // cv = Σ cs * ck
        loop([](DataType &a, const DataType b, const DataType c) { a += b * c; }, // MAC
             broadcast<"k", {NDim<InputMatrixType>}>(CV),                         // results
             broadcast<"c", {d_model}>(CS),                                       // inputs
             broadcast<"1", {1}>(CK));                                            // weights

        // xv = ReLU(X @ W_V)
        auto XV_renamed            = replace<"kv", "BC">(XV);
        auto VWeightMatrix_renamed = replace<"iv", "IO">(VWeightMatrix);

        functions::linear::Linear(input_renamed_renamed, XV_renamed, VWeightMatrix_renamed, Matrix<DataType, "E", 1>{0}, [](DataType a) {
            if (a >= static_cast<DataType>(0)) {
                return a;
            }
            return DataType(0);
        });

        // cv_xv = cv ⊙ xv
        loop([](DataType &a, const DataType b, const DataType c) { a = b * c; }, broadcast<"1", {1}>(cv_xv), broadcast<"k", {NDim<InputMatrixType>}>(CV), broadcast<"1", {1}>(replace<"v", "c">(XV)));

        auto out_renamed_renamed   = replace<"Nv", "BC">(out_renamed);
        auto OWeightMatrix_renamed = replace<"co", "IO">(OWeightMatrix);
        auto cv_xv_renamed         = replace<"kc", "BC">(cv_xv);
        functions::linear::Linear(cv_xv_renamed, out_renamed_renamed, OWeightMatrix_renamed, Matrix<DataType, "E", 1>{0}, PassThrough<DataType>);

        // matrixAssign(power_two, out_renamed); //weird
        // loop([](DataType &a) { a *= a; }, power_two); //weird
        loop([](DataType &a, const DataType b) { a += b * b; }, broadcast<"v", {d_model}>(sum_sq), out_renamed); // this ois how you do an elementwise square summation into a new matrix
        loop([](DataType &a) { a = sqrtf((a / static_cast<DataType>(d_model)) + static_cast<DataType>(1e-8)); }, sum_sq);
        loop([](DataType &a, const DataType b) { a /= b; }, out_renamed, broadcast<"v", {d_model}>(sum_sq)); // Not fuzed bc sum is broadcasted so it would be more ops if fuzed
    }
};

static_assert(IsValidLayer<LARFormer<>>, "LARFormer does not meet the requirements of a valid layer");

} // namespace layers