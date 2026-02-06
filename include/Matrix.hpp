#pragma once
#include <array>
#include <cstddef>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

#include "helpers/constexpressions.hpp"
#include "helpers/cpp_helpers.hpp"

typedef std::size_t Dim_size_t;

// constexpr std::size_t DimensionOrderStringsLength = 7; // 1 more than the longest string

struct DimensionOrder {
    using type                             = char;
    static constexpr Dim_size_t max_length = 10; // 1 more than the longest string

    std::array<type, max_length> order;

    template <std::size_t N>
    constexpr DimensionOrder(const type (&str)[N]) : order(toArray<max_length>(str)) {
        static_assert(N <= max_length, "String length exceeds max_length");
    }

    constexpr DimensionOrder(const type element) : order({element}) {
    }

    consteval bool unique() const noexcept {
        for (std::size_t i = 0; i < max_length; i++) {
            if (order[i] == '\0') {
                continue;
            }
            for (std::size_t j = i + 1; j < max_length; j++) {
                if (order[j] == order[i]) {
                    return false;
                }
            }
        }
        return true;
    }

    template <std::size_t N>
    consteval std::array<Dim_size_t, N> permutationOrderComputation(const DimensionOrder &From) const noexcept {
        std::array<Dim_size_t, N> permutation{};
        for (std::size_t i = 0; i < N; i++) {
            permutation[i] = N; // default other invalid index
            for (std::size_t j = 0; j < max_length; j++) {
                if (order[i] == From[j]) {
                    permutation[i] = j;
                    break;
                }
            }
        }
        return permutation;
    }

    consteval type operator[](std::size_t index) const noexcept {
        return order[index];
    }

    consteval DimensionOrder range(std::size_t index_start, std::size_t index_stop) const noexcept {
        DimensionOrder result = DimensionOrder("");
        for (std::size_t i = index_start; i < index_stop && i < max_length; i++) {
            result.order[i - index_start] = order[i];
        }
        return result;
    }

    constexpr std::size_t length() const noexcept {
        std::size_t length = 0;
        for (std::size_t i = 0; i < max_length; i++) {
            if (order[i] != '\0') {
                length++;
            } else {
                break; // stop counting at the first null character
            }
        }
        return length;
    }

    consteval bool operator==(const DimensionOrder &other) const noexcept {
        for (std::size_t i = 0; i < max_length; i++) {
            if (order[i] != other.order[i]) {
                return false;
            }
        }
        return true;
    }

    consteval DimensionOrder operator+(const DimensionOrder &other) const {
        DimensionOrder result = *this;
        std::size_t    length = result.length();
        // if (length + other.length() > max_length) {
        //     throw std::range_error("Combined dimension order exceeds maximum length");
        // }
        for (std::size_t i = 0; i < other.length(); i++) {
            if (result.order[i + length] == '\0') {
                result.order[i + length] = other.order[i];
            }
#if __cpp_exceptions == 199711
            else {
                throw std::invalid_argument("Dimension order already contains this character");
            }
#endif
        }
        return result;
    }

    consteval DimensionOrder replace(const DimensionOrder &other, const DimensionOrder &replacement) const {
        DimensionOrder result = *this;
        for (std::size_t a = 0; a < other.length(); a++) {
            for (std::size_t b = 0; b < length(); b++) {
                if (this->order[b] == other.order[a]) {
                    result.order[b] = replacement.order[a];
                }
            }
        }
        return result;
    }

    consteval bool containsAny(const DimensionOrder &other) const noexcept {
        for (std::size_t a = 0; a < other.length(); a++) {
            if (contains(other.order[a])) {
                return true;
            }
        }
        return false;
    }

    consteval bool containsAll(const DimensionOrder &other) const noexcept {
        for (std::size_t a = 0; a < other.length(); a++) {
            if (!contains(other.order[a])) {
                return false;
            }
        }
        return true;
    }

    consteval bool containsOnly(const DimensionOrder &other) const noexcept {
        for (std::size_t a = 0; a < other.length(); a++) {
            if (!contains(other.order[a])) {
                return false;
            }
        }
        if (length() != other.length()) {
            return false; // if there are any characters left that are not in 'other', return false
        }
        return true;
    }

    consteval bool contains(const type &c) const noexcept {
        for (std::size_t i = 0; i < max_length; i++) {
            if (order[i] == c) {
                return true;
            }
        }
        return false;
    }

    consteval DimensionOrder insert(const char &at, const DimensionOrder &other) const {
        DimensionOrder result = *this;
#if __cpp_exceptions == 199711
        std::size_t length = result.length();
        if (length + other.length() - 1 > max_length) {
            throw std::range_error("Combined dimension order exceeds maximum length");
        }
#endif
        std::size_t index = 0;
        for (std::size_t i = 0; i < max_length; i++) {
            result.order[i] = this->order[index];
            index++;
            if (result.order[i] == at) {
                for (std::size_t j = 0; j < other.length(); j++) {
                    result.order[i] = other.order[j];
                    i++;
                }
                i--; // Adjust index to account for the extra character inserted
            }
        }
        return result;
    }

    template <std::size_t N>
    consteval DimensionOrder multiInsert(const std::array<char, N> &at, const std::array<DimensionOrder, N> &other) const {
        DimensionOrder result = *this;
        for (std::size_t n = 0; n < N; n++) {
#if __cpp_exceptions == 199711
            std::size_t length = result.length();
            if (length + other[n].length() - 1 > max_length) {
                throw std::range_error("Combined dimension order exceeds maximum length");
            }
#endif
            std::size_t    index = 0;
            DimensionOrder ref   = result;
            for (std::size_t i = 0; i < max_length; i++) {
                result.order[i] = ref[index];
                index++;
                if (result.order[i] == at[n]) {
                    for (std::size_t j = 0; j < other[n].length(); j++) {
                        result.order[i] = other[n].order[j];
                        i++;
                    }
                    i--; // Adjust index to account for the extra character inserted
                }
            }
        }
        return result;
    }

    consteval DimensionOrder remove(const type &c) const {
        DimensionOrder result = *this;
        std::size_t    index  = 0;
        for (std::size_t i = 0; i < max_length; i++) {
            if (result.order[i] == c) {
                continue; // skip the character to be removed
            }
            if (index < max_length) {
                result.order[index++] = result.order[i];
            }
        }
        // Fill the rest with null characters
        for (std::size_t i = index; i < max_length; i++) {
            result.order[i] = '\0';
        }
        return result;
    }

    consteval DimensionOrder remove(const DimensionOrder &other) const {
        DimensionOrder result = *this;
        for (std::size_t i = 0; i < max_length; i++) {
            if (other.order[i] == '\0') {
                break; // stop at the first null character
            }
            result = result.remove(other.order[i]);
        }
        return result;
    }

    consteval std::size_t indexOf(const type &c) const noexcept {
        for (std::size_t i = 0; i < max_length; i++) {
            if (order[i] == c) {
                return i;
            }
        }
        return max_length; // return max_length if not found
    }

    consteval DimensionOrder toLowerCase() const noexcept {
        DimensionOrder result = *this;
        for (std::size_t i = 0; i < max_length; i++) {
            if (result.order[i] >= 'A' && result.order[i] <= 'Z') {
                result.order[i] += ('a' - 'A');
            }
        }
        return result;
    }

    consteval bool operator==(const DimensionOrder &other) noexcept {
        for (std::size_t i = 0; i < max_length; i++) {
            if (order[i] != other.order[i]) {
                return false;
            }
        }
        return true;
    }
};

template <Dim_size_t... Dims>
constexpr std::array<Dim_size_t, sizeof...(Dims)> calculateOffsets(std::array<Dim_size_t, sizeof...(Dims)> dimensions = {Dims...}) {
    Dim_size_t base = 1;
    for (std::size_t i = 0; i < sizeof...(Dims); i++) {
        if (dimensions[i] == 0) {
            base = 0; // if any dimension is 0, the offset is 0
        }
    }
    for (std::size_t a = 0; a < sizeof...(Dims); a++) {
        dimensions[a] = base;
        for (std::size_t b = a + 1; b < sizeof...(Dims); b++) {
            dimensions[a] *= dimensions[b];
        }
    }
    return dimensions;
}

#if false
#include <iostream>

template <size_t N>
constexpr void checkBoundsHelper(Dim_size_t Dim, Dim_size_t accessed_index) {
    // std::cout << "Dim: " << Dim << " accessed_index: " << accessed_index << std::endl;
    // if (accessed_index < 0) {
    //     std::cout << "Index out of bounds" << std::endl;
    //     std::cout << "Dim: " << Dim << " accessed_index: " << accessed_index << std::endl;
    //     std::cout << "Number of dimensions: " << N << std::endl;
    //     throw std::out_of_range("Index out of bounds");
    // }
    if (Dim == 0) {
        std::cout << "Dimension is zero" << std::endl;
        std::cout << "Dim: " << Dim << " accessed_index: " << accessed_index << std::endl;
        std::cout << "Number of dimensions: " << N << std::endl;
        throw std::out_of_range("Dimension is zero");
    }
    if (accessed_index >= Dim) {
        std::cout << "Index out of bounds" << std::endl;
        std::cout << "Dim: " << Dim << " accessed_index: " << accessed_index << std::endl;
        std::cout << "Number of dimensions: " << N << std::endl;
        throw std::out_of_range("Index out of bounds");
    }
}

template <Dim_size_t... Dims>
struct CheckBoundsT {
    template <size_t... Indexes>
    __attribute__((always_inline)) constexpr static inline void check(std::array<Dim_size_t, sizeof...(Dims)> dims, std::index_sequence<Indexes...>) {
        ((checkBoundsHelper<sizeof...(Dims)>(std::array<Dim_size_t, sizeof...(Dims)>{Dims...}[Indexes], dims[Indexes])), ...);
    }
};

template <Dim_size_t... Dims>
__attribute__((always_inline)) constexpr inline void checkBounds(std::array<Dim_size_t, sizeof...(Dims)> dims) {
    CheckBoundsT<Dims...>::check(dims, std::make_index_sequence<sizeof...(Dims)>());
}
#else
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

template <Dim_size_t... Dims>
__attribute__((always_inline)) constexpr inline void checkBounds(std::array<Dim_size_t, sizeof...(Dims)> dims) {
}

#pragma clang diagnostic pop
#pragma GCC diagnostic pop
#endif

template <std::size_t Length, std::array<Dim_size_t, Length> Dims, std::array<Dim_size_t, Length> PermutationIndices, std::size_t... VariadicIndices>
static consteval std::array<Dim_size_t, Length> calculateDimensions(std::index_sequence<VariadicIndices...>) {
    // Calculate the dimensions based on the permutation indices
    return {Dims[PermutationIndices[VariadicIndices]]...};
}

/****************************************************************************************************************************************
 *                                                      MATRIX TEMPLATE CLASS
 *****************************************************************************************************************************************/
// Constraints
template <typename MatrixType>
concept IsMatrixType = requires {
    &std::remove_cvref_t<MatrixType>::order;      // An Order must exist
    &std::remove_cvref_t<MatrixType>::dimensions; // Dimensions must exist
    &std::remove_cvref_t<MatrixType>::number_of_dimensions;
    &std::remove_cvref_t<MatrixType>::k_has_zero_dimension;
    typename std::remove_cvref_t<MatrixType>::value_type; // Value type must exist
    // &MatrixType::at;                                           // At function must exist
};

template <typename MatrixType>
concept IsFuzedMatrixType = requires {
    requires IsMatrixType<MatrixType>;
    &std::tuple_size_v<typename std::remove_cvref_t<MatrixType>::value_type>;
};

template <typename IndexType>
concept IsIndexType = std::is_convertible_v<IndexType, Dim_size_t>;

// Matrix declarations
template <typename Type,
          DimensionOrder LocalOrder                                     = DimensionOrder(""),
          template <typename, std::size_t> class ContainerType          = std::array,
          std::size_t                                NumberOfDimensions = 0, //
          std::array<Dim_size_t, NumberOfDimensions> Dims               = std::array<Dim_size_t, NumberOfDimensions>{},
          typename VariadicIndcices                                     = std::index_sequence<>>
    requires(LocalOrder.unique() && LocalOrder.length() > 0)
struct MatrixType;

template <IsMatrixType   BaseMatrixType, // The Base Matrix
          DimensionOrder LocalOrder = DimensionOrder(""),
          typename VariadicIndcices = std::index_sequence<>>
    requires(LocalOrder.unique() && LocalOrder.length() > 0 && std::remove_cvref_t<BaseMatrixType>::order.containsOnly(LocalOrder))
struct PermutedMatrixType;

template <std::size_t ConcatenatedMatrixDimension = 0, //
          typename VariadicIndcices               = std::index_sequence<>,
          IsMatrixType... MatrixTypes>
    requires(((std::tuple_element_t<0, std::tuple<std::remove_cvref_t<MatrixTypes>...>>::order == std::remove_cvref_t<MatrixTypes>::order) && ...) &&
             (all_same<typename std::remove_cvref_t<MatrixTypes>::value_type...>))
struct ConcatenatedMatrixType;

template <IsMatrixType                                 BaseMatrixType,
          DimensionOrder                               SlicedOrder,
          std::array<Dim_size_t, SlicedOrder.length()> Slices,
          typename VariadicIndices,
          typename VariadicSliceIndices,
          typename VariadicReducedIndices>
    requires(SlicedOrder.unique() && SlicedOrder.length() > 0 && std::remove_cvref_t<BaseMatrixType>::order.containsAll(SlicedOrder) &&
             SlicedOrder.remove(std::remove_cvref_t<BaseMatrixType>::order).length() == 0)
struct SlicedMatrixType;

template <IsMatrixType                                BaseMatrixType,
          DimensionOrder                              AddedOrder,
          std::array<Dim_size_t, AddedOrder.length()> Lengths,
          typename VariadicIndices,
          typename VariadicNewIndices,
          typename VariadicBaseIndices>
    requires(std::remove_cvref_t<BaseMatrixType>::order.length() + AddedOrder.length() <= DimensionOrder::max_length && AddedOrder.unique() && AddedOrder.length() > 0 &&
             (std::remove_cvref_t<BaseMatrixType>::order + AddedOrder).unique())
struct BroadcastedMatrixType;

template <IsMatrixType                                   BaseMatrixType,
          DimensionOrder                                 ReplicatOrder,
          std::array<Dim_size_t, ReplicatOrder.length()> Lengths,
          typename VariadicIndices,
          typename VariadicNewIndices,
          typename VariadicBaseIndices>
    requires(std::remove_cvref_t<BaseMatrixType>::order.containsAll(ReplicatOrder) && ReplicatOrder.unique() && ReplicatOrder.length() > 0)
struct ReplicatedMatrixType;

template <IsMatrixType BaseMatrixType, DimensionOrder ReplaceFrom, DimensionOrder ReplaceTo, typename VariadicIndices>
    requires(ReplaceFrom.unique() && ReplaceTo.unique() && ReplaceFrom.length() == ReplaceTo.length() && (std::remove_cvref_t<BaseMatrixType>::order.replace(ReplaceFrom, ReplaceTo)).unique() &&
             std::remove_cvref_t<BaseMatrixType>::order.containsAll(ReplaceFrom) && !std::remove_cvref_t<BaseMatrixType>::order.remove(ReplaceFrom).containsAny(ReplaceTo))
struct ReplacedMatrixType;

template <IsMatrixType BaseMatrixType, typename VariadicIndices>
struct NegativeMatrixType;

template <IsMatrixType BaseMatrixType, DimensionOrder Old, DimensionOrder New, std::array<Dim_size_t, New.length()> NewDimensions, typename... VarVariadicIndices>
    requires(Old.length() == 1 && std::remove_cvref_t<BaseMatrixType>::order.contains(Old[0]) && !std::remove_cvref_t<BaseMatrixType>::order.remove(Old[0]).containsAny(New) && New.unique())
struct SplitMatrixType;

template <IsMatrixType BaseMatrixType, DimensionOrder Old, DimensionOrder New, typename VariadicIndices, typename VariadicIndicesM1, typename VariadicOriginalIndices, typename VariadicOldIndices>
    requires(New.length() == 1 && std::remove_cvref_t<BaseMatrixType>::order.containsAll(Old) && !std::remove_cvref_t<BaseMatrixType>::order.remove(Old).containsAny(New) && New.unique())
struct CollapsedMatrixType;

template <typename Is, IsMatrixType... BaseMatrixType>
    requires(((std::tuple_element_t<0, std::tuple<std::remove_cvref_t<BaseMatrixType>...>>::number_of_dimensions == std::remove_cvref_t<BaseMatrixType>::number_of_dimensions) &&
              ...) // All matrices must have the same number of dimensions
             && ((std::tuple_element_t<0, std::tuple<std::remove_cvref_t<BaseMatrixType>...>>::order == std::remove_cvref_t<BaseMatrixType>::order) &&
                 ...) // All matrices must have the same dimension order
             && ((std::tuple_element_t<0, std::tuple<std::remove_cvref_t<BaseMatrixType>...>>::dimensions == std::remove_cvref_t<BaseMatrixType>::dimensions) &&
                 ...) // All matrices must have the same dimensions
             )
struct FusedMatrixType;



// Matrix definitions
template <typename DataType,
          DimensionOrder LocalOrder,
          template <typename, std::size_t>
          class ContainerType,
          std::size_t                                NumberOfDimensions,
          std::array<Dim_size_t, NumberOfDimensions> Dims,
          Dim_size_t... VariadicIndices>
struct MatrixType<DataType, LocalOrder, ContainerType, NumberOfDimensions, Dims, std::index_sequence<VariadicIndices...>> {
    using value_type                                                                   = DataType;
    static constexpr std::size_t                                  number_of_dimensions = NumberOfDimensions;
    static constexpr DimensionOrder                               order                = LocalOrder;
    static constexpr std::array<Dim_size_t, number_of_dimensions> dimensions           = Dims;
    static constexpr std::array<Dim_size_t, number_of_dimensions> offsets              = calculateOffsets<dimensions[VariadicIndices]...>();
    static constexpr bool                                         k_has_zero_dimension = ((dimensions[VariadicIndices] == 0) || ...);

    static_assert(sizeof...(VariadicIndices) == number_of_dimensions, "How The FUCK did you manage to get this error? Variadic indices must match number of dimensions");
    static_assert(order.length() == number_of_dimensions, "Dimension order length must match number of dimensions");

    ContainerType<DataType, (dimensions[VariadicIndices] * ...)> data = {};

    template <IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions)
    __attribute__((always_inline)) inline decltype(auto) at(DimTypes... dim) {
        checkBounds<dimensions[VariadicIndices]...>({(Dim_size_t)dim...});
        return data[((offsets[VariadicIndices] * std::get<VariadicIndices>(std::make_tuple(dim...))) + ...)];
    }

    template <IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions)
    __attribute__((always_inline)) constexpr inline value_type at(DimTypes... dim) const {
        checkBounds<dimensions[VariadicIndices]...>({(Dim_size_t)dim...});
        return data[((offsets[VariadicIndices] * std::get<VariadicIndices>(std::make_tuple(dim...))) + ...)];
    }

    template <DimensionOrder InterpretedDimensionOrder, IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions && InterpretedDimensionOrder.length() == order.length())
    __attribute__((always_inline)) inline decltype(auto) at(DimTypes... dim) {
        constexpr std::array<Dim_size_t, number_of_dimensions> permutation_order = order.template permutationOrderComputation<number_of_dimensions>(InterpretedDimensionOrder);
        const auto                                             dims_tupled       = std::make_tuple(dim...);
        const auto                                             dims_permuted     = std::make_tuple(std::get<permutation_order[VariadicIndices]>(dims_tupled)...);
        return this->at(std::get<VariadicIndices>(dims_permuted)...);
    }

    template <DimensionOrder InterpretedDimensionOrder, IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions && InterpretedDimensionOrder.length() == order.length())
    __attribute__((always_inline)) constexpr inline value_type at(DimTypes... dim) const {
        constexpr std::array<Dim_size_t, number_of_dimensions> permutation_order = order.template permutationOrderComputation<number_of_dimensions>(InterpretedDimensionOrder);
        const auto                                             dims_tupled       = std::make_tuple(dim...);
        const auto                                             dims_permuted     = std::make_tuple(std::get<permutation_order[VariadicIndices]>(dims_tupled)...);
        return this->at(std::get<VariadicIndices>(dims_permuted)...);
    }
};

template <IsMatrixType   BaseMatrixType, // The Base Matrix
          DimensionOrder LocalOrder,
          Dim_size_t... VariadicIndices>
    requires(LocalOrder.unique() && LocalOrder.length() > 0)
struct PermutedMatrixType<BaseMatrixType, LocalOrder, std::index_sequence<VariadicIndices...>> {

    using BaseMatrixTypeNoRef = std::remove_cvref_t<BaseMatrixType>;
    using value_type          = typename BaseMatrixTypeNoRef::value_type;
    using storage_type = std::conditional_t<std::is_rvalue_reference_v<BaseMatrixType>, std::remove_reference_t<BaseMatrixType>, std::add_lvalue_reference_t<std::remove_reference_t<BaseMatrixType>>>;

    static constexpr std::size_t                                  number_of_dimensions = BaseMatrixTypeNoRef::number_of_dimensions;
    static constexpr DimensionOrder                               order                = LocalOrder;
    static constexpr std::array<Dim_size_t, number_of_dimensions> permutation_indices  = LocalOrder.template permutationOrderComputation<number_of_dimensions>(BaseMatrixTypeNoRef::order);
    static constexpr std::array<Dim_size_t, number_of_dimensions> base_dimsensions     = BaseMatrixTypeNoRef::dimensions;
    static constexpr std::array<Dim_size_t, number_of_dimensions> dimensions =
            calculateDimensions<number_of_dimensions, base_dimsensions, permutation_indices>(std::index_sequence<VariadicIndices...>()); // {base_dimsensions[permutation_indices[VariadicIndices]]...};
    // {base_dimsensions[permutation_indices[VariadicIndices]]...};
    static constexpr bool k_has_zero_dimension = BaseMatrixTypeNoRef::k_has_zero_dimension;

    static constexpr auto permutation_to_base = BaseMatrixTypeNoRef::order.template permutationOrderComputation<number_of_dimensions>(LocalOrder);

    static_assert(sizeof...(VariadicIndices) == number_of_dimensions, "How The FUCK did you manage to get this error? Variadic indices must match number of dimensions");

    storage_type data;

    PermutedMatrixType(reference_or_rvalue<BaseMatrixType> ref) : data(ref) {
    }

    constexpr PermutedMatrixType(const_reference_or_rvalue<BaseMatrixType> ref) : data(ref) {
    }

    template <IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions && !std::is_const_v<std::remove_reference_t<storage_type>>)
    __attribute__((always_inline)) inline decltype(auto) at(DimTypes... dim) {
        checkBounds<dimensions[VariadicIndices]...>({(Dim_size_t)dim...});
        // return data.at(std::get<permutation_to_base[VariadicIndices]>(std::make_tuple(dim...))...);
        return data.template at<order>(dim...);
    }

    template <IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions)
    __attribute__((always_inline)) constexpr inline value_type at(DimTypes... dim) const {
        checkBounds<dimensions[VariadicIndices]...>({(Dim_size_t)dim...});
        // return data.at(std::get<permutation_to_base[VariadicIndices]>(std::make_tuple(dim...))...);
        return data.template at<order>(dim...);
    }

    template <DimensionOrder InterpretedDimensionOrder, IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions && InterpretedDimensionOrder.length() == order.length() && !std::is_const_v<std::remove_reference_t<storage_type>>)
    __attribute__((always_inline)) inline decltype(auto) at(DimTypes... dim) {
        constexpr std::array<Dim_size_t, number_of_dimensions> permutation_order = order.template permutationOrderComputation<number_of_dimensions>(InterpretedDimensionOrder);
        const auto                                             dims_tupled       = std::make_tuple(dim...);
        const auto                                             dims_permuted     = std::make_tuple(std::get<permutation_order[VariadicIndices]>(dims_tupled)...);
        return this->at(std::get<VariadicIndices>(dims_permuted)...);
    }

    template <DimensionOrder InterpretedDimensionOrder, IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions && InterpretedDimensionOrder.length() == order.length())
    __attribute__((always_inline)) constexpr inline value_type at(DimTypes... dim) const {
        constexpr std::array<Dim_size_t, number_of_dimensions> permutation_order = order.template permutationOrderComputation<number_of_dimensions>(InterpretedDimensionOrder);
        const auto                                             dims_tupled       = std::make_tuple(dim...);
        const auto                                             dims_permuted     = std::make_tuple(std::get<permutation_order[VariadicIndices]>(dims_tupled)...);
        return this->at(std::get<VariadicIndices>(dims_permuted)...);
    }
};

template <std::size_t ConcatenatedMatrixDimension, //
          Dim_size_t... VariadicIndices,
          IsMatrixType... MatrixTypes>
    requires(((variadicArrayCompare<DimensionOrder::type, DimensionOrder::max_length>(std::remove_cvref_t<MatrixTypes>::order.order...)[VariadicIndices] ||
               (VariadicIndices == ConcatenatedMatrixDimension)) &&
              ...))
struct ConcatenatedMatrixType<ConcatenatedMatrixDimension, std::index_sequence<VariadicIndices...>, MatrixTypes...> {

    using value_type = typename std::tuple_element_t<0, std::tuple<std::remove_cvref_t<MatrixTypes>...>>::value_type; // Assuming all matrices have the same value type
    using element_0  = std::tuple_element_t<0, std::tuple<std::remove_cvref_t<MatrixTypes>...>>;                      // First matrix type for dimensions and offsets
    using storage_type =
            std::tuple<std::conditional_t<std::is_rvalue_reference_v<MatrixTypes>, std::remove_reference_t<MatrixTypes>, std::add_lvalue_reference_t<std::remove_reference_t<MatrixTypes>>>...>;

    static constexpr std::size_t                                    number_of_dimensions = sizeof...(VariadicIndices);
    static constexpr DimensionOrder                                 order                = element_0::order;
    static constexpr std::array<Dim_size_t, number_of_dimensions>   dimensions           = {((VariadicIndices == ConcatenatedMatrixDimension)
                                                                                                     ? (variadicArrayAdd<number_of_dimensions>(std::remove_cvref_t<MatrixTypes>::dimensions...)[VariadicIndices])
                                                                                                     : element_0::dimensions[VariadicIndices])...};
    static constexpr std::array<Dim_size_t, sizeof...(MatrixTypes)> offsets              = variadicCumSum(std::remove_cvref_t<MatrixTypes>::dimensions[ConcatenatedMatrixDimension]...);
    static constexpr bool                                           k_has_zero_dimension = ((std::remove_cvref_t<MatrixTypes>::k_has_zero_dimension) || ...);

    static_assert(sizeof...(VariadicIndices) == number_of_dimensions, "How The FUCK did you manage to get this error? Variadic indices must match number of dimensions");

    storage_type data;

    ConcatenatedMatrixType(reference_or_rvalue<MatrixTypes>... values) : data(values...) {
    }

    constexpr ConcatenatedMatrixType(const_reference_or_rvalue<MatrixTypes>... values) : data(values...) {
    }

    template <IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions && !std::is_const_v<std::remove_reference_t<storage_type>>)
    __attribute__((always_inline)) inline decltype(auto) at(DimTypes... dim) {
        checkBounds<dimensions[VariadicIndices]...>({(Dim_size_t)dim...});
        return SelectionHelper<sizeof...(MatrixTypes) - 1, storage_type, DimTypes...>::dataSelection(data, dim...);
    }

    template <IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions)
    __attribute__((always_inline)) constexpr inline value_type at(DimTypes... dim) const {
        checkBounds<dimensions[VariadicIndices]...>({(Dim_size_t)dim...});
        return SelectionHelper<sizeof...(MatrixTypes) - 1, storage_type, DimTypes...>::dataSelection(data, dim...);
    }

    template <DimensionOrder InterpretedDimensionOrder, IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions && InterpretedDimensionOrder.length() == order.length() && !std::is_const_v<std::remove_reference_t<storage_type>>)
    __attribute__((always_inline)) inline decltype(auto) at(DimTypes... dim) {
        constexpr std::array<Dim_size_t, number_of_dimensions> permutation_order = order.template permutationOrderComputation<number_of_dimensions>(InterpretedDimensionOrder);
        return this->at(std::get<permutation_order[VariadicIndices]>(std::make_tuple(dim...))...);
    }

    template <DimensionOrder InterpretedDimensionOrder, IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions && InterpretedDimensionOrder.length() == order.length())
    __attribute__((always_inline)) constexpr inline value_type at(DimTypes... dim) const {
        constexpr std::array<Dim_size_t, number_of_dimensions> permutation_order = order.template permutationOrderComputation<number_of_dimensions>(InterpretedDimensionOrder);
        return this->at(std::get<permutation_order[VariadicIndices]>(std::make_tuple(dim...))...);
    }

  private:
    template <std::size_t Selection, typename DataTuple, IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions)
    struct SelectionHelper {
        __attribute__((always_inline)) static inline value_type &dataSelection(DataTuple &data, DimTypes... dim) {
            auto       tuple = std::make_tuple(dim...);
            const auto value = static_cast<Dim_size_t>(std::get<ConcatenatedMatrixDimension>(tuple));
            if (value >= offsets[Selection - 1]) {
                std::get<ConcatenatedMatrixDimension>(tuple) -= offsets[Selection - 1];
                return std::get<Selection>(data).template at<order>(std::get<VariadicIndices>(tuple)...);
            }
            return SelectionHelper<Selection - 1, DataTuple, DimTypes...>::dataSelection(data, dim...);
        }

        __attribute__((always_inline)) constexpr static inline value_type dataSelection(const DataTuple &data, DimTypes... dim) {
            auto       tuple = std::make_tuple(dim...);
            const auto value = static_cast<Dim_size_t>(std::get<ConcatenatedMatrixDimension>(tuple));
            if (value >= offsets[Selection - 1]) {
                std::get<ConcatenatedMatrixDimension>(tuple) -= offsets[Selection - 1];
                return std::get<Selection>(data).template at<order>(std::get<VariadicIndices>(tuple)...);
            }
            return SelectionHelper<Selection - 1, DataTuple, DimTypes...>::dataSelection(data, dim...);
        }
    };

    template <typename DataTuple, IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions)
    struct SelectionHelper<0, DataTuple, DimTypes...> {
        __attribute__((always_inline)) static inline value_type &dataSelection(DataTuple &data, DimTypes... dim) {
            return std::get<0>(data).template at<order>(dim...);
        }

        __attribute__((always_inline)) constexpr static inline value_type dataSelection(const DataTuple &data, DimTypes... dim) {
            return std::get<0>(data).template at<order>(dim...);
        }
    };
};

template <IsMatrixType                                 BaseMatrixType,
          DimensionOrder                               SlicedOrder,
          std::array<Dim_size_t, SlicedOrder.length()> Slices,
          Dim_size_t... VariadicIndices,
          Dim_size_t... VariadicSliceIndices,
          Dim_size_t... VariadicReducedIndices>
struct SlicedMatrixType<BaseMatrixType, SlicedOrder, Slices, std::index_sequence<VariadicIndices...>, std::index_sequence<VariadicSliceIndices...>, std::index_sequence<VariadicReducedIndices...>> {

    using BaseMatrixTypeNoRef = std::remove_cvref_t<BaseMatrixType>;
    using value_type          = typename BaseMatrixTypeNoRef::value_type;
    using storage_type = std::conditional_t<std::is_rvalue_reference_v<BaseMatrixType>, std::remove_reference_t<BaseMatrixType>, std::add_lvalue_reference_t<std::remove_reference_t<BaseMatrixType>>>;

    static constexpr std::size_t                                  number_of_dimensions = BaseMatrixTypeNoRef::number_of_dimensions;
    static constexpr DimensionOrder                               order                = BaseMatrixTypeNoRef::order;
    static constexpr std::array<Dim_size_t, SlicedOrder.length()> slices               = Slices;

    static constexpr std::array<Dim_size_t, number_of_dimensions> original_dimensions = BaseMatrixTypeNoRef::dimensions;

    static constexpr DimensionOrder order_sliced_at_back                = order.remove(SlicedOrder) + SlicedOrder;
    static constexpr auto           permutation_order_sliced_to_back    = order_sliced_at_back.template permutationOrderComputation<number_of_dimensions>(order);
    static constexpr auto           permutation_order_sliced_integrated = order.template permutationOrderComputation<number_of_dimensions>(order_sliced_at_back);

    static constexpr std::array<Dim_size_t, number_of_dimensions> dimensions = {std::get<permutation_order_sliced_integrated[VariadicIndices]>(
            std::make_tuple(original_dimensions[permutation_order_sliced_to_back[VariadicReducedIndices]]..., Slices[VariadicSliceIndices]...))...};

    static constexpr bool k_has_zero_dimension = ((dimensions[VariadicIndices] <= 0) || ...);

    static_assert(sizeof...(VariadicIndices) == number_of_dimensions, "How The FUCK did you manage to get this error? Variadic indices must match number of dimensions");
    static_assert(vmax(VariadicIndices...) == number_of_dimensions - 1, "How The FUCK did you manage to get this error? the higest variadic index must not exceed the number of dimensions");

    static_assert(((Slices[VariadicIndices] >= 0) || ...), "Slice sizes must be greater or equal 0");
    static_assert(((dimensions[VariadicIndices] <= BaseMatrixTypeNoRef::dimensions[VariadicIndices]) || ...), "Slices upper bound must be <= base matrix dimensions");

    storage_type                                       data;
    const std::array<Dim_size_t, SlicedOrder.length()> offset;

    SlicedMatrixType(reference_or_rvalue<BaseMatrixType> ref, const std::array<Dim_size_t, SlicedOrder.length()> offset = makeFilledArray<Dim_size_t, SlicedOrder.length()>(0))
            : data(ref), offset(offset) {
    }

    constexpr SlicedMatrixType(const_reference_or_rvalue<BaseMatrixType> ref, const std::array<Dim_size_t, SlicedOrder.length()> offset = makeFilledArray<Dim_size_t, SlicedOrder.length()>(0))
            : data(ref), offset(offset) {
    }

    template <IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions && !std::is_const_v<std::remove_reference_t<storage_type>>)
    __attribute__((always_inline)) inline decltype(auto) at(DimTypes... dim) {
        checkBounds<dimensions[VariadicIndices]...>({(Dim_size_t)dim...});
        const auto indices = std::make_tuple(std::get<permutation_order_sliced_to_back[VariadicIndices]>(std::make_tuple(static_cast<Dim_size_t>(dim)...))...);
        auto sliced = std::make_tuple(std::get<VariadicReducedIndices>(indices)..., (std::get<sizeof...(VariadicReducedIndices) + VariadicSliceIndices>(indices) + offset[VariadicSliceIndices])...);
        return data.template at<order>(std::get<permutation_order_sliced_integrated[VariadicIndices]>(sliced)...);
    }

    template <IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions)
    __attribute__((always_inline)) constexpr inline value_type at(DimTypes... dim) const {
        checkBounds<dimensions[VariadicIndices]...>({(Dim_size_t)dim...});
        const auto indices = std::make_tuple(std::get<permutation_order_sliced_to_back[VariadicIndices]>(std::make_tuple(static_cast<Dim_size_t>(dim)...))...);
        auto sliced = std::make_tuple(std::get<VariadicReducedIndices>(indices)..., (std::get<sizeof...(VariadicReducedIndices) + VariadicSliceIndices>(indices) + offset[VariadicSliceIndices])...);
        return data.template at<order>(std::get<permutation_order_sliced_integrated[VariadicIndices]>(sliced)...);
    }

    template <DimensionOrder InterpretedDimensionOrder, IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions && InterpretedDimensionOrder.length() == order.length() && !std::is_const_v<std::remove_reference_t<storage_type>>)
    __attribute__((always_inline)) inline decltype(auto) at(DimTypes... dim) {
        constexpr std::array<Dim_size_t, number_of_dimensions> permutation_order = order.template permutationOrderComputation<number_of_dimensions>(InterpretedDimensionOrder);
        return this->at(std::get<permutation_order[VariadicIndices]>(std::make_tuple(dim...))...);
    }

    template <DimensionOrder InterpretedDimensionOrder, IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions && InterpretedDimensionOrder.length() == order.length())
    __attribute__((always_inline)) constexpr inline value_type at(DimTypes... dim) const {
        constexpr std::array<Dim_size_t, number_of_dimensions> permutation_order = order.template permutationOrderComputation<number_of_dimensions>(InterpretedDimensionOrder);
        return this->at(std::get<permutation_order[VariadicIndices]>(std::make_tuple(dim...))...);
    }
};

template <IsMatrixType                                BaseMatrixType,
          DimensionOrder                              AddedOrder,
          std::array<Dim_size_t, AddedOrder.length()> Lengths,
          Dim_size_t... VariadicIndices,
          Dim_size_t... VariadicNewIndices,
          Dim_size_t... VariadicBaseIndices>
    requires((Lengths[VariadicNewIndices] > 0) && ...)
struct BroadcastedMatrixType<BaseMatrixType, AddedOrder, Lengths, std::index_sequence<VariadicIndices...>, std::index_sequence<VariadicNewIndices...>, std::index_sequence<VariadicBaseIndices...>> {

    using BaseMatrixTypeNoRef = std::remove_cvref_t<BaseMatrixType>;
    using value_type          = typename BaseMatrixTypeNoRef::value_type;
    using storage_type = std::conditional_t<std::is_rvalue_reference_v<BaseMatrixType>, std::remove_reference_t<BaseMatrixType>, std::add_lvalue_reference_t<std::remove_reference_t<BaseMatrixType>>>;

    static constexpr DimensionOrder                               order                = BaseMatrixTypeNoRef::order + AddedOrder;
    static constexpr std::size_t                                  number_of_dimensions = order.length();
    static constexpr std::array<Dim_size_t, number_of_dimensions> dimensions           = {BaseMatrixTypeNoRef::dimensions[VariadicBaseIndices]..., Lengths[VariadicNewIndices]...};

    static constexpr bool k_has_zero_dimension = BaseMatrixTypeNoRef::k_has_zero_dimension;

    static_assert(sizeof...(VariadicIndices) == number_of_dimensions, "How the FUCK did you manage to get this error? Variadic indices must match number of dimensions");

    storage_type data;

    BroadcastedMatrixType(reference_or_rvalue<BaseMatrixType> ref) : data(ref) {
    }

    constexpr BroadcastedMatrixType(const_reference_or_rvalue<BaseMatrixType> ref) : data(ref) {
    }

    template <IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions && !std::is_const_v<std::remove_reference_t<storage_type>>)
    __attribute__((always_inline)) inline decltype(auto) at(DimTypes... dim) {
        checkBounds<dimensions[VariadicIndices]...>({(Dim_size_t)dim...});
        return data.template at<BaseMatrixTypeNoRef::order>(std::get<VariadicBaseIndices>(std::make_tuple(dim...))...);
    }

    template <IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions)
    __attribute__((always_inline)) constexpr inline value_type at(DimTypes... dim) const {
        checkBounds<dimensions[VariadicIndices]...>({(Dim_size_t)dim...});
        return data.template at<BaseMatrixTypeNoRef::order>(std::get<VariadicBaseIndices>(std::make_tuple(dim...))...);
    }

    template <DimensionOrder InterpretedDimensionOrder, IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions && InterpretedDimensionOrder.length() == order.length() && !std::is_const_v<std::remove_reference_t<storage_type>>)
    __attribute__((always_inline)) inline decltype(auto) at(DimTypes... dim) {
        constexpr std::array<Dim_size_t, number_of_dimensions> permutation_order = order.template permutationOrderComputation<number_of_dimensions>(InterpretedDimensionOrder);
        return this->at(std::get<permutation_order[VariadicIndices]>(std::make_tuple(dim...))...);
    }

    template <DimensionOrder InterpretedDimensionOrder, IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions && InterpretedDimensionOrder.length() == order.length())
    __attribute__((always_inline)) constexpr inline value_type at(DimTypes... dim) const {
        constexpr std::array<Dim_size_t, number_of_dimensions> permutation_order = order.template permutationOrderComputation<number_of_dimensions>(InterpretedDimensionOrder);
        return this->at(std::get<permutation_order[VariadicIndices]>(std::make_tuple(dim...))...);
    }
};

template <IsMatrixType                                   BaseMatrixType,
          DimensionOrder                                 ReplicatOrder,
          std::array<Dim_size_t, ReplicatOrder.length()> Lengths,
          std::size_t... VariadicIndices,
          std::size_t... VariadicNewIndices,
          std::size_t... VariadicReducedIndices>
    requires(std::remove_cvref_t<BaseMatrixType>::order.containsAll(ReplicatOrder) && ReplicatOrder.unique() && ReplicatOrder.length() > 0)
struct ReplicatedMatrixType<BaseMatrixType,
                            ReplicatOrder,
                            Lengths,
                            std::index_sequence<VariadicIndices...>,
                            std::index_sequence<VariadicNewIndices...>,
                            std::index_sequence<VariadicReducedIndices...>> {
    using BaseMatrixTypeNoRef = std::remove_cvref_t<BaseMatrixType>;
    using value_type          = typename BaseMatrixTypeNoRef::value_type;
    using storage_type = std::conditional_t<std::is_rvalue_reference_v<BaseMatrixType>, std::remove_reference_t<BaseMatrixType>, std::add_lvalue_reference_t<std::remove_reference_t<BaseMatrixType>>>;

    static constexpr DimensionOrder                                 order                  = BaseMatrixTypeNoRef::order;
    static constexpr DimensionOrder                                 order_original_at_back = order.remove(ReplicatOrder) + ReplicatOrder;
    static constexpr std::size_t                                    number_of_dimensions   = order.length();
    static constexpr std::array<Dim_size_t, number_of_dimensions>   original_dimensions    = BaseMatrixTypeNoRef::dimensions;
    static constexpr std::array<Dim_size_t, ReplicatOrder.length()> zeros                  = makeFilledArray<Dim_size_t, ReplicatOrder.length()>(0);

    static constexpr bool k_has_zero_dimension = BaseMatrixTypeNoRef::k_has_zero_dimension;

    static constexpr auto permutation_order         = order_original_at_back.template permutationOrderComputation<number_of_dimensions>(order);
    static constexpr auto permutation_order_inverse = order.template permutationOrderComputation<number_of_dimensions>(order_original_at_back);

    static constexpr std::array<Dim_size_t, number_of_dimensions> dimensions = {
            std::get<permutation_order_inverse[VariadicIndices]>(std::make_tuple(original_dimensions[permutation_order[VariadicReducedIndices]]..., Lengths[VariadicNewIndices]...))...};

    static_assert(sizeof...(VariadicIndices) == number_of_dimensions, "How the FUCK did you manage to get this error? Variadic indices must match number of dimensions");
    static_assert(((BaseMatrixTypeNoRef::dimensions[order.indexOf(ReplicatOrder.order[VariadicNewIndices])] == 1) && ...), "The replicated dimensions must be base 1");

    storage_type data;

    ReplicatedMatrixType(reference_or_rvalue<BaseMatrixType> ref) : data(ref) {
    }

    constexpr ReplicatedMatrixType(const_reference_or_rvalue<BaseMatrixType> ref) : data(ref) {
    }

    template <IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions && !std::is_const_v<std::remove_reference_t<storage_type>>)
    __attribute__((always_inline)) inline decltype(auto) at(DimTypes... dim) {
        checkBounds<dimensions[VariadicIndices]...>({(Dim_size_t)dim...});
        const auto permuted_dims = std::make_tuple(std::get<permutation_order[VariadicReducedIndices]>(std::make_tuple(dim...))..., zeros[VariadicNewIndices]...);
        const auto dims_in_order = std::make_tuple(std::get<permutation_order_inverse[VariadicIndices]>(permuted_dims)...);
        return data.at(std::get<VariadicIndices>(dims_in_order)...);
    }

    template <IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions)
    __attribute__((always_inline)) constexpr inline value_type at(DimTypes... dim) const {
        checkBounds<dimensions[VariadicIndices]...>({(Dim_size_t)dim...});
        const auto permuted_dims = std::make_tuple(std::get<permutation_order[VariadicReducedIndices]>(std::make_tuple(dim...))..., zeros[VariadicNewIndices]...);
        const auto dims_in_order = std::make_tuple(std::get<permutation_order_inverse[VariadicIndices]>(permuted_dims)...);
        return data.at(std::get<VariadicIndices>(dims_in_order)...);
    }

    template <DimensionOrder InterpretedDimensionOrder, IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions && InterpretedDimensionOrder.length() == order.length() && !std::is_const_v<std::remove_reference_t<storage_type>>)
    __attribute__((always_inline)) inline decltype(auto) at(DimTypes... dim) {
        constexpr std::array<Dim_size_t, number_of_dimensions> permutation_order = order.template permutationOrderComputation<number_of_dimensions>(InterpretedDimensionOrder);
        return this->at(std::get<permutation_order[VariadicIndices]>(std::make_tuple(dim...))...);
    }

    template <DimensionOrder InterpretedDimensionOrder, IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions && InterpretedDimensionOrder.length() == order.length())
    __attribute__((always_inline)) constexpr inline value_type at(DimTypes... dim) const {
        constexpr std::array<Dim_size_t, number_of_dimensions> permutation_order = order.template permutationOrderComputation<number_of_dimensions>(InterpretedDimensionOrder);
        return this->at(std::get<permutation_order[VariadicIndices]>(std::make_tuple(dim...))...);
    }
};

template <IsMatrixType BaseMatrixType, DimensionOrder ReplaceFrom, DimensionOrder ReplaceTo, Dim_size_t... VariadicIndices>
struct ReplacedMatrixType<BaseMatrixType, ReplaceFrom, ReplaceTo, std::index_sequence<VariadicIndices...>> {
    using BaseMatrixTypeNoRef = std::remove_cvref_t<BaseMatrixType>;
    using value_type          = typename BaseMatrixTypeNoRef::value_type;
    using storage_type = std::conditional_t<std::is_rvalue_reference_v<BaseMatrixType>, std::remove_reference_t<BaseMatrixType>, std::add_lvalue_reference_t<std::remove_reference_t<BaseMatrixType>>>;

    static constexpr DimensionOrder                               order                = BaseMatrixTypeNoRef::order.replace(ReplaceFrom, ReplaceTo);
    static constexpr std::size_t                                  number_of_dimensions = order.length();
    static constexpr std::array<Dim_size_t, number_of_dimensions> dimensions           = BaseMatrixTypeNoRef::dimensions;
    static constexpr bool                                         k_has_zero_dimension = BaseMatrixTypeNoRef::k_has_zero_dimension;

    static_assert(sizeof...(VariadicIndices) == number_of_dimensions, "How the FUCK did you manage to get this error? Variadic indices must match number of dimensions");

    storage_type data;

    ReplacedMatrixType(reference_or_rvalue<BaseMatrixType> ref) : data(ref) {
    }

    constexpr ReplacedMatrixType(const_reference_or_rvalue<BaseMatrixType> ref) : data(ref) {
    }

    template <IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions && !std::is_const_v<std::remove_reference_t<storage_type>>)
    __attribute__((always_inline)) inline decltype(auto) at(DimTypes... dim) {
        checkBounds<dimensions[VariadicIndices]...>({(Dim_size_t)dim...});
        return data.at(dim...);
    }

    template <IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions)
    __attribute__((always_inline)) constexpr inline value_type at(DimTypes... dim) const {
        checkBounds<dimensions[VariadicIndices]...>({(Dim_size_t)dim...});
        return data.at(dim...);
    }

    template <DimensionOrder InterpretedDimensionOrder, IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions && InterpretedDimensionOrder.length() == order.length() && !std::is_const_v<std::remove_reference_t<storage_type>>)
    __attribute__((always_inline)) inline decltype(auto) at(DimTypes... dim) {
        constexpr std::array<Dim_size_t, number_of_dimensions> permutation_order = order.template permutationOrderComputation<number_of_dimensions>(InterpretedDimensionOrder);
        return this->at(std::get<permutation_order[VariadicIndices]>(std::make_tuple(dim...))...);
    }

    template <DimensionOrder InterpretedDimensionOrder, IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions && InterpretedDimensionOrder.length() == order.length())
    __attribute__((always_inline)) constexpr inline value_type at(DimTypes... dim) const {
        constexpr std::array<Dim_size_t, number_of_dimensions> permutation_order = order.template permutationOrderComputation<number_of_dimensions>(InterpretedDimensionOrder);
        return this->at(std::get<permutation_order[VariadicIndices]>(std::make_tuple(dim...))...);
    }
};

template <IsMatrixType BaseMatrixType, Dim_size_t... VariadicIndices>
struct NegativeMatrixType<BaseMatrixType, std::index_sequence<VariadicIndices...>> {
    using BaseMatrixTypeNoRef = std::remove_cvref_t<BaseMatrixType>;
    using value_type          = typename BaseMatrixTypeNoRef::value_type;
    using storage_type = std::conditional_t<std::is_rvalue_reference_v<BaseMatrixType>, std::remove_reference_t<BaseMatrixType>, std::add_lvalue_reference_t<std::remove_reference_t<BaseMatrixType>>>;
    static constexpr std::size_t                                  number_of_dimensions = BaseMatrixTypeNoRef::number_of_dimensions;
    static constexpr DimensionOrder                               order                = BaseMatrixTypeNoRef::order;
    static constexpr std::array<Dim_size_t, number_of_dimensions> dimensions           = BaseMatrixTypeNoRef::dimensions;
    static constexpr bool                                         k_has_zero_dimension = BaseMatrixTypeNoRef::k_has_zero_dimension;

    static_assert(sizeof...(VariadicIndices) == number_of_dimensions, "How The FUCK did you manage to get this error? Variadic indices must match number of dimensions");
    static_assert(order.length() == number_of_dimensions, "Dimension order length must match number of dimensions");

    storage_type data;

    NegativeMatrixType(reference_or_rvalue<BaseMatrixType> ref) : data(ref) {
    }

    constexpr NegativeMatrixType(const_reference_or_rvalue<BaseMatrixType> ref) : data(ref) {
    }

    template <IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions)
    __attribute__((always_inline)) constexpr inline value_type at(DimTypes... dim) const {
        checkBounds<dimensions[VariadicIndices]...>({(Dim_size_t)dim...});
        return -data.template at<order>(dim...);
    }

    template <DimensionOrder InterpretedDimensionOrder, IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions && InterpretedDimensionOrder.length() == order.length())
    __attribute__((always_inline)) inline decltype(auto) at(DimTypes... dim) const {
        constexpr std::array<Dim_size_t, number_of_dimensions> permutation_order = order.template permutationOrderComputation<number_of_dimensions>(InterpretedDimensionOrder);
        return this->at(std::get<permutation_order[VariadicIndices]>(std::make_tuple(dim...))...);
    }
};

template <IsMatrixType                         BaseMatrixType,
          DimensionOrder                       Old,
          DimensionOrder                       New,
          std::array<Dim_size_t, New.length()> NewDimensions,
          Dim_size_t... VariadicIndices,
          Dim_size_t... VariadicNewIndices,
          Dim_size_t... VariadicOldIndices,
          Dim_size_t... VariadicOldIndicesM1>
    requires(sizeof...(VariadicNewIndices) == New.length() && sizeof...(VariadicOldIndicesM1) + 1 == std::remove_cvref_t<BaseMatrixType>::order.length())
struct SplitMatrixType<BaseMatrixType,
                       Old,
                       New,
                       NewDimensions,
                       std::index_sequence<VariadicIndices...>,
                       std::index_sequence<VariadicNewIndices...>,
                       std::index_sequence<VariadicOldIndices...>,
                       std::index_sequence<VariadicOldIndicesM1...>> {
    using BaseMatrixTypeNoRef = std::remove_cvref_t<BaseMatrixType>;
    using value_type          = typename BaseMatrixTypeNoRef::value_type;
    using storage_type = std::conditional_t<std::is_rvalue_reference_v<BaseMatrixType>, std::remove_reference_t<BaseMatrixType>, std::add_lvalue_reference_t<std::remove_reference_t<BaseMatrixType>>>;

    static constexpr DimensionOrder                                        order                = BaseMatrixTypeNoRef::order.insert(Old[0], New);
    static constexpr std::size_t                                           number_of_dimensions = order.length();
    static constexpr std::array<Dim_size_t, sizeof...(VariadicNewIndices)> offsets              = calculateOffsets<NewDimensions[VariadicNewIndices]...>();

    static constexpr DimensionOrder order_original         = BaseMatrixTypeNoRef::order;
    static constexpr DimensionOrder order_original_at_back = order_original.remove(Old[0]) + Old;
    static constexpr DimensionOrder order_split_at_back    = order_original.remove(Old[0]) + New;

    static constexpr std::array<Dim_size_t, BaseMatrixTypeNoRef::number_of_dimensions> permutation_order_remove_to_back =
            order_original_at_back.template permutationOrderComputation<BaseMatrixTypeNoRef::number_of_dimensions>(order_original);
    static constexpr std::array<Dim_size_t, number_of_dimensions> permutation_order_new_to_correct = order.template permutationOrderComputation<number_of_dimensions>(order_split_at_back);

    // permute( permute (dims)[slice] append new dims) -> corretly ordered dimensions
    static constexpr std::array<Dim_size_t, number_of_dimensions> dimensions{std::get<permutation_order_new_to_correct[VariadicIndices]>(
            std::make_tuple(BaseMatrixTypeNoRef::dimensions[permutation_order_remove_to_back[VariadicOldIndicesM1]]..., NewDimensions[VariadicNewIndices]...))...};

    static constexpr Dim_size_t removed_dimension =
            std::get<BaseMatrixTypeNoRef::number_of_dimensions - 1>(std::make_tuple(BaseMatrixTypeNoRef::dimensions[permutation_order_remove_to_back[VariadicOldIndices]]...));
    static constexpr Dim_size_t new_dimension        = (NewDimensions[VariadicNewIndices] * ...);
    static constexpr bool       k_has_zero_dimension = (BaseMatrixTypeNoRef::k_has_zero_dimension || ((NewDimensions[VariadicNewIndices] == 0) || ...));

    static_assert(sizeof...(VariadicIndices) == number_of_dimensions, "How The FUCK did you manage to get this error? Variadic indices must match number of dimensions");
    // static_assert(new_dimension == removed_dimension, "Split dimensions must in product result in the removed dimension");

    storage_type data;

    SplitMatrixType(reference_or_rvalue<BaseMatrixType> ref) : data(ref) {
    }

    constexpr SplitMatrixType(const_reference_or_rvalue<BaseMatrixType> ref) : data(ref) {
    }

    template <IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions && !std::is_const_v<std::remove_reference_t<storage_type>>)
    __attribute__((always_inline)) inline decltype(auto) at(DimTypes... dim) {
        checkBounds<dimensions[VariadicIndices]...>({(Dim_size_t)dim...});
        constexpr auto permutation_to_split_at_back  = order_split_at_back.permutationOrderComputation<number_of_dimensions>(order);
        constexpr auto permutation_to_original_order = order_original.template permutationOrderComputation<sizeof...(VariadicOldIndices)>(order_original_at_back);

        const auto       permuted_dims          = std::make_tuple(std::get<permutation_to_split_at_back[VariadicIndices]>(std::make_tuple(dim...))...);
        const Dim_size_t split_collapsed_fim    = ((offsets[VariadicNewIndices] * std::get<BaseMatrixTypeNoRef::number_of_dimensions - 1 + VariadicNewIndices>(permuted_dims)) + ...);
        const auto       collapsed_dims         = std::make_tuple(std::get<VariadicOldIndicesM1>(permuted_dims)..., split_collapsed_fim);
        const auto       ordered_collapsed_dims = std::make_tuple(std::get<permutation_to_original_order[VariadicOldIndices]>(collapsed_dims)...);
        return data.template at<order_original>(std::get<VariadicOldIndices>(ordered_collapsed_dims)...);
    }

    template <IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions)
    __attribute__((always_inline)) constexpr inline value_type at(DimTypes... dim) const {
        checkBounds<dimensions[VariadicIndices]...>({(Dim_size_t)dim...});
        constexpr auto permutation_to_split_at_back  = order_split_at_back.permutationOrderComputation<number_of_dimensions>(order);
        constexpr auto permutation_to_original_order = order_original.template permutationOrderComputation<sizeof...(VariadicOldIndices)>(order_original_at_back);

        const auto       permuted_dims          = std::make_tuple(std::get<permutation_to_split_at_back[VariadicIndices]>(std::make_tuple(dim...))...);
        const Dim_size_t split_collapsed_fim    = ((offsets[VariadicNewIndices] * std::get<BaseMatrixTypeNoRef::number_of_dimensions - 1 + VariadicNewIndices>(permuted_dims)) + ...);
        const auto       collapsed_dims         = std::make_tuple(std::get<VariadicOldIndicesM1>(permuted_dims)..., split_collapsed_fim);
        const auto       ordered_collapsed_dims = std::make_tuple(std::get<permutation_to_original_order[VariadicOldIndices]>(collapsed_dims)...);
        return data.template at<order_original>(std::get<VariadicOldIndices>(ordered_collapsed_dims)...);
    }

    template <DimensionOrder InterpretedDimensionOrder, IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions && InterpretedDimensionOrder.length() == order.length() && !std::is_const_v<std::remove_reference_t<storage_type>>)
    __attribute__((always_inline)) inline decltype(auto) at(DimTypes... dim) {
        constexpr std::array<Dim_size_t, number_of_dimensions> permutation_order = order.template permutationOrderComputation<number_of_dimensions>(InterpretedDimensionOrder);
        return this->at(std::get<permutation_order[VariadicIndices]>(std::make_tuple(dim...))...);
    }

    template <DimensionOrder InterpretedDimensionOrder, IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions && InterpretedDimensionOrder.length() == order.length())
    __attribute__((always_inline)) constexpr inline value_type at(DimTypes... dim) const {
        constexpr std::array<Dim_size_t, number_of_dimensions> permutation_order = order.template permutationOrderComputation<number_of_dimensions>(InterpretedDimensionOrder);
        return this->at(std::get<permutation_order[VariadicIndices]>(std::make_tuple(dim...))...);
    }
};

template <IsMatrixType   BaseMatrixType,
          DimensionOrder Old,
          DimensionOrder New,
          std::size_t... VariadicIndices,
          std::size_t... VariadicIndicesM1,
          std::size_t... VariadicOriginalIndices,
          std::size_t... VariadicOldIndices>
struct CollapsedMatrixType<BaseMatrixType,
                            Old,
                            New,
                            std::index_sequence<VariadicIndices...>,
                            std::index_sequence<VariadicIndicesM1...>,
                            std::index_sequence<VariadicOriginalIndices...>,
                            std::index_sequence<VariadicOldIndices...>> {
    using BaseMatrixTypeNoRef = std::remove_cvref_t<BaseMatrixType>;
    using value_type          = typename BaseMatrixTypeNoRef::value_type;
    using storage_type = std::conditional_t<std::is_rvalue_reference_v<BaseMatrixType>, std::remove_reference_t<BaseMatrixType>, std::add_lvalue_reference_t<std::remove_reference_t<BaseMatrixType>>>;
    static constexpr DimensionOrder order                = BaseMatrixTypeNoRef::order.remove(Old) + New;
    static constexpr std::size_t    number_of_dimensions = order.length();

    static constexpr DimensionOrder order_original         = BaseMatrixTypeNoRef::order;
    static constexpr DimensionOrder order_original_at_back = order_original.remove(Old) + Old;

    static constexpr std::array<Dim_size_t, BaseMatrixTypeNoRef::number_of_dimensions> permutation_order_remove_to_back =
            order_original_at_back.template permutationOrderComputation<BaseMatrixTypeNoRef::number_of_dimensions>(order_original);

    static constexpr std::array<Dim_size_t, BaseMatrixTypeNoRef::number_of_dimensions> dimensions_ordered_back{
            BaseMatrixTypeNoRef::dimensions[permutation_order_remove_to_back[VariadicOriginalIndices]]...};
    static constexpr std::array<Dim_size_t, sizeof...(VariadicOldIndices)> removed_dimensions{dimensions_ordered_back[number_of_dimensions - 1 + VariadicOldIndices]...};
    static constexpr Dim_size_t                                            new_dimension_product = (removed_dimensions[VariadicOldIndices] * ...);
    static constexpr std::array<Dim_size_t, number_of_dimensions>          dimensions{dimensions_ordered_back[VariadicIndicesM1]..., new_dimension_product};

    static constexpr std::array<Dim_size_t, sizeof...(VariadicOldIndices) + 1> offsets{new_dimension_product, calculateOffsets<removed_dimensions[VariadicOldIndices]...>()[VariadicOldIndices]...};

    static constexpr bool k_has_zero_dimension = ((dimensions[VariadicIndices] == 0) || ...);

    static_assert(sizeof...(VariadicIndices) == number_of_dimensions, "How The FUCK did you manage to get this error? Variadic indices must match number of dimensions");
    static_assert((BaseMatrixTypeNoRef::dimensions[VariadicOriginalIndices] * ...) == (dimensions[VariadicIndices] * ...), "The total size has changed, this should be impossible");

    storage_type data;

    CollapsedMatrixType(reference_or_rvalue<BaseMatrixType> ref) : data(ref) {
    }

    constexpr CollapsedMatrixType(const_reference_or_rvalue<BaseMatrixType> ref) : data(ref) {
    }

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdivision-by-zero"

    // zero division gets caught by bounds checker
    template <IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions && !std::is_const_v<std::remove_reference_t<storage_type>>)
    __attribute__((always_inline)) inline decltype(auto) at(DimTypes... dim) {
        checkBounds<dimensions[VariadicIndices]...>({(Dim_size_t)dim...});

        const Dim_size_t collapsed_dim = std::get<number_of_dimensions - 1>(std::make_tuple(dim...));

        const auto collapsed_dim_split = std::make_tuple((collapsed_dim % offsets[VariadicOldIndices]) / offsets[VariadicOldIndices + 1]...);
        const auto full_dims           = std::make_tuple(std::get<VariadicIndicesM1>(std::make_tuple(dim...))..., std::get<VariadicOldIndices>(collapsed_dim_split)...);
        const auto full_dims_original_order =
                std::make_tuple(std::get<order_original.template permutationOrderComputation<sizeof...(VariadicOriginalIndices)>(order_original_at_back)[VariadicOriginalIndices]>(full_dims)...);
        return data.template at<order_original>(std::get<VariadicOriginalIndices>(full_dims_original_order)...);
    }

    template <IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions)
    __attribute__((always_inline)) constexpr inline value_type at(DimTypes... dim) const {
        checkBounds<dimensions[VariadicIndices]...>({(Dim_size_t)dim...});

        const Dim_size_t collapsed_dim = std::get<number_of_dimensions - 1>(std::make_tuple(dim...));

        const auto collapsed_dim_split = std::make_tuple((collapsed_dim % offsets[VariadicOldIndices]) / offsets[VariadicOldIndices + 1]...);
        const auto full_dims           = std::make_tuple(std::get<VariadicIndicesM1>(std::make_tuple(dim...))..., std::get<VariadicOldIndices>(collapsed_dim_split)...);
        const auto full_dims_original_order =
                std::make_tuple(std::get<order_original.template permutationOrderComputation<sizeof...(VariadicOriginalIndices)>(order_original_at_back)[VariadicOriginalIndices]>(full_dims)...);
        return data.template at<order_original>(std::get<VariadicOriginalIndices>(full_dims_original_order)...);
    }

#pragma clang diagnostic pop

    template <DimensionOrder InterpretedDimensionOrder, IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions && InterpretedDimensionOrder.length() == order.length() && !std::is_const_v<std::remove_reference_t<storage_type>>)
    __attribute__((always_inline)) inline decltype(auto) at(DimTypes... dim) {
        constexpr std::array<Dim_size_t, number_of_dimensions> permutation_order = order.template permutationOrderComputation<number_of_dimensions>(InterpretedDimensionOrder);
        return this->at(std::get<permutation_order[VariadicIndices]>(std::make_tuple(dim...))...);
    }

    template <DimensionOrder InterpretedDimensionOrder, IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions && InterpretedDimensionOrder.length() == order.length())
    __attribute__((always_inline)) constexpr inline value_type at(DimTypes... dim) const {
        constexpr std::array<Dim_size_t, number_of_dimensions> permutation_order = order.template permutationOrderComputation<number_of_dimensions>(InterpretedDimensionOrder);
        return this->at(std::get<permutation_order[VariadicIndices]>(std::make_tuple(dim...))...);
    }
};

template <IsMatrixType BaseMatrixType>
struct ReferencedMatrixType {
    using BaseMatrixTypeNoRef = std::remove_cvref_t<BaseMatrixType>;
    using value_type          = typename BaseMatrixTypeNoRef::value_type;
    using storage_type = std::conditional_t<std::is_rvalue_reference_v<BaseMatrixType>, std::remove_reference_t<BaseMatrixType>, std::add_lvalue_reference_t<std::remove_reference_t<BaseMatrixType>>>;
    static constexpr DimensionOrder order                = BaseMatrixTypeNoRef::order;
    static constexpr std::size_t    number_of_dimensions = BaseMatrixTypeNoRef::number_of_dimensions;

    static constexpr std::array<Dim_size_t, number_of_dimensions> dimensions           = BaseMatrixTypeNoRef::dimensions;
    static constexpr bool                                         k_has_zero_dimension = BaseMatrixTypeNoRef::k_has_zero_dimension;

    storage_type data;

    ReferencedMatrixType(reference_or_rvalue<BaseMatrixType> ref) : data(ref) {
    }

    constexpr ReferencedMatrixType(const_reference_or_rvalue<BaseMatrixType> ref) : data(ref) {
    }

    template <IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions && !std::is_const_v<std::remove_reference_t<storage_type>>)
    __attribute__((always_inline)) inline decltype(auto) at(DimTypes... dim) {
        return data.at(dim...);
    }

    template <IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions)
    __attribute__((always_inline)) constexpr inline value_type at(DimTypes... dim) const {
        return data.at(dim...);
    }

    template <DimensionOrder InterpretedDimensionOrder, IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions && InterpretedDimensionOrder.length() == order.length() && !std::is_const_v<std::remove_reference_t<storage_type>>)
    __attribute__((always_inline)) inline decltype(auto) at(DimTypes... dim) {
        return data.template at<InterpretedDimensionOrder>(dim...);
    }

    template <DimensionOrder InterpretedDimensionOrder, IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions && InterpretedDimensionOrder.length() == order.length())
    __attribute__((always_inline)) constexpr inline value_type at(DimTypes... dim) const {
        return data.template at<InterpretedDimensionOrder>(dim...);
    }
};

template <std::size_t... Is, IsMatrixType... BaseMatrixType>
    requires(((std::tuple_element_t<0, std::tuple<std::remove_cvref_t<BaseMatrixType>...>>::number_of_dimensions == std::remove_cvref_t<BaseMatrixType>::number_of_dimensions) &&
              ...) // All matrices must have the same number of dimensions
             && ((std::tuple_element_t<0, std::tuple<std::remove_cvref_t<BaseMatrixType>...>>::order == std::remove_cvref_t<BaseMatrixType>::order) &&
                 ...) // All matrices must have the same dimension order
             && ((std::tuple_element_t<0, std::tuple<std::remove_cvref_t<BaseMatrixType>...>>::dimensions == std::remove_cvref_t<BaseMatrixType>::dimensions) &&
                 ...) // All matrices must have the same dimensions
             )
struct FusedMatrixType<std::index_sequence<Is...>, BaseMatrixType...> {
    using value_type                                                                   = std::tuple<typename std::remove_cvref_t<BaseMatrixType>::value_type...>;
    using BaseType0NoRef                                                               = std::remove_cvref_t<std::tuple_element_t<0, std::tuple<std::remove_cvref_t<BaseMatrixType>...>>>;
    static constexpr std::size_t                                  number_of_dimensions = BaseType0NoRef::number_of_dimensions;
    static constexpr DimensionOrder                               order                = BaseType0NoRef::order;
    static constexpr std::array<Dim_size_t, number_of_dimensions> dimensions           = BaseType0NoRef::dimensions;
    static constexpr bool                                         k_has_zero_dimension = BaseType0NoRef::k_has_zero_dimension;

    static_assert(order.length() == number_of_dimensions, "Dimension order length must match number of dimensions");
    template <typename Type>
    using conditional_dereference = std::conditional_t<std::is_rvalue_reference_v<Type>, std::remove_reference_t<Type>, std::add_lvalue_reference_t<std::remove_reference_t<Type>>>;

    using storage_type = std::tuple<conditional_dereference<BaseMatrixType>...>;

    storage_type data;

    FusedMatrixType(reference_or_rvalue<BaseMatrixType>... ref) : data(ref...) {
    }

    constexpr FusedMatrixType(const_reference_or_rvalue<BaseMatrixType>... ref) : data(ref...) {
    }

    template <IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions)
    __attribute__((always_inline)) inline decltype(auto) at(DimTypes... dim) {
        return std::tie(std::get<Is>(data).at(dim...)...);
    }

    template <IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions)
    __attribute__((always_inline)) constexpr inline value_type at(DimTypes... dim) const {
        return std::make_tuple(std::get<Is>(data).at(dim...)...);
    }

    template <DimensionOrder InterpretedDimensionOrder, IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions && InterpretedDimensionOrder.length() == order.length())
    __attribute__((always_inline)) inline decltype(auto) at(DimTypes... dim) {
        return std::tie(std::get<Is>(data).template at<InterpretedDimensionOrder>(dim...)...);
    }

    template <DimensionOrder InterpretedDimensionOrder, IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions && InterpretedDimensionOrder.length() == order.length())
    __attribute__((always_inline)) constexpr inline value_type at(DimTypes... dim) const {
        return std::make_tuple(std::get<Is>(data).template at<InterpretedDimensionOrder>(dim...)...);
    }
};

template <std::size_t Is, IsFuzedMatrixType BaseMatrixType>
    requires(Is < std::tuple_size_v<typename std::remove_cvref_t<BaseMatrixType>::value_type> // Index is within bounds
             && Is >= 0                                                                       // Index is non-negative
             )
struct SelectFusedMatrixType {
    using BaseMatrixTypeNoRef = std::remove_cvref_t<BaseMatrixType>;
    using storage_type = std::conditional_t<std::is_rvalue_reference_v<BaseMatrixType>, std::remove_reference_t<BaseMatrixType>, std::add_lvalue_reference_t<std::remove_reference_t<BaseMatrixType>>>;
    using value_type   = std::tuple_element_t<Is, typename BaseMatrixTypeNoRef::value_type>;
    static constexpr std::size_t                                  number_of_dimensions = BaseMatrixTypeNoRef::number_of_dimensions;
    static constexpr DimensionOrder                               order                = BaseMatrixTypeNoRef::order;
    static constexpr std::array<Dim_size_t, number_of_dimensions> dimensions           = BaseMatrixTypeNoRef::dimensions;
    static constexpr bool                                         k_has_zero_dimension = BaseMatrixTypeNoRef::k_has_zero_dimension;

    static_assert(order.length() == number_of_dimensions, "Dimension order length must match number of dimensions");

    storage_type data;

    SelectFusedMatrixType(reference_or_rvalue<BaseMatrixType> ref) : data(ref) {
    }

    constexpr SelectFusedMatrixType(const_reference_or_rvalue<BaseMatrixType> ref) : data(ref) {
    }

    template <IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions)
    __attribute__((always_inline)) inline decltype(auto) at(DimTypes... dim) {
        return std::get<Is>(data.at(dim...));
    }

    template <IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions)
    __attribute__((always_inline)) constexpr inline value_type at(DimTypes... dim) const {
        return std::get<Is>(data.at(dim...));
    }

    template <DimensionOrder InterpretedDimensionOrder, IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions && InterpretedDimensionOrder.length() == order.length())
    __attribute__((always_inline)) inline decltype(auto) at(DimTypes... dim) {
        return std::get<Is>(data.template at<InterpretedDimensionOrder>(dim...));
    }

    template <DimensionOrder InterpretedDimensionOrder, IsIndexType... DimTypes>
        requires(sizeof...(DimTypes) == number_of_dimensions && InterpretedDimensionOrder.length() == order.length())
    __attribute__((always_inline)) constexpr inline value_type at(DimTypes... dim) const {
        return std::get<Is>(data.template at<InterpretedDimensionOrder>(dim...));
    }
};

// template <typename Type, auto dimension_order, template <typename, std::size_t> class ContainerType=std::array , Dim_size_t... Dims>
// using SimpleMatrix = Matrix<Type, dimension_order, ContainerType, std::index_sequence<Dims...>, std::make_index_sequence<sizeof...(Dims)>>;

template <typename Type, DimensionOrder Order, Dim_size_t... Dims>
using Matrix = MatrixType<Type, Order, std::array, sizeof...(Dims), {Dims...}, std::make_index_sequence<sizeof...(Dims)>>;

template <IsMatrixType BaseMatrixType>
using MaterializedMatrix = MatrixType<typename std::remove_cvref_t<BaseMatrixType>::value_type,
                                      std::remove_cvref_t<BaseMatrixType>::order,
                                      std::array,
                                      std::remove_cvref_t<BaseMatrixType>::number_of_dimensions,
                                      std::remove_cvref_t<BaseMatrixType>::dimensions,
                                      std::make_index_sequence<std::remove_cvref_t<BaseMatrixType>::number_of_dimensions>>;

template <DimensionOrder LocalOrder, IsMatrixType BaseMatrixType>
using PermutedMatrix = PermutedMatrixType<BaseMatrixType, LocalOrder, std::make_index_sequence<std::remove_cvref_t<BaseMatrixType>::number_of_dimensions>>;

template <std::size_t ConcatinationDimension, IsMatrixType... BaseMatrixType>
using ConcatenatedMatrix =
        ConcatenatedMatrixType<ConcatinationDimension, std::make_index_sequence<std::tuple_element_t<0, std::tuple<std::remove_cvref_t<BaseMatrixType>...>>::number_of_dimensions>, BaseMatrixType...>;

template <IsMatrixType BaseMatrixType, DimensionOrder SlicedOrder, std::array<Dim_size_t, SlicedOrder.length()> Slices>
using SlicedMatrix = SlicedMatrixType<BaseMatrixType,
                                      SlicedOrder,
                                      Slices,
                                      std::make_index_sequence<std::remove_cvref_t<BaseMatrixType>::number_of_dimensions>,
                                      std::make_index_sequence<SlicedOrder.length()>,
                                      std::make_index_sequence<std::remove_cvref_t<BaseMatrixType>::number_of_dimensions - SlicedOrder.length()>>;

template <IsMatrixType BaseMatrixType, DimensionOrder AddedOrder, std::array<Dim_size_t, AddedOrder.length()> Lengths>
using BroadcastedMatrix = BroadcastedMatrixType<BaseMatrixType,
                                                AddedOrder,
                                                Lengths,
                                                std::make_index_sequence<std::remove_cvref_t<BaseMatrixType>::number_of_dimensions + AddedOrder.length()>,
                                                std::make_index_sequence<AddedOrder.length()>,
                                                std::make_index_sequence<std::remove_cvref_t<BaseMatrixType>::number_of_dimensions>>;

template <IsMatrixType BaseMatrixType, DimensionOrder ReplicatOrder, std::array<Dim_size_t, ReplicatOrder.length()> Lengths>
using ReplicatedMatrix = ReplicatedMatrixType<BaseMatrixType,
                                              ReplicatOrder,
                                              Lengths,
                                              std::make_index_sequence<std::remove_cvref_t<BaseMatrixType>::number_of_dimensions>,
                                              std::make_index_sequence<ReplicatOrder.length()>,
                                              std::make_index_sequence<std::remove_cvref_t<BaseMatrixType>::number_of_dimensions - ReplicatOrder.length()>>;

template <IsMatrixType BaseMatrixType, DimensionOrder ReplaceFrom, DimensionOrder ReplaceTo>
using ReplacedMatrix = ReplacedMatrixType<BaseMatrixType, ReplaceFrom, ReplaceTo, std::make_index_sequence<std::remove_cvref_t<BaseMatrixType>::number_of_dimensions>>;

template <IsMatrixType BaseMatrixType>
using NegativeMatrix = NegativeMatrixType<BaseMatrixType, std::make_index_sequence<std::remove_cvref_t<BaseMatrixType>::number_of_dimensions>>;

template <IsMatrixType BaseMatrixType, DimensionOrder Old, DimensionOrder New, Dim_size_t... NewDimensions>
using SplitMatrix = SplitMatrixType<BaseMatrixType,
                                    Old,
                                    New,
                                    std::array<Dim_size_t, sizeof...(NewDimensions)>{NewDimensions...},
                                    std::make_index_sequence<std::remove_cvref_t<BaseMatrixType>::number_of_dimensions + sizeof...(NewDimensions) - 1>,
                                    std::make_index_sequence<New.length()>,
                                    std::make_index_sequence<std::remove_cvref_t<BaseMatrixType>::number_of_dimensions>,
                                    std::make_index_sequence<std::remove_cvref_t<BaseMatrixType>::number_of_dimensions - 1>>;

template <IsMatrixType BaseMatrixType, DimensionOrder Old, DimensionOrder New>
using CollapsedMatrix = CollapsedMatrixType<BaseMatrixType,
                                             Old,
                                             New,
                                             std::make_index_sequence<std::remove_cvref_t<BaseMatrixType>::number_of_dimensions - Old.length() + 1>,
                                             std::make_index_sequence<std::remove_cvref_t<BaseMatrixType>::number_of_dimensions - Old.length()>,
                                             std::make_index_sequence<std::remove_cvref_t<BaseMatrixType>::number_of_dimensions>,
                                             std::make_index_sequence<Old.length()>>;

template <IsMatrixType BaseMatrixType>
using ReferencedMatrix = ReferencedMatrixType<BaseMatrixType>;

template <IsMatrixType... BaseMatrixType>
using FusedMatrix = FusedMatrixType<std::make_index_sequence<sizeof...(BaseMatrixType)>, BaseMatrixType...>;

template <std::size_t Is, IsFuzedMatrixType BaseMatrixType>
using SelectFusedMatrix = SelectFusedMatrixType<Is, BaseMatrixType>;

template <IsMatrixType BaseMatrixType, DimensionOrder Order, Dim_size_t Size>
    requires(Order.length() == 1 && Size > 0)
using OverrideDimensionMatrix =
        MaterializedMatrix<ReplicatedMatrix<SlicedMatrix<BaseMatrixType, Order, {1}>, Order, {Size}>>; // Override the one dimension of a matrix with a value, useful only for Type information

template <IsMatrixType BaseMatrixType, typename OverrideType>
using OverrideTypeMatrix = MatrixType<OverrideType,
                                      std::remove_cvref_t<BaseMatrixType>::order,
                                      std::array,
                                      std::remove_cvref_t<BaseMatrixType>::number_of_dimensions,
                                      std::remove_cvref_t<BaseMatrixType>::dimensions,
                                      std::make_index_sequence<std::remove_cvref_t<BaseMatrixType>::number_of_dimensions>>;

template <IsMatrixType BaseMatrixType, DimensionOrder RemoveOrder>
struct OverrideRemoveDimensionHelper {
    static constexpr DimensionOrder order          = BaseMatrixType::order.remove(RemoveOrder);
    static constexpr std::size_t    removed_length = RemoveOrder.length();
    static_assert(order.length() == BaseMatrixType::order.length() - RemoveOrder.length(), "Remove order must be a subset of the base matrix order");
    static_assert(order.length() > 0, "Remove order must not remove all dimensions from the base matrix order");
    static constexpr DimensionOrder                         first_dim        = DimensionOrder(order.order[0]); // The first dimension in the new order is the last dimension in the old order
    static constexpr DimensionOrder                         Fused_order      = RemoveOrder + first_dim;
    static constexpr std::array<Dim_size_t, removed_length> dimension_slices = makeFilledArray<Dim_size_t, removed_length>(1); // The removed dimensions are all 1, since we are removing the dimension

    using LocalSlicedMatrix = SlicedMatrixType<BaseMatrixType,
                                               RemoveOrder,
                                               dimension_slices,
                                               std::make_index_sequence<std::remove_cvref_t<BaseMatrixType>::number_of_dimensions>,
                                               std::make_index_sequence<removed_length>,
                                               std::make_index_sequence<std::remove_cvref_t<BaseMatrixType>::number_of_dimensions - removed_length>>;

    using LocalCollapsedMatrix    = CollapsedMatrix<LocalSlicedMatrix, Fused_order, first_dim>;
    using LocalPermutedMatrix     = PermutedMatrix<order, LocalCollapsedMatrix>;
    using LocalMaterializedMatrix = MaterializedMatrix<LocalPermutedMatrix>;
};

// template <IsMatrixType BaseMatrixType, DimensionOrder RemoveOrder>
// using OverrideRemoveDimensionMatrix = MaterializedMatrix<
//     /*BaseMatrixType=*/PermutedMatrix<
//         /*LocalOrder=*/BaseMatrixType::order.remove(RemoveOrder),
//         /*BaseMatrixType=*/CollapsedMatrix<
//             /*BaseMatrixType=*/SlicedMatrix<
//                 /*BaseMatrixType=*/BaseMatrixType,
//                 /*SlicedOrder=*/RemoveOrder,
//                 /*Slices=*/makeFilledArray<std::size_t, RemoveOrder.length()>(1)>,
//             /*Old=*/RemoveOrder + DimensionOrder(BaseMatrixType::order.remove(RemoveOrder).order[0]),
//             /*New=*/DimensionOrder(BaseMatrixType::order.remove(RemoveOrder).order[0])>
//         >
//     >;
template <IsMatrixType BaseMatrixType, DimensionOrder RemoveOrder>
using OverrideRemoveDimensionMatrix = OverrideRemoveDimensionHelper<BaseMatrixType, RemoveOrder>::LocalMaterializedMatrix;

template <typename CmpMatrixType, typename BaseMatrixType>
concept IsPermutationalSame = (IsMatrixType<CmpMatrixType> && IsMatrixType<BaseMatrixType> &&                                               // Both are matrix types
                               std::remove_cvref_t<BaseMatrixType>::order.length() == std::remove_cvref_t<CmpMatrixType>::order.length() && // Same length
                               std::remove_cvref_t<BaseMatrixType>::order.containsAll(std::remove_cvref_t<CmpMatrixType>::order)            // Same named dimensions
                               //    PermutedMatrix<std::remove_cvref_t<BaseMatrixType>::order, std::remove_cvref_t<CmpMatrixType>>::dimensions == std::remove_cvref_t<BaseMatrixType>::dimensions //
                               //    Same dimensions // doesnt work
);

template <typename T>
concept IsBaseMatrixType = std::is_same_v<std::remove_cvref_t<T>, MaterializedMatrix<std::remove_cvref_t<T>>>;

template <DimensionOrder LocalOrder, IsMatrixType BaseMatrixType>
constexpr conditional_const<std::is_const_v<std::remove_reference_t<BaseMatrixType>>, PermutedMatrix<LocalOrder, BaseMatrixType &&>> permute(BaseMatrixType &&matrix) {
    return PermutedMatrix<LocalOrder, BaseMatrixType &&>(std::forward<BaseMatrixType>(matrix));
}

template <std::size_t ConcatinationDimension, IsMatrixType... MatrixTypes>
constexpr conditional_const<(std::is_const_v<std::remove_reference_t<MatrixTypes>> || ...), ConcatenatedMatrix<ConcatinationDimension, MatrixTypes &&...>> concatenate(MatrixTypes &&...matrices) {
    return ConcatenatedMatrix<ConcatinationDimension, MatrixTypes &&...>(std::forward<MatrixTypes>(matrices)...);
}

template <DimensionOrder SlicedDimensions, Dim_size_t... Slices, IsMatrixType BaseMatrixType>
constexpr conditional_const<std::is_const_v<std::remove_reference_t<BaseMatrixType>>, SlicedMatrix<BaseMatrixType &&, SlicedDimensions, std::array{Slices...}>> slice(
        BaseMatrixType &&matrices, const std::array<Dim_size_t, sizeof...(Slices)> &offsets = makeFilledArray<Dim_size_t, sizeof...(Slices)>(0)) {
    return SlicedMatrix<BaseMatrixType &&, SlicedDimensions, std::array{Slices...}>(std::forward<BaseMatrixType>(matrices), offsets);
}

template <DimensionOrder AddedOrder, std::array<Dim_size_t, AddedOrder.length()> Lengths = makeFilledArray<Dim_size_t, AddedOrder.length()>(1), IsMatrixType BaseMatrixType>
constexpr conditional_const<std::is_const_v<std::remove_reference_t<BaseMatrixType>>, BroadcastedMatrix<BaseMatrixType &&, AddedOrder, Lengths>> broadcast(BaseMatrixType &&matrix) {
    return BroadcastedMatrix<BaseMatrixType &&, AddedOrder, Lengths>(std::forward<BaseMatrixType>(matrix));
}

template <DimensionOrder AddedOrder, std::array<Dim_size_t, AddedOrder.length()> Lengths = makeFilledArray<Dim_size_t, AddedOrder.length()>(1), IsMatrixType BaseMatrixType>
    requires(!std::remove_cvref_t<BaseMatrixType>::order.containsAny(AddedOrder))
constexpr conditional_const<std::is_const_v<std::remove_reference_t<BaseMatrixType>>, BroadcastedMatrix<BaseMatrixType &&, AddedOrder, Lengths>> conditionalBroadcast(BaseMatrixType &&matrix) {
    return BroadcastedMatrix<BaseMatrixType &&, AddedOrder, Lengths>(std::forward<BaseMatrixType>(matrix));
}

template <DimensionOrder AddedOrder, std::array<Dim_size_t, AddedOrder.length()> Lengths = makeFilledArray<Dim_size_t, AddedOrder.length()>(1), IsMatrixType BaseMatrixType>
    requires(std::remove_cvref_t<BaseMatrixType>::order.containsAny(AddedOrder))
constexpr conditional_const<std::is_const_v<std::remove_reference_t<BaseMatrixType>>, ReferencedMatrix<BaseMatrixType &&>> conditionalBroadcast(BaseMatrixType &&matrix) {
    return ReferencedMatrix<BaseMatrixType &&>(std::forward<BaseMatrixType>(matrix)); // No expansion needed, return the original matrix wrapped in a ReferencedMatrix
}

template <DimensionOrder ReplicatOrder, std::array<Dim_size_t, ReplicatOrder.length()> Lengths = makeFilledArray<Dim_size_t, ReplicatOrder.length()>(1), IsMatrixType BaseMatrixType>
constexpr conditional_const<std::is_const_v<std::remove_reference_t<BaseMatrixType>>, ReplicatedMatrix<BaseMatrixType &&, ReplicatOrder, Lengths>> replicate(BaseMatrixType &&matrix) {
    return ReplicatedMatrix<BaseMatrixType &&, ReplicatOrder, Lengths>(std::forward<BaseMatrixType>(matrix));
}

template <DimensionOrder ReplicatOrder, std::array<Dim_size_t, ReplicatOrder.length()> Lengths = makeFilledArray<Dim_size_t, ReplicatOrder.length()>(1), IsMatrixType BaseMatrixType>
    requires(std::remove_cvref_t<BaseMatrixType>::order.containsAll(ReplicatOrder))
constexpr conditional_const<std::is_const_v<std::remove_reference_t<BaseMatrixType>>, ReplicatedMatrix<BaseMatrixType &&, ReplicatOrder, Lengths>> conditionalReplicate(BaseMatrixType &&matrix) {
    return ReplicatedMatrix<BaseMatrixType &&, ReplicatOrder, Lengths>(std::forward<BaseMatrixType>(matrix));
}

template <DimensionOrder ReplicatOrder, std::array<Dim_size_t, ReplicatOrder.length()> Lengths = makeFilledArray<Dim_size_t, ReplicatOrder.length()>(1), IsMatrixType BaseMatrixType>
    requires(!std::remove_cvref_t<BaseMatrixType>::order.containsAny(ReplicatOrder))
constexpr conditional_const<std::is_const_v<std::remove_reference_t<BaseMatrixType>>, ReferencedMatrix<BaseMatrixType &&>> conditionalReplicate(BaseMatrixType &&matrix) {
    return ReferencedMatrix<BaseMatrixType &&>(
            std::forward<BaseMatrixType>(matrix)); // No expansion needed as it does not contain the dimesnion, return the original matrix wrapped in a ReferencedMatrix
}

template <DimensionOrder ReplicatOrder, std::array<Dim_size_t, ReplicatOrder.length()> Lengths = makeFilledArray<Dim_size_t, ReplicatOrder.length()>(1), IsMatrixType BaseMatrixType>
    requires(!std::remove_cvref_t<BaseMatrixType>::order.containsAll(ReplicatOrder) && std::remove_cvref_t<BaseMatrixType>::order.containsAny(ReplicatOrder))
constexpr void conditionalReplicate(BaseMatrixType &&) {
    static_assert(std::remove_cvref_t<BaseMatrixType>::order.containsAll(ReplicatOrder) || !std::remove_cvref_t<BaseMatrixType>::order.containsAny(ReplicatOrder),
                  "partial replication is not supported, you must split it up");
}

template <DimensionOrder ReplaceFrom, DimensionOrder ReplaceTo, IsMatrixType BaseMatrixType>
constexpr conditional_const<std::is_const_v<std::remove_reference_t<BaseMatrixType>>, ReplacedMatrix<BaseMatrixType &&, ReplaceFrom, ReplaceTo>> replace(BaseMatrixType &&matrix) {
    return ReplacedMatrix<BaseMatrixType &&, ReplaceFrom, ReplaceTo>(std::forward<BaseMatrixType>(matrix));
}

template <DimensionOrder ReplaceFrom, DimensionOrder ReplaceTo, IsMatrixType BaseMatrixType>
    requires(std::remove_cvref_t<BaseMatrixType>::order.containsAll(ReplaceTo) && !std::remove_cvref_t<BaseMatrixType>::order.containsAny(ReplaceFrom.remove(ReplaceTo)))
constexpr conditional_const<std::is_const_v<std::remove_reference_t<BaseMatrixType>>, ReferencedMatrix<BaseMatrixType &&>> conditionalReplace(BaseMatrixType &&matrix) {
    return ReferencedMatrix<BaseMatrixType &&>(std::forward<BaseMatrixType>(matrix)); // No expansion needed, return the original matrix  wrapped in a ReferencedMatrix
}

template <DimensionOrder ReplaceFrom, DimensionOrder ReplaceTo, IsMatrixType BaseMatrixType>
    requires(!std::remove_cvref_t<BaseMatrixType>::order.containsAll(ReplaceTo) && std::remove_cvref_t<BaseMatrixType>::order.containsAny(ReplaceFrom.remove(ReplaceTo)))
constexpr auto conditionalReplace(BaseMatrixType &&matrix) {
    return replace<ReplaceFrom, ReplaceTo>(std::forward<BaseMatrixType>(matrix));
}

template <DimensionOrder Old, DimensionOrder New, Dim_size_t... NewDimensions, IsMatrixType BaseMatrixType>
constexpr conditional_const<std::is_const_v<std::remove_reference_t<BaseMatrixType>>, SplitMatrix<BaseMatrixType &&, Old, New, NewDimensions...>> split(BaseMatrixType &&matrix) {
    return SplitMatrix<BaseMatrixType &&, Old, New, NewDimensions...>(std::forward<BaseMatrixType>(matrix));
}

template <DimensionOrder Old, DimensionOrder New, IsMatrixType BaseMatrixType>
constexpr conditional_const<std::is_const_v<std::remove_reference_t<BaseMatrixType>>, CollapsedMatrix<BaseMatrixType &&, Old, New>> collapse(BaseMatrixType &&matrix) {
    return CollapsedMatrix<BaseMatrixType &&, Old, New>(std::forward<BaseMatrixType>(matrix));
}

template <IsMatrixType MatrixType>
constexpr decltype(auto) operator-(MatrixType &&matrix) {
    return NegativeMatrix<MatrixType>(std::forward<MatrixType>(matrix));
}

template <std::size_t _=0, IsMatrixType... MatrixTypes>
constexpr conditional_const<(std::is_const_v<std::remove_reference_t<MatrixTypes>> || ...), FusedMatrixType<std::make_index_sequence<sizeof...(MatrixTypes)>, MatrixTypes &&...>> fuse(MatrixTypes &&...matrices) {
    return FusedMatrixType<std::make_index_sequence<sizeof...(MatrixTypes)>,MatrixTypes &&...>(std::forward<MatrixTypes>(matrices)...);
}

template <std::size_t Is, IsFuzedMatrixType MatrixType>
constexpr conditional_const<std::is_const_v<std::remove_reference_t<MatrixType>>, SelectFusedMatrix<Is, MatrixType &&>> selectFused(MatrixType &&matrix) {
    return SelectFusedMatrix<Is, MatrixType&&>(std::forward<MatrixType>(matrix));
}

static_assert(IsMatrixType<Matrix<float, DimensionOrder("12"), 1, 2>>, "Example Assert, for sanity, Matrix<float, DimensionOrder(\"12\"), 1,2> is not a valid MatrixType");
static_assert(IsMatrixType<PermutedMatrix<DimensionOrder("21"), Matrix<float, DimensionOrder("12"), 1, 2>>>,
              "Example Assert, for sanity, PermutedMatrix<DimensionOrder(\"21\"),Matrix<float, DimensionOrder(\"12\"), 1,2 >> is not a valid MatrixType");

using __TestMatrix = Matrix<float, DimensionOrder("12"), 2, 3>; // Example matrix type for testing
static_assert(IsMatrixType<__TestMatrix>, "Example Assert, for sanity, __TestMatrix is not a valid MatrixType");
static_assert(IsBaseMatrixType<__TestMatrix>, "Example Assert, for sanity, __TestMatrix is not a valid Base Matrix Type");
static_assert(IsMatrixType<PermutedMatrix<"21", __TestMatrix>>, "Example Assert, for sanity, PermutedMatrix<\"21\",__TestMatrix> is not a valid MatrixType");
static_assert(!IsBaseMatrixType<PermutedMatrix<"21", __TestMatrix>>, "Example Assert, for sanity, PermutedMatrix<\"21\",__TestMatrix> is not a valid Base Matrix Type");
static_assert(IsMatrixType<ConcatenatedMatrix<1, __TestMatrix, __TestMatrix>>, "Example Assert, for sanity, ConcatenatedMatrix<1,__TestMatrix, __TestMatrix> is not a valid MatrixType");
static_assert(IsMatrixType<SlicedMatrix<__TestMatrix, "21", {1, 2}>>, "Example Assert, for sanity, SlicedMatrix<__TestMatrix, \"21\", {1,2}> is not a valid MatrixType");
static_assert(IsMatrixType<BroadcastedMatrix<__TestMatrix, "34", {2, 3}>>, "Example Assert, for sanity, BroadcastedMatrix<__TestMatrix, \"34\", {2, 3}> is not a valid MatrixType");
static_assert(IsMatrixType<ReplicatedMatrix<SlicedMatrix<__TestMatrix, "12", {1, 1}>, "12", {4, 2}>>,
              "Example Assert, for sanity, ReplicatedMatrix<__TestMatrix, \"12\", {4, 2}> is not a valid MatrixType");
static_assert(IsMatrixType<ReplacedMatrix<__TestMatrix, "12", "34">>, "Example Assert, for sanity, ReplacedMatrix<__TestMatrix, \"12\", \"34\"> is not a valid MatrixType");
static_assert(IsMatrixType<SplitMatrix<__TestMatrix, "2", "34", 1, 2>>, "Example Assert, for sanity, SplitMatrix<__TestMatrix, \"2\", \"34\", 1, 2> is not a valid MatrixType");
static_assert(IsMatrixType<CollapsedMatrix<__TestMatrix, "12", "Q">>, "Example Assert, for sanity, CollapsedMatrix<__TestMatrix, \"12\", \"Q\"> is not a valid MatrixType");
static_assert(IsMatrixType<NegativeMatrix<__TestMatrix>>, "Example Assert, for sanity, NegativeMatrix<__TestMatrix> is not a valid MatrixType");
static_assert(IsMatrixType<ReferencedMatrix<__TestMatrix>>, "Example Assert, for sanity, ReferencedMatrix<__TestMatrix> is not a valid MatrixType");
static_assert(IsMatrixType<FusedMatrix<__TestMatrix, __TestMatrix, __TestMatrix>>, "Example Assert, for sanity, FusedMatrix<__TestMatrix,__TestMatrix,__TestMatrix> is not a valid MatrixType");
static_assert(IsMatrixType<MaterializedMatrix<__TestMatrix>>, "Example Assert, for sanity, MaterializedMatrix<__TestMatrix> is not a valid MatrixType");
static_assert(IsBaseMatrixType<MaterializedMatrix<__TestMatrix>>, "Example Assert, for sanity, MaterializedMatrix<__TestMatrix> is not a valid MatrixType");
static_assert(IsMatrixType<OverrideDimensionMatrix<__TestMatrix, "1", 5>>, "Example Assert, for sanity, OverrideDimensionMatrix<\"1\", __TestMatrix, 5> is not a valid MatrixType");

/****************************************************************************************************************************************************************************
                                                              Example Uses
*****************************************************************************************************************************************************************************

Matrix<float, DimensionOrder("12"), 1, 2> A;
Matrix<float, "12" , 1, 2> B; // same as above, but with string literal

auto C = permute<"21">(A); // Permute the matrix A to have the dimensions in the order "21"
auto D = concatenate<0>(A, B); // Concatenate the matrices A and B along dimension 0 which is the first dimension TODO: change to named dimensions
auto E = slice<"1",1>(A, {0}); // Slice the matrix A along dimension "1" of size 1 with an offset of 0
auto F = broadcast<"34", {2, 3}>(A); // Broadcast the matrix A to have dimensions "1234" with new dimension lengths 2 and 3
auto G = replicate<"1", {4}>(A); // Replicate the matrix A to have dimensions "12" with dimension lengths 4, 2
auto H = replace<"12", "34">(A); // Replace the dimensions "12" of matrix A with "34"
auto I = split<"2", "34", 1, 2>(A); // Split the dimension "2" of matrix A into two dimensions "3" and "4" with lengths 1 and 2 product of the split dimensions must match the original dimension
auto J = collapse<"12", "Q">(A); // Collapse the dimensions "12" of matrix A into a single dimension "Q"
// Negative of a matrix
auto K = -A; // Negate the matrix A, this will return a NegativeMatrix
auto L = fuse(A, B); // Fuse multiple matrices into a single matrix, the resulting matrix will have a tuple of values at each index


ReferencedMatrixType<BaseMatrixType>    //Doesn't do anything, just used for conditional type changes

// overrides only usable for type information, not for data

OverrideDimensionMatrix<Matrix, Order, Size>        // Changes the Dimension at Order position to new size
OverrideTypeMatrix<Matrix, OverrideType>            // Changes the value_type of the matrix to OverrideType
OverrideRemoveDimensionMatrix<Matrix, RemoveOrder>  // Removes the dimension RemoveOrder from the matrix, if it exists, and returns a new matrix with the remaining dimensions
****************************************************************************************************************************************************************************
                                                                 Example Uses End
****************************************************************************************************************************************************************************/
