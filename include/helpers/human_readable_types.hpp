#pragma once
#include <complex>
#include <cstdint>
#include <type_traits>

#include "../Matrix.hpp"
#include "../MatrixOperations.hpp"
#include "../types/Complex.hpp"
#include "./constexpressions.hpp"

#define HumanReadableTypeArrayDEF(Typename)                                                                                                                                                            \
    template <>                                                                                                                                                                                        \
    constexpr auto human_readable_type<Typename> = toArrayAuto(#Typename);

template <typename Type>
constexpr auto human_readable_type = toArrayAuto("Unknown");
template <typename Type>
requires(std::is_lvalue_reference_v<Type&>)
constexpr auto human_readable_type<Type &> = concat(human_readable_type<Type>, toArrayAuto("&"));
template <typename Type>
constexpr auto human_readable_type<const Type> = concat(toArrayAuto("const "), human_readable_type<Type>);
template <typename Type>
constexpr auto human_readable_type<Type *> = concat(human_readable_type<Type>, toArrayAuto("*"));
template <typename Type>
requires(std::is_rvalue_reference_v<Type&&>)
constexpr auto human_readable_type<Type &&> = concat(human_readable_type<Type>, toArrayAuto("&&"));
template <typename... Types>
constexpr auto human_readable_type<std::tuple<Types...>> = concat(toArrayAuto("std::tuple<"), concat(human_readable_type<Types>, toArrayAuto(", "))..., toArrayAuto("\b\b>"));
template <std::size_t Align, typename... Types>
constexpr auto human_readable_type<AlignedMatrixCollection<Align,Types...>> = concat(toArrayAuto("AlignedMatrixCollection<"),num_to_string<Align>,toArrayAuto(", "), concat(human_readable_type<Types>, toArrayAuto(", "))..., toArrayAuto("\b\b>"));
template <typename Type>
constexpr auto human_readable_type<Complex<Type>> = concat(toArrayAuto("Complex<"), human_readable_type<Type>, toArrayAuto(">"));

template <>
constexpr auto human_readable_type<DimensionOrder> = toArrayAuto("DimensionOrder");

HumanReadableTypeArrayDEF(void);
HumanReadableTypeArrayDEF(bool);
HumanReadableTypeArrayDEF(char);
HumanReadableTypeArrayDEF(int8_t);
HumanReadableTypeArrayDEF(int16_t);
HumanReadableTypeArrayDEF(int32_t);
HumanReadableTypeArrayDEF(int64_t);
HumanReadableTypeArrayDEF(uint8_t);
HumanReadableTypeArrayDEF(uint16_t);
HumanReadableTypeArrayDEF(uint32_t);
HumanReadableTypeArrayDEF(uint64_t);
HumanReadableTypeArrayDEF(float);
HumanReadableTypeArrayDEF(double);

template <typename Type,
          DimensionOrder LocalOrder /* = DimensionOrder("") */,
          template <typename, std::size_t>
          class ContainerType /*= std::array */,
          std::size_t                                NumberOfDimensions /*= 0 */, //
          std::array<Dim_size_t, NumberOfDimensions> Dims /*= std::array<Dim_size_t, NumberOfDimensions>{}*/,
          std::size_t... Indices>
constexpr auto human_readable_type<MatrixType<Type, LocalOrder, ContainerType, NumberOfDimensions, Dims, std::index_sequence<Indices...>>> =
        concat(toArrayAuto("Matrix<"),
               human_readable_type<Type>,
               toArrayAuto(", \""),
               LocalOrder.order,
               toArrayAuto("\", "),
               concat(num_to_string<Dims[Indices]>, toArrayAuto(", "))...,
               toArrayAuto("\b\b>"));

template <IsMatrixType BaseMatrixType, DimensionOrder LocalOrder /*= DimensionOrder("")*/>
constexpr auto human_readable_type<PermutedMatrixType<BaseMatrixType, LocalOrder, std::make_index_sequence<std::remove_cvref_t<BaseMatrixType>::number_of_dimensions>>> =
        concat(toArrayAuto("PermutedMatrix<\""), LocalOrder.order, toArrayAuto("\", "), human_readable_type<BaseMatrixType>, toArrayAuto(">"));

template <std::size_t ConcatenatedMatrixDimension /*= 0*/, //
          typename VariadicIndcices               /*= std::index_sequence<>*/,
          IsMatrixType... MatrixTypes>
constexpr auto human_readable_type<ConcatinadedMatrixType<ConcatenatedMatrixDimension, VariadicIndcices, MatrixTypes...>> =
        concat(toArrayAuto("ConcatenatedMatrix<"), num_to_string<ConcatenatedMatrixDimension>, toArrayAuto(", "), concat(human_readable_type<MatrixTypes>, toArrayAuto(", "))..., toArrayAuto("\b\b>"));

template <IsMatrixType                                 BaseMatrixType,
          DimensionOrder                               SlicedOrder,
          std::array<Dim_size_t, SlicedOrder.length()> Slices,
          typename VariadicIndices,
          typename VariadicSliceIndices,
          typename VariadicReducedIndices>
constexpr auto human_readable_type<SlicedMatrixType<BaseMatrixType, SlicedOrder, Slices, VariadicIndices, VariadicSliceIndices, VariadicReducedIndices>> =
        concat(toArrayAuto("SlicedMatrix<"),
               human_readable_type<BaseMatrixType>,
               toArrayAuto(", \""),
               SlicedOrder.order,
               toArrayAuto("\", "),
               array_to_string<SlicedOrder.length(), Slices>,
               toArrayAuto(">"));

template <IsMatrixType                                BaseMatrixType,
          DimensionOrder                              AddedOrder,
          std::array<Dim_size_t, AddedOrder.length()> Lengths,
          typename VariadicIndices,
          typename VariadicNewIndices,
          typename VariadicBaseIndices>
constexpr auto human_readable_type<BroadcastedMatrixType<BaseMatrixType, AddedOrder, Lengths, VariadicIndices, VariadicNewIndices, VariadicBaseIndices>> =
        concat(toArrayAuto("BroadcastedMatrix<"),
               human_readable_type<BaseMatrixType>,
               toArrayAuto(", \""),
               AddedOrder.order,
               toArrayAuto("\", "),
               array_to_string<AddedOrder.length(), Lengths>,
               toArrayAuto(">"));

template <IsMatrixType                                   BaseMatrixType,
          DimensionOrder                                 ReplicatOrder,
          std::array<Dim_size_t, ReplicatOrder.length()> Lengths,
          typename VariadicIndices,
          typename VariadicNewIndices,
          typename VariadicBaseIndices>
constexpr auto human_readable_type<ReplicatedMatrixType<BaseMatrixType, ReplicatOrder, Lengths, VariadicIndices, VariadicNewIndices, VariadicBaseIndices>> =
        concat(toArrayAuto("ReplicatedMatrix<"),
               human_readable_type<BaseMatrixType>,
               toArrayAuto(", \""),
               ReplicatOrder.order,
               toArrayAuto("\", "),
               array_to_string<ReplicatOrder.length(), Lengths>,
               toArrayAuto(">"));

template <IsMatrixType BaseMatrixType, DimensionOrder ReplaceFrom, DimensionOrder ReplaceTo, typename VariadicIndices>
constexpr auto human_readable_type<ReplacedMatrixType<BaseMatrixType, ReplaceFrom, ReplaceTo, VariadicIndices>> =
        concat(toArrayAuto("ReplacedMatrix<"), human_readable_type<BaseMatrixType>, toArrayAuto(", \""), ReplaceFrom.order, toArrayAuto("\", \""), ReplaceTo.order, toArrayAuto("\">"));

template <IsMatrixType BaseMatrixType, typename VariadicIndices>
constexpr auto human_readable_type<NegativeMatrixType<BaseMatrixType, VariadicIndices>> = concat(toArrayAuto("NegativeMatrix<"), human_readable_type<BaseMatrixType>, toArrayAuto(">"));

template <IsMatrixType   BaseMatrixType,
          DimensionOrder Old,
          DimensionOrder New,
          std::array     NewDimensions,
          typename VariadicIndices,
          typename VariadicNewIndices,
          typename VariadicOldIndices,
          typename VariadicReducedIndices>
constexpr auto human_readable_type<SplitMatrixType<BaseMatrixType, Old, New, NewDimensions, VariadicIndices, VariadicNewIndices, VariadicOldIndices, VariadicReducedIndices>> =
        concat(toArrayAuto("SplitMatrix<"),
               human_readable_type<BaseMatrixType>,
               toArrayAuto(", \""),
               Old.order,
               toArrayAuto("\", \""),
               New.order,
               toArrayAuto("\", {"),
               array_to_string<New.length(), NewDimensions>,
               toArrayAuto("}>"));

template <IsMatrixType BaseMatrixType, DimensionOrder Old, DimensionOrder New, typename VariadicIndices, typename VariadicIndicesM1, typename VariadicOriginalIndices, typename VariadicOldIndices>
constexpr auto human_readable_type<CollaplsedMatrixType<BaseMatrixType, Old, New, VariadicIndices, VariadicIndicesM1, VariadicOriginalIndices, VariadicOldIndices>> =
        concat(toArrayAuto("CollapsedMatrix<"), human_readable_type<BaseMatrixType>, toArrayAuto(", \""), Old.order, toArrayAuto("\", \""), New.order, toArrayAuto("\">"));

template <IsMatrixType BaseMatrixType>
constexpr auto human_readable_type<ReferencedMatrixType<BaseMatrixType>> = concat(toArrayAuto("ReferencedMatrix<"), human_readable_type<BaseMatrixType>, toArrayAuto(">"));

template<IsMatrixType... BaseMatrixTypes>
constexpr auto human_readable_type<FusedMatrix<BaseMatrixTypes...>> =
        concat(toArrayAuto("FusedMatrix<"), concat(human_readable_type<BaseMatrixTypes>, toArrayAuto(", "))..., toArrayAuto("\b\b>"));

template<std::size_t Is, IsFuzedMatrixType BaseFuzedMatrixType>
constexpr auto human_readable_type<SelectFusedMatrixType<Is, BaseFuzedMatrixType>> =
        concat(toArrayAuto("SelectFusedMatrix<"), num_to_string<Is>, toArrayAuto(", "), human_readable_type<BaseFuzedMatrixType>, toArrayAuto(">"));