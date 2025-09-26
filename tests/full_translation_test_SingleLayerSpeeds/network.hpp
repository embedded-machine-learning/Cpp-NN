#pragma once
#include "./include/NeuralNetwork.hpp"
#include "./weights.inc"
#include "./weights_unrolled.inc"

constexpr std::size_t SUB_BATCH = 1;
// const auto __attribute__(( section(".data") )) network=layers::Sequence(
constexpr auto network=layers::Sequence(
    layers::Sedge<float,Complex<float>>(A0, B0, B0_bias, C0, C0_bias, SkipLayer0_weights, FastTanh<float>)
);

