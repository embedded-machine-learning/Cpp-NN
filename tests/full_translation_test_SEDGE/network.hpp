#pragma once
#include "include/NeuralNetwork.hpp"
#include "weights.inc"
#include "weights_unrolled.inc"

// const auto __attribute__(( section(".data") )) network=layers::Sequence(
const auto network=layers::Sequence(
    layers::Sedge<float,Complex<float>>(A0, B0_1_48, B0_bias, C0_1_96, C0_bias, SkipLayer0_1_112, LeakyReLU<float>),
    layers::Sedge<float,Complex<float>>(A1, B1_1_48, B1_bias, C1_1_96, C1_bias, SkipLayer1_1_112, LeakyReLU<float>),
    layers::Sedge<float,Complex<float>>(A2, B2_1_48, B2_bias, C2_1_96, C2_bias, SkipLayer2_1_112, LeakyReLU<float>),
    layers::Sedge<float,Complex<float>>(A3, B3_1_48, B3_bias, C3_1_96, C3_bias, SkipLayer3_1_112, LeakyReLU<float>),
    layers::Sedge<float,Complex<float>>(A4, B4_1_48, B4_bias, C4_1_96, C4_bias, SkipLayer4_1_112, LeakyReLU<float>),
    layers::Sedge<float,Complex<float>>(A5, B5_1_48, B5_bias, C5_1_96, C5_bias, SkipLayer5_1_112, LeakyReLU<float>),
    layers::SumReduction<"S">(),
    layers::Linear<float>(W_1_112, b, PassThrough<float>)
);

