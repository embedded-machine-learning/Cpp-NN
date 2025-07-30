#pragma once
#include "weights.hpp"
#include "./include/NeuralNetwork.hpp"

// const auto __attribute__(( section(".data") )) Seq=layers::Sequential(
const auto Seq=layers::Sequential(
    layers::S5(A0, B0, C0, B0_bias, C0_bias, LeakyReLU),
    layers::S5(A1, B1, C1, B1_bias, C1_bias, LeakyReLU),
    layers::S5(A2, B2, C2, B2_bias, C2_bias, LeakyReLU),
    layers::S5(A3, B3, C3, B3_bias, C3_bias, LeakyReLU),
    layers::S5(A4, B4, C4, B4_bias, C4_bias, LeakyReLU),
    layers::S5(A5, B5, C5, B5_bias, C5_bias, LeakyReLU)
);

// const auto __attribute__(( section(".data") )) Decoder=layers::Sequential(
const auto Decoder=layers::Sequential(
    layers::Linear<float>(W, b, Passthrough)
);

