#pragma once
#include "./weights.hpp"
#include "./weights_unrolled.hpp"
#include "./NeuralNetwork.hpp"

constexpr std::size_t SUB_BATCH = 4;
// const auto __attribute__(( section(".data") )) network=layers::Sequence(
constexpr auto network=layers::Sequence(
    layers::SSMPiano<float,Complex<float>,SUB_BATCH,SUB_BATCH>(A0, B0_1_12, B0_bias, C0_1_24, C0_bias, FastTanh<float>),
    layers::Sedge<float,Complex<float>,SUB_BATCH,SUB_BATCH>(A1, B1_1_12, B1_bias, C1_1_24, C1_bias, SkipLayer1_weights_1_24, FastTanh<float>),
    layers::Sedge<float,Complex<float>,SUB_BATCH,SUB_BATCH>(A2, B2_1_12, B2_bias, C2_1_24, C2_bias, SkipLayer2_weights_1_24, FastTanh<float>),
    layers::Sedge<float,Complex<float>,SUB_BATCH,SUB_BATCH>(A3, B3_1_12, B3_bias, C3_1_24, C3_bias, SkipLayer3_weights_1_24, FastTanh<float>),
    layers::Linear<float,8,"BS">(Decoder_weights_1_1, Decoder_bias, PassThrough<float>)
);

