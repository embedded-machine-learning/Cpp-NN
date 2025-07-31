#pragma once
#include "./include/NeuralNetwork.hpp"
#include "./weights.inc"
#include "./weights_unrolled.inc"

constexpr std::size_t SUB_BATCH = 1;
// const auto __attribute__(( section(".data") )) network=layers::Sequence(
constexpr auto network=layers::Sequence(
    layers::SSMPiano<float,Complex<float>,SUB_BATCH,SUB_BATCH>(A0, B0_1_48, B0_bias, C0_1_96, C0_bias, FastTanh<float>),
    layers::Sedge<float,Complex<float>,SUB_BATCH,SUB_BATCH>(A1, B1_1_48, B1_bias, C1_1_96, C1_bias, SkipLayer1_weights_1_112, FastTanh<float>),
    layers::Sedge<float,Complex<float>,SUB_BATCH,SUB_BATCH>(A2, B2_1_48, B2_bias, C2_1_96, C2_bias, SkipLayer2_weights_1_112, FastTanh<float>),
    layers::Sedge<float,Complex<float>,SUB_BATCH,SUB_BATCH>(A3, B3_1_48, B3_bias, C3_1_96, C3_bias, SkipLayer3_weights_1_112, FastTanh<float>),
    layers::Linear<float,2*SUB_BATCH,"BS">(Decoder_weights_1_112, Decoder_bias, PassThrough<float>)
);

