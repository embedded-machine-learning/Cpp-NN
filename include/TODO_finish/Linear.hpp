#pragma once
#include "./Matrix.hpp"

/*
Simple Linear Layer, uses stack memory
*/
template <Dim_size_t M1_1, Dim_size_t M1_2, Dim_size_t M2_1, typename Lambda, typename InputType = float, typename AccumulationType = float, typename OutputType = float, typename... ActivationInformation>
__attribute__((always_inline)) inline Matrix<OutputType, M1_1, M2_1> Linear(const Matrix<InputType, M1_1, M1_2>  &Input,
                                                                            const Matrix<InputType, M2_1, M1_2>  &Weights,
                                                                            const Matrix<AccumulationType, M2_1> &Bias,
                                                                            Lambda                                &Act,
                                                                            const Matrix<ActivationInformation, M2_1> &...ActivationParameters) {
    
    Matrix<OutputType, M1_1, M2_1> out;
    for (Dim_size_t i1_1 = 0; i1_1 < M1_1; i1_1++) {
        for (Dim_size_t i2_1 = 0; i2_1 < M2_1; i2_1++) {
            AccumulationType sum{Bias.data[i2_1]};
            for (Dim_size_t i1_2 = 0; i1_2 < M1_2; i1_2 += 1) {
                sum += static_cast<AccumulationType>(Input.data[i1_1][i1_2]) * static_cast<AccumulationType>(Weights.data[i2_1][i1_2]);
            }
            out.data[i1_1][i2_1] = Act(sum, ActivationParameters[i2_1]...);
        }
    }
    return out;
}

/*
Simple Linear Layer, uses mostly passed memory
*/
template <Dim_size_t M1_1, Dim_size_t M1_2, Dim_size_t M2_1, typename Lambda, typename InputType = float, typename AccumulationType = float, typename OutputType = float, typename... ActivationInformation>
__attribute__((always_inline)) inline void Linear(const Matrix<InputType, M1_1, M1_2>  &Input,
                                                  Matrix<OutputType, M1_1, M2_1>       &out,
                                                  const Matrix<InputType, M2_1, M1_2>  &Weights,
                                                  const Matrix<AccumulationType, M2_1> &Bias,
                                                  Lambda                                Act,
                                                  const Matrix<ActivationInformation, M2_1> &...ActivationParameters) {
    for (Dim_size_t i1_1 = 0; i1_1 < M1_1; i1_1++) {
        for (Dim_size_t i2_1 = 0; i2_1 < M2_1; i2_1++) {
            AccumulationType sum{Bias.data[i2_1]};
            for (Dim_size_t i1_2 = 0; i1_2 < M1_2; i1_2 += 1) {
                sum += Input.data[i1_1][i1_2] * Weights.data[i2_1][i1_2];
            }
            out.data[i1_1][i2_1] = Act(sum, ActivationParameters[i2_1]...);
        }
    }
}

template <typename Input, typename Weights, typename OutputType>
struct Linear_Generate_out_type_helper;

template <Dim_size_t M1_1, Dim_size_t M1_2, Dim_size_t M2_1, typename InputType, typename OutputType>
struct Linear_Generate_out_type_helper<Matrix<InputType, M1_1, M1_2>, Matrix<InputType, M2_1, M1_2>, OutputType> {
    using type = Matrix<OutputType, M1_1, M2_1>;
};

template <typename Input, typename Weights, typename OutputType>
using Linear_Generate_out_type = typename Linear_Generate_out_type_helper<std::remove_cvref_t<Input>, std::remove_cvref_t<Weights>, OutputType>::type;

template <Dim_size_t M1_1,
          Dim_size_t M1_2,
          Dim_size_t M2_1,
          typename Lambda,
          typename InputType        = float,
          typename AccumulationType = float,
          typename OutputType       = float,
          typename Type_backprob    = int8_t,
          typename... ActivationInformation>
__attribute__((always_inline)) inline std::pair<Matrix<OutputType, M1_1, M2_1>, Matrix<Type_backprob, M1_1, M2_1>> Linear_train(const Matrix<InputType, M1_1, M1_2>  &Input,
                                                                                                                                const Matrix<InputType, M2_1, M1_2>  &Weights,
                                                                                                                                const Matrix<AccumulationType, M2_1> &Bias,
                                                                                                                                Lambda                                Act,
                                                                                                                                const Matrix<ActivationInformation, M2_1> &...ActivationParameters) {
    Matrix<OutputType, M1_1, M2_1>    out;
    Matrix<Type_backprob, M1_1, M2_1> backprob;
    for (Dim_size_t i1_1 = 0; i1_1 < M1_1; i1_1++) {
        for (Dim_size_t i2_1 = 0; i2_1 < M2_1; i2_1++) {
            AccumulationType sum{Bias.data[i2_1]};
            for (Dim_size_t i1_2 = 0; i1_2 < M1_2; i1_2 += 1) {
                sum += Input.data[i1_1][i1_2] * Weights.data[i2_1][i1_2];
            }
            std::tie(out.data[i1_1][i2_1], backprob.data[i1_1][i2_1]) = Act(sum, ActivationParameters[i2_1]...);
        }
    }
    return {out, backprob};
}

template <Dim_size_t M1_1, Dim_size_t M1_2, Dim_size_t M2_2, typename InputType = float, typename AccumulationType = float, typename OutputType = float, typename Type_Act = int8_t>
__attribute__((always_inline)) inline Matrix<InputType, M1_1, M2_2> Linear_back_Input(const Matrix<InputType, M1_1, M1_2> &Output_Grad,
                                                                                      const Matrix<Type_Act, M1_1, M1_2>  &Act_Grad,
                                                                                      const Matrix<InputType, M1_2, M2_2> &Weights) {
    // std::cout << "Linear_back_Input: " << M1_1 << " , " << M2_2 << "\n";
    Matrix<OutputType, M1_1, M2_2> out;
    for (Dim_size_t i1_1 = 0; i1_1 < M1_1; i1_1++) {
        for (Dim_size_t i2_2 = 0; i2_2 < M2_2; i2_2++) {
            AccumulationType sum{static_cast<AccumulationType>(0)};
            for (Dim_size_t i1_2 = 0; i1_2 < M1_2; i1_2++) {
                sum += static_cast<AccumulationType>(Act_Grad.data[i1_1][i1_2]) * static_cast<AccumulationType>(Output_Grad.data[i1_1][i1_2]) * static_cast<AccumulationType>(Weights.data[i1_2][i2_2]);
            }
            out.data[i1_1][i2_2] = static_cast<OutputType>(sum);
        }
    }
    return out;
}

template <Dim_size_t M1_1, Dim_size_t M1_2, Dim_size_t M2_2, typename InputType = float, typename AccumulationType = float, typename OutputType = float, typename Type_Act = int8_t>
__attribute__((always_inline)) inline Matrix<InputType, M1_2, M2_2> Linear_back_Weights(const Matrix<InputType, M1_1, M1_2> &Output_Grad,
                                                                                        const Matrix<Type_Act, M1_1, M1_2>  &Act_Grad,
                                                                                        const Matrix<InputType, M1_1, M2_2> &Input) {
    Matrix<OutputType, M1_2, M2_2> out;
    for (Dim_size_t i1_2 = 0; i1_2 < M1_2; i1_2 += 1) {
        for (Dim_size_t i2_2 = 0; i2_2 < M2_2; i2_2++) {
            AccumulationType sum{static_cast<AccumulationType>(0)};
            for (Dim_size_t i1_1 = 0; i1_1 < M1_1; i1_1++) {
                sum += static_cast<AccumulationType>(Act_Grad.data[i1_1][i1_2]) * static_cast<AccumulationType>(Output_Grad.data[i1_1][i1_2]) * static_cast<AccumulationType>(Input.data[i1_1][i2_2]);
            }
            out.data[i1_2][i2_2] = static_cast<OutputType>(sum);
        }
    }
    return out;
}

template <Dim_size_t M1_1, Dim_size_t M1_2, typename InputType = float, typename AccumulationType = float, typename OutputType = float, typename Type_Act = int8_t>
__attribute__((always_inline)) inline Matrix<OutputType, M1_2> Linear_back_Bias(const Matrix<InputType, M1_1, M1_2> &Output_Grad, const Matrix<Type_Act, M1_1, M1_2> &Act_Grad) {
    Matrix<OutputType, M1_2> out;
    for (Dim_size_t i1_2 = 0; i1_2 < M1_2; i1_2 += 1) {
        AccumulationType sum{static_cast<AccumulationType>(0)};
        for (Dim_size_t i1_1 = 0; i1_1 < M1_1; i1_1++) {
            sum += static_cast<AccumulationType>(Act_Grad.data[i1_1][i1_2]) * static_cast<AccumulationType>(Output_Grad.data[i1_1][i1_2]);
        }
        out.data[i1_2] = static_cast<OutputType>(sum);
    }
    return out;
}
