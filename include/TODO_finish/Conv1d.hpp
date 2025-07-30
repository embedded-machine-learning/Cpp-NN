#pragma once
#include "./Matrix.hpp"

/*
Simple Conv1d Layer, uses stack memory
*/
template <Dim_size_t stride  = 1,
          Dim_size_t padding = 0,
          Dim_size_t M1_1, // Input batch
          Dim_size_t M1_2, // Input Channels
          Dim_size_t M1_3, // Input Width
          Dim_size_t M2_1, // Weight Output Channels
          // Dim_size_t M2_2,	// Weight Input Channels has to be the same as M1_2
          Dim_size_t M2_3, // Weight Kernel shape
          typename Lambda,
          typename InputType        = float,
          typename AccumulationType = float,
          typename OutputType       = float,
          typename... ActivationInformation>
__attribute__((always_inline)) inline Matrix<OutputType, M1_1, M2_1, ((M1_3 - M2_3 + 2 * padding) / stride + 1)> Conv1d(const Matrix<InputType, M1_1, M1_2, M1_3> &Input,
                                                                                                                        const Matrix<InputType, M2_1, M1_2, M2_3> &Weights,
                                                                                                                        const Matrix<AccumulationType, M2_1>      &Bias,
                                                                                                                        Lambda                                     Act,
                                                                                                                        const Matrix<ActivationInformation, M2_1> &...ActivationParameters) {
    const Dim_size_t out_shape = (M1_3 - M2_3 + 2 * padding) / stride + 1;

    Matrix<OutputType, M1_1, M2_1, out_shape> out;

    for (Dim_size_t i1_1 = 0; i1_1 < M1_1; i1_1++) {
        for (Dim_size_t i2_1 = 0; i2_1 < M2_1; i2_1++) {
            for (Dim_size_t i1_3 = -padding; i1_3 < M1_3 - M2_3 + 1 + padding; i1_3 += stride) {
                // Preload the bias
                AccumulationType sum{Bias.data[i2_1]};
                // The convolution step
                for (Dim_size_t i1_2 = 0; i1_2 < M1_2; i1_2++) {
                    for (Dim_size_t i2_3 = 0; i2_3 < M2_3; i2_3++) {
                        if (i1_3 + i2_3 < 0 || i1_3 + i2_3 >= M1_3)
                            continue;
                        else {
                            sum += static_cast<AccumulationType>(Input.data[i1_1][i1_2][i1_3 + i2_3]) * static_cast<AccumulationType>(Weights.data[i2_1][i1_2][i2_3]);
                        }
                    }
                }
                out.data[i1_1][i2_1][(i1_3 + padding) / stride] = Act(sum, ActivationParameters[i2_1]...);
            }
        }
    }
    return out;
}

/*
Simple Conv1d Layer, uses mostly passed memory
*/
template <Dim_size_t stride,
          Dim_size_t padding,
          Dim_size_t M1_1, // Input batch
          Dim_size_t M1_2, // Input Channels
          Dim_size_t M1_3, // Input Width
          Dim_size_t M2_1, // Weight Output Channels
          // Dim_size_t M2_2,	// Weight Input Channels has to be the same as M1_2
          Dim_size_t M2_3, // Weight Kernel shape
          typename Lambda,
          typename InputType        = float,
          typename AccumulationType = float,
          typename OutputType       = float,
          typename... ActivationInformation>
__attribute__((always_inline)) inline void Conv1d(const Matrix<InputType, M1_1, M1_2, M1_3>                                  &Input,
                                                  Matrix<OutputType, M1_1, M2_1, ((M1_3 - M2_3 + 2 * padding) / stride + 1)> &out,
                                                  const Matrix<InputType, M2_1, M1_2, M2_3>                                  &Weights,
                                                  const Matrix<AccumulationType, M2_1>                                       &Bias,
                                                  Lambda                                                                      Act,
                                                  const Matrix<ActivationInformation, M2_1> &...ActivationParameters) {
    for (Dim_size_t i1_1 = 0; i1_1 < M1_1; i1_1++) {
        for (Dim_size_t i2_1 = 0; i2_1 < M2_1; i2_1++) {
            for (Dim_size_t i1_3 = -padding; i1_3 < M1_3 - M2_3 + 1 + padding; i1_3 += stride) {
                // Preload the bias
                AccumulationType sum{Bias.data[i2_1]};
                // The convolution step
                for (Dim_size_t i1_2 = 0; i1_2 < M1_2; i1_2++) {
                    for (Dim_size_t i2_3 = 0; i2_3 < M2_3; i2_3++) {
                        if (i1_3 + i2_3 < 0 || i1_3 + i2_3 >= M1_3)
                            continue;
                        else {
                            sum += static_cast<AccumulationType>(Input.data[i1_1][i1_2][i1_3 + i2_3]) * static_cast<AccumulationType>(Weights.data[i2_1][i1_2][i2_3]);
                        }
                    }
                }
                out.data[i1_1][i2_1][(i1_3 + padding) / stride] = Act(sum);
            }
        }
    }
}

/*
Type information for the Conv1d Layer
*/
template <Dim_size_t stride, Dim_size_t padding, typename Input, typename Weights, typename OutputType>
struct Conv_Generate_out_type;

template <Dim_size_t stride,
          Dim_size_t padding,
          Dim_size_t M1_1, // Input batch
          Dim_size_t M1_2, // Input Channels
          Dim_size_t M1_3, // Input Width
          Dim_size_t M2_1, // Weight Output Channels
          // Dim_size_t M2_2,	// Weight Input Channels has to be the same as M1_2
          Dim_size_t M2_3, // Weight Kernel shape
          typename InputType,
          typename OutputType>
struct Conv_Generate_out_type<stride, padding, Matrix<InputType, M1_1, M1_2, M1_3>, Matrix<InputType, M2_1, M1_2, M2_3>, OutputType> {
    using type = Matrix<OutputType, M1_1, M2_1, ((M1_3 - M2_3 + 2 * padding) / stride + 1)>;
};

template <Dim_size_t stride,
          Dim_size_t padding,
          Dim_size_t M1_1, // Input batch
          Dim_size_t M1_2, // Input Channels
          Dim_size_t M1_3, // Input Width
          Dim_size_t M2_1, // Weight Output Channels
          // Dim_size_t M2_2,	// Weight Input Channels has to be the same as M1_2
          Dim_size_t M2_3, // Weight Kernel shape
          typename InputType,
          typename OutputType>
struct Conv_Generate_out_type<stride, padding, const Matrix<InputType, M1_1, M1_2, M1_3> &, Matrix<InputType, M2_1, M1_2, M2_3>, OutputType> {
    using type = Matrix<OutputType, M1_1, M2_1, ((M1_3 - M2_3 + 2 * padding) / stride + 1)>;
};

/*
Trainable Conv1d Layer
TODO: Fix type mess
*/
template <Dim_size_t stride  = 1,
          Dim_size_t padding = 0,
          Dim_size_t M1_1, // Input batch
          Dim_size_t M1_2, // Input Channels
          Dim_size_t M1_3, // Input Width
          Dim_size_t M2_1, // Weight Output Channels
          // Dim_size_t M2_2,	// Weight Input Channels has to be the same as M1_2
          Dim_size_t M2_3, // Weight Kernel shape
          typename Lambda,
          typename InputType        = float,
          typename AccumulationType = float,
          typename OutputType       = float,
          typename GradType         = int8_t,
          typename... ActivationInformation>
__attribute__((always_inline)) inline std::pair<Matrix<OutputType, M1_1, M2_1, ((M1_3 - M2_3 + 2 * padding) / stride + 1)>, Matrix<GradType, M1_1, M2_1, ((M1_3 - M2_3 + 2 * padding) / stride + 1)>>
Conv1d_train(const Matrix<InputType, M1_1, M1_2, M1_3> &Input,
             const Matrix<InputType, M2_1, M1_2, M2_3> &Weights,
             const Matrix<AccumulationType, M2_1>      &Bias,
             Lambda                                     Act,
             const Matrix<ActivationInformation, M2_1> &...ActivationParameters) {
    const Dim_size_t                            out_shape = (M1_3 - M2_3 + 2 * padding) / stride + 1;
    Matrix<OutputType, M1_1, M2_1, out_shape> out;
    Matrix<GradType, M1_1, M2_1, out_shape>   backprob;

    for (Dim_size_t i1_1 = 0; i1_1 < M1_1; i1_1++) {
        for (Dim_size_t i2_1 = 0; i2_1 < M2_1; i2_1++) {
            for (Dim_size_t i1_3 = -padding; i1_3 < M1_3 - M2_3 + 1 + padding; i1_3 += stride) {
                // Preload the bias
                AccumulationType sum{Bias.data[i2_1]};
                // The convolution step
                for (Dim_size_t i1_2 = 0; i1_2 < M1_2; i1_2++) {
                    for (Dim_size_t i2_3 = 0; i2_3 < M2_3; i2_3++) {
                        if (i1_3 + i2_3 < 0 || i1_3 + i2_3 >= M1_3)
                            continue;
                        else {
                            sum += static_cast<AccumulationType>(Input.data[i1_1][i1_2][i1_3 + i2_3]) * static_cast<AccumulationType>(Weights.data[i2_1][i1_2][i2_3]);
                        }
                    }
                }
                std::tie(out.data[i1_1][i2_1][(i1_3 + padding) / stride], backprob.data[i1_1][i2_1][(i1_3 + padding) / stride]) = Act(sum, ActivationParameters[i2_1]...);
            }
        }
    }
    return {out, backprob};
}

template <Dim_size_t stride,
          // Dim_size_t padding,
          Dim_size_t M1_1, // Input batch
          Dim_size_t M1_2, // Input Channels
          Dim_size_t M1_3, // Input Width
          typename InputType        = float,
          typename AccumulationType = float,
          typename OutputType       = float,
          typename GradType         = int8_t>
__attribute__((always_inline)) inline Matrix<OutputType, M1_2> Conv1d_back_Bias(const Matrix<InputType, M1_1, M1_2, M1_3> &Output_Grad, const Matrix<GradType, M1_1, M1_2, M1_3> &Act_Grad) {
    Matrix<OutputType, M1_2> out;
    for (Dim_size_t i1_2 = 0; i1_2 < M1_2; i1_2 += 1) {
        AccumulationType sum{static_cast<AccumulationType>(0)};
        for (Dim_size_t i1_1 = 0; i1_1 < M1_1; i1_1++)
            for (Dim_size_t i1_3 = 0; i1_3 < M1_3; i1_3++) {
                sum += static_cast<AccumulationType>(Act_Grad.data[i1_1][i1_2][i1_3]) * static_cast<AccumulationType>(Output_Grad.data[i1_1][i1_2][i1_3]);
            }
        out.data[i1_2] = static_cast<OutputType>(sum);
    }
    return out;
}

template <Dim_size_t stride,
          Dim_size_t padding,
          Dim_size_t M1_1,
          Dim_size_t M1_2,
          Dim_size_t M1_3,
          // Dim_size_t M2_1,
          Dim_size_t M2_2,
          Dim_size_t M2_3,
          typename InputType  = float,
          typename OutputType = float,
          typename GradType   = int8_t>
__attribute__((always_inline)) inline Matrix<OutputType, M1_1, M2_2, M1_3 + M2_3 - 1 - 2 * padding> Conv1d_back_Input(const Matrix<InputType, M1_1, M1_2, M1_3> &Output_Grad,
                                                                                                                      const Matrix<GradType, M1_1, M1_2, M1_3>  &Act_Grad,
                                                                                                                      const Matrix<InputType, M1_2, M2_2, M2_3> &Weights) {
    // input_grad[:,:,b+a] += (true_resT.grad[:,:,b].view(s[0],s[1],1)*weightsT[:,:,a].view(1,s2[0],s2[1])).sum(1)
    Matrix<OutputType, M1_1, M2_2, M1_3 + M2_3 - 1 - 2 * padding> out;
    for (Dim_size_t i1_1 = 0; i1_1 < M1_1; i1_1++) {
        for (Dim_size_t i2_2 = 0; i2_2 < M2_2; i2_2++) {
            for (Dim_size_t i = 0; i < M1_3 + M2_3 - 1 - 2 * padding; i++) {
                out.data[i1_1][i2_2][i] = static_cast<AccumulationType>(0);
            }
            for (Dim_size_t i1_3 = 0; i1_3 < M1_3; i1_3++) {
                for (Dim_size_t i2_3 = 0; i2_3 < M2_3; i2_3++) {
                    for (Dim_size_t i1_2 = 0; i1_2 < M1_2; i1_2++) {
                        if (i1_3 + i2_3 - padding < 0 || i1_3 + i2_3 - padding >= M1_3 + M2_3 - 1 - 2 * padding)
                            continue;
                        else
                            out.data[i1_1][i2_2][i1_3 + i2_3 - padding] +=
                                static_cast<OutputType>(static_cast<AccumulationType>(Act_Grad.data[i1_1][i1_2][i1_3]) * static_cast<AccumulationType>(Output_Grad.data[i1_1][i1_2][i1_3]) *
                                                        static_cast<AccumulationType>(Weights.data[i1_2][i2_2][i2_3]));
                    }
                }
            }
        }
    }
    return out;
}

template <Dim_size_t stride,
          Dim_size_t padding,
          Dim_size_t M1_1,
          Dim_size_t M1_2,
          Dim_size_t M1_3,
          // Dim_size_t M2_1,
          Dim_size_t M2_2,
          Dim_size_t M2_3,
          typename InputType        = float,
          typename AccumulationType = float,
          typename OutputType       = float,
          typename GradType         = int8_t>
__attribute__((always_inline)) inline Matrix<OutputType, M1_2, M2_2, M2_3 - M1_3 + 1 + 2 * padding> Conv1d_back_Weights(const Matrix<InputType, M1_1, M1_2, M1_3> &Output_Grad,
                                                                                                                        const Matrix<GradType, M1_1, M1_2, M1_3>  &Act_Grad,
                                                                                                                        const Matrix<InputType, M1_1, M2_2, M2_3> &Inputs) {
    // tmp = true_resT.grad.view(s[0],1,s[1],s[2])*inputT[:,:,a:inputT.shape[2]-weightsT.shape[2]+a+1].view(s2[0],s2[1],1,s[2])
    // tmp.sum(-1).sum(0).T
    Matrix<OutputType, M1_2, M2_2, M2_3 - M1_3 + 1 + 2 * padding> out;

    for (Dim_size_t i1_2 = 0; i1_2 < M1_2; i1_2++) {
        for (Dim_size_t i2_2 = 0; i2_2 < M2_2; i2_2++) {
            for (Dim_size_t i = -padding; i < M2_3 - M1_3 + 1 + padding; i++) {
                AccumulationType sum{static_cast<AccumulationType>(0)};
                for (Dim_size_t i1_1 = 0; i1_1 < M1_1; i1_1++) {
                    for (Dim_size_t i1_3 = 0; i1_3 < M1_3; i1_3++) {
                        if (i1_3 + i < 0 || i1_3 + i >= M2_3)
                            continue;
                        else
                            sum += static_cast<AccumulationType>(Act_Grad.data[i1_1][i1_2][i1_3]) * static_cast<AccumulationType>(Output_Grad.data[i1_1][i1_2][i1_3]) *
                                   static_cast<AccumulationType>(Inputs.data[i1_1][i2_2][i1_3 + i]);
                    }
                }
                out.data[i1_2][i2_2][i + padding] = static_cast<OutputType>(sum);
            }
        }
    }

    return out;
}