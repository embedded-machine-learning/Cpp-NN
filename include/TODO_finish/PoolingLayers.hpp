#pragma once
#include <type_traits>


#include "./Matrix.hpp"



template<
	Dim_size_t M1_1,
	Dim_size_t M1_2,
	Dim_size_t M1_3,
	typename Type = float>
__attribute__((always_inline))
inline
Matrix<Type, M1_1, M1_2 * M1_3> Flatten(const Matrix<float,M1_1,M1_2,M1_3>& Input){
	Matrix<Type, M1_1, M1_2 * M1_3> out;
	for(Dim_size_t i1_1 = 0; i1_1 < M1_1; i1_1++)
		for(Dim_size_t i1_2 = 0; i1_2 < M1_2; i1_2++)
			for(Dim_size_t i1_3 = 0; i1_3 < M1_3; i1_3++)
				out.data[i1_1][i1_2*M1_3+i1_3] = Input.data[i1_1][i1_2][i1_3];
	return out;
}

template<
	Dim_size_t M1_1,
	Dim_size_t M1_2,
	Dim_size_t M1_3,
	Dim_size_t M1_4,
	typename Type = float>
__attribute__((noinline))
inline
Matrix<Type, M1_1, M1_2*M1_3*M1_4> Flatten(const Matrix<Type,M1_1,M1_2,M1_3,M1_4>& Input){
	// Matrix<Type, M1_1, M1_2*M1_3*M1_4> out;
	// for(Dim_size_t i1_1 = 0; i1_1 < M1_1; i1_1++)
	// 	for(Dim_size_t i1_2 = 0; i1_2 < M1_2; i1_2++)
	// 		for(Dim_size_t i1_3 = 0; i1_3 < M1_3; i1_3++)
	// 			for(Dim_size_t i1_4 = 0; i1_4 < M1_4; i1_4++)
	// 				out.data[i1_1][i1_2*M1_3*M1_4+i1_3*M1_4+i1_4] = Input.data[i1_1][i1_2][i1_3][i1_4];
	// return out;
	return reinterpret_cast<const Matrix<Type, M1_1, M1_2*M1_3*M1_4>&>(Input);
}


template<
	Dim_size_t M1_1,
	Dim_size_t M1_2,
	Dim_size_t M1_3,
	typename Type = float>
__attribute__((always_inline))
inline
Matrix<Type, M1_1, M1_2, M1_3> UnFlatten(const Matrix<float,M1_1,M1_2*M1_3>& Input){
	Matrix<Type, M1_1, M1_2, M1_3> out;
	for(Dim_size_t i1_1 = 0; i1_1 < M1_1; i1_1++)
		for(Dim_size_t i1_2 = 0; i1_2 < M1_2; i1_2++)
			for(Dim_size_t i1_3 = 0; i1_3 < M1_3; i1_3++)
				out.data[i1_1][i1_2][i1_3] = Input.data[i1_1][i1_2*M1_3+i1_3];
	return out;
}


template<
	Dim_size_t M1_1,
	Dim_size_t M1_2,
	Dim_size_t M1_3,
	typename Type = float>
__attribute__((always_inline))
inline
Matrix<Type, M1_1, M1_2> AdaptiveAveragePool(const Matrix<Type,M1_1,M1_2,M1_3>& Input){
	Matrix<Type, M1_1, M1_2> out;
	for(Dim_size_t i1_1 = 0; i1_1 < M1_1; i1_1++)
		for(Dim_size_t i1_2 = 0; i1_2 < M1_2; i1_2++){
			Type sum{static_cast<Type>(0)};
			for(Dim_size_t i1_3 = 0; i1_3 < M1_3; i1_3++)
				sum += Input.data[i1_1][i1_2][i1_3];
			out.data[i1_1][i1_2] = sum / M1_3;
		}
	return out;
}
template<
	Dim_size_t M1_1,
	Dim_size_t M1_2,
	Dim_size_t M1_3,
	typename Type = float>
__attribute__((always_inline))
inline
void AdaptiveAveragePool(const Matrix<Type,M1_1,M1_2,M1_3>& Input,Matrix<Type, M1_1, M1_2> &out ){
	for(Dim_size_t i1_1 = 0; i1_1 < M1_1; i1_1++)
		for(Dim_size_t i1_2 = 0; i1_2 < M1_2; i1_2++){
			Type sum{static_cast<Type>(0)};
			for(Dim_size_t i1_3 = 0; i1_3 < M1_3; i1_3++)
				sum += Input.data[i1_1][i1_2][i1_3];
			out.data[i1_1][i1_2] = sum / M1_3;
		}
}

template<
	Dim_size_t pool_size,
	Dim_size_t M1_1,
	Dim_size_t M1_2,
	Dim_size_t M1_3,
	Dim_size_t M1_4,
	typename Type = float>
__attribute__((noinline))
inline
Matrix<Type, M1_1, M1_2,M1_3/pool_size,M1_4/pool_size> MaxPool2d(const Matrix<Type,M1_1,M1_2,M1_3,M1_4>& Input){
    static_assert(M1_3 % pool_size == 0, "Input Width has to be divisible by pool_size");
    static_assert(M1_4 % pool_size == 0, "Input Height has to be divisible by pool_size");

	Matrix<Type, M1_1, M1_2,M1_3/pool_size,M1_4/pool_size> out;
	for(Dim_size_t i1_1 = 0; i1_1 < M1_1; i1_1++)
		for(Dim_size_t i1_2 = 0; i1_2 < M1_2; i1_2++){
			for(Dim_size_t i1_3 = 0; i1_3 < M1_3; i1_3+=pool_size)
				for(Dim_size_t i1_4 = 0; i1_4 < M1_4; i1_4+=pool_size)
					{
					Type max{static_cast<Type>(0)};
					for(Dim_size_t i = 0; i < pool_size; i++)
						for(Dim_size_t j = 0; j < pool_size; j++)
							if(Input.data[i1_1][i1_2][i1_3+i][i1_4+j] > max)
								max = Input.data[i1_1][i1_2][i1_3+i][i1_4+j];
					out.data[i1_1][i1_2][i1_3/pool_size][i1_4/pool_size] = max;
					}
		}
	return out;
}