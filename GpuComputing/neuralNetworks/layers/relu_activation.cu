#include <vector_types.h>

#include "nn_layer.h"
#include "../utils/cuda_API_check.h"


struct Matrix;
/*
ReLUActivation forward pass CUDA kernel: A_i = fmaxf(Z_i, 0)
*/
__global__ void reluActivationForward(float* Z, float* A, int Z_x_dim, int Z_y_dim) {

	// TODO

}

/*
ReLUActivation backward pass CUDA kernel: We had to use if statement here in
order to check whether Z input was greater or lower than 0.
*/
__global__ void reluActivationBackprop(float* Z, float* dA, float* dZ, int Z_x_dim, int Z_y_dim) {

	// TODO

}

Matrix& ReLUActivation::forward(Matrix& Z) {
	this->Z = Z;
	A.allocateMemoryIfNotAllocated(Z.shape);

	dim3 block_size(256);
	dim3 num_of_blocks((Z.shape.y_dim * Z.shape.x_dim + block_size.x - 1) / block_size.x);

	/*reluActivationForward<<2>>(Z.data_device,
		A.data_device,
		Z.shape.x_dim, Z.shape.y_dim);*/
	return A;
}

Matrix& ReLUActivation::backprop(Matrix& dA, float learning_rate) {
	dZ.allocateMemoryIfNotAllocated(Z.shape);
	dim3 block_size(256);
	dim3 num_of_blocks((Z.shape.y_dim * Z.shape.x_dim + block_size.x - 1) / block_size.x);
	/*reluActivationBackprop<<2>>(Z.data_device,
		dA.data_device,
		dZ.data_device,
		Z.shape.x_dim, Z.shape.y_dim);*/
	return dZ;
}


