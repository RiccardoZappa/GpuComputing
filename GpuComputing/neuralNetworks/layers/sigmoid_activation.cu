#include <vector_types.h>

#include "../utils/matrix.h"
#include "nn_layer.h"
#include "../utils/cuda_API_check.h"
#include <cuda_runtime.h>

__device__ float sigmoid(float x) {
	return 1.0f / (1 + __expf(-x));
}

/*
SigmoidActivation forward pass CUDA kernel: We calculate index for current
thread that is executing the kernel, then we check if this index is within
matrix bounds and compute sigmoid activation.
*/
__global__ void sigmoidActivationForward(float* Z, float* A, int Z_x_dim, int Z_y_dim) {

	// TODO

}

/*
SigmoidActivation backward pass CUDA kernel: Backward pass logic is very similar
to forward pass, the only difference is that this time we are implementing another
equation.
*/
__global__ void sigmoidActivationBackprop(float* Z, float* dA, float* dZ, int Z_x_dim, int Z_y_dim) {

	// TODO

}

Matrix& SigmoidActivation::forward(Matrix& Z) {
	this->Z = Z;
	A.allocateMemoryIfNotAllocated(Z.shape);

	dim3 block_size(256);
	dim3 num_of_blocks((Z.shape.y_dim * Z.shape.x_dim + block_size.x - 1) / block_size.x);

	/*sigmoidActivationForward<<2>>(Z.data_device,
		A.data_device,
		Z.shape.x_dim, Z.shape.y_dim);*/
	return A;
}

Matrix& SigmoidActivation::backprop(Matrix& dA, float learning_rate) {
	dZ.allocateMemoryIfNotAllocated(Z.shape);

	dim3 block_size(256);
	dim3 num_of_blocks((Z.shape.y_dim * Z.shape.x_dim + block_size.x - 1) / block_size.x);
	/*sigmoidActivationBackprop<<2>>(Z.data_device,
		dA.data_device,
		dZ.data_device,
		Z.shape.x_dim, Z.shape.y_dim);*/
	return dZ;
}
