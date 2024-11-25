#include 
#include "nn_layer.h"
#include "../utils/cuda_API_check.h"

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