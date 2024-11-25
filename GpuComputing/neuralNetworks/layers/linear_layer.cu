#include <device_launch_parameters.h>
#include <random>
#include <vector_types.h>

#include "nn_layer.h"

#include "../utils/cuda_API_check.h"

// kernels

/*
LinearLayer forward pass CUDA kernel: simply multiply each row element of
matrix W with each column element of matrix A. Finally we add a bias to our result.
The computed output of a linear layer forward pass is the Z matrix.
*/
__global__ void linearLayerForward(float* W, float* A, float* Z, float* b,
	int W_x_dim, int W_y_dim,
	int A_x_dim, int A_y_dim) {

	// TODO

}

/*
LinearLayer backward pass CUDA kernel (computing dA):we need W matrix
to be transposed. Instead of making separate kernel for transposition we
can simply multiply each W column by dZ columns. It will be equivalent of
computing transpose(W)*dZ
*/
__global__ void linearLayerBackprop(float* W, float* dZ, float* dA,
	int W_x_dim, int W_y_dim,
	int dZ_x_dim, int dZ_y_dim) {

	// TODO

}

/*
LinearLayer weights update CUDA kernel (gradient descent): We apply similar
trick here to pretend that A matrix is transposed. The final step in this kernel
is updating weights matrix. We are using the simplest form of gradient descent
here and just subtract gradient value multiplied by learning rate from current
weights matrix.
*/
__global__ void linearLayerUpdateWeights(float* dZ, float* A, float* W,
	int dZ_x_dim, int dZ_y_dim,
	int A_x_dim, int A_y_dim,
	float learning_rate) {

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	// A is treated as transposed
	int W_x_dim = A_y_dim;
	int W_y_dim = dZ_y_dim;

	float dW_value = 0.0f;

	if (row < W_y_dim && col < W_x_dim) {
		for (int i = 0; i < dZ_x_dim; i++) {
			dW_value += dZ[row * dZ_x_dim + i] * A[col * A_x_dim + i];
		}
		W[row * W_x_dim + col] = W[row * W_x_dim + col] - learning_rate * (dW_value / A_x_dim);
	}
}

/*
LinearLayer bias update CUDA kernel (gradient descent): The last step during
backpropagation in our linear layer is performing bias vector update.
*/
__global__ void linearLayerUpdateBias(float* dZ, float* b,
	int dZ_x_dim, int dZ_y_dim,
	int b_x_dim,
	float learning_rate) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < dZ_x_dim * dZ_y_dim) {
		int dZ_x = index % dZ_x_dim;
		int dZ_y = index / dZ_x_dim;
		atomicAdd(&b[dZ_y], -learning_rate * (dZ[dZ_y * dZ_x_dim + dZ_x] / dZ_x_dim));
	}
}

// constructor
LinearLayer::LinearLayer(std::string name, Shape shape) : W(shape), b(shape.y_dim, 1) {
	this->name = name;
	this->shape = shape;
	b.allocateMemory();
	W.allocateMemory();
	initializeWeightsRandomly();
	initializeBiasWithZeros();
}

void LinearLayer::initializeWeightsRandomly() {
	std::default_random_engine generator;
//	std::normal_distribution normal_distribution(0.0, 1.0);

	for (int x = 0; x < W.shape.x_dim; x++) {
		for (int y = 0; y < W.shape.y_dim; y++) {
			W.data_host[y * W.shape.x_dim + x] = /*std::normal_distribution(generator) * */weights_init_threshold;
		}
	}
	W.copyHostToDevice();
}

void LinearLayer::initializeBiasWithZeros() {
	for (int x = 0; x < shape.x_dim; x++)
		b.data_host[x] = 0;
	b.copyHostToDevice();
}

Matrix& LinearLayer::forward(Matrix& A) {
	this->A = A;

	Z.allocateMemoryIfNotAllocated({ A.shape.x_dim, W.shape.y_dim });
	computeAndStoreLayerOutput(A);
	return Z;
}

void LinearLayer::computeAndStoreLayerOutput(Matrix& A) {
	dim3 block_size(16, 16);
	dim3 num_of_blocks((Z.shape.x_dim + block_size.x - 1) / block_size.x,
		(Z.shape.y_dim + block_size.y - 1) / block_size.y);
	/*linearLayerForward<<2>>(W.data_device,
		A.data_device,
		Z.data_device,
		b.data_device,*/
		/*W.shape.x_dim, W.shape.y_dim,
		A.shape.x_dim, A.shape.y_dim);*/
}

Matrix& LinearLayer::backprop(Matrix& dZ, float learning_rate) {
	dA.allocateMemoryIfNotAllocated(A.shape);
	computeAndStoreBackpropError(dZ);
	updateBias(dZ, learning_rate);
	updateWeights(dZ, learning_rate);

	return dA;
}

void LinearLayer::computeAndStoreBackpropError(Matrix& dZ) {
	dim3 block_size(8, 8);
	dim3 num_of_blocks((A.shape.x_dim + block_size.x - 1) / block_size.x, (A.shape.y_dim + block_size.y - 1) / block_size.y);
	/*linearLayerBackprop<<2>>(W.data_device,
		dZ.data_device,
		dA.data_device,
		W.shape.x_dim, W.shape.y_dim,
		dZ.shape.x_dim, dZ.shape.y_dim);*/
}

void LinearLayer::updateWeights(Matrix& dZ, float learning_rate) {
	dim3 block_size(8, 8);
	dim3 num_of_blocks((W.shape.x_dim + block_size.x - 1) / block_size.x, (W.shape.y_dim + block_size.y - 1) / block_size.y);
	/*linearLayerUpdateWeights<<2>>(dZ.data_device,
		A.data_device,
		W.data_device,
		dZ.shape.x_dim, dZ.shape.y_dim,
		A.shape.x_dim, A.shape.y_dim,
		learning_rate);*/
}

void LinearLayer::updateBias(Matrix& dZ, float learning_rate) {
	dim3 block_size(256);
	dim3 num_of_blocks((dZ.shape.y_dim * dZ.shape.x_dim + block_size.x - 1) / block_size.x);
	/*linearLayerUpdateBias<<2>>(dZ.data_device,
		b.data_device,
		dZ.shape.x_dim, dZ.shape.y_dim,
		b.shape.x_dim, learning_rate);*/
}

int LinearLayer::getXDim() const {
	return W.shape.x_dim;
}

int LinearLayer::getYDim() const {
	return W.shape.y_dim;
}

Matrix LinearLayer::getWeightsMatrix() const {
	return W;
}

Matrix LinearLayer::getBiasVector() const {
	return b;
}