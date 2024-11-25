
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>


__global__ void MatricesMultShared(float* d_M, float* d_N, float* d_P, int width, int height)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < height && col < height)
    {
    	float pvalue = 0.0f;
    	for (int k = 0; k < width; k++)
    	{
    		pvalue += d_M[row * width + k] * d_N[k * height + col];
    	}

    	d_P[row * height + col] = pvalue;
    }
}

int main()
{

    const int WIDTH = 50;
    const int HEIGHT = 60;

    float h_M[HEIGHT][WIDTH];
    float h_N[WIDTH][HEIGHT];
    float h_P[HEIGHT][HEIGHT];

    // initializing the matrices
    for (int i = 0; i < HEIGHT; i++)
    {
        for (int j = 0; j < WIDTH; j++)
        {
            h_M[i][j] = 1.0f;
        }
    }

    for (int i = 0; i < WIDTH; i++)
    {
        for (int j = 0; j < HEIGHT; j++)
        {
            h_N[i][j] = 1.0f;
        }
    }
    //instanziate and allocate the device variables
    float* d_M, * d_N, * d_P;
    cudaMalloc((void**)&d_M, WIDTH * HEIGHT * sizeof(float));
    cudaMalloc((void**)&d_N, WIDTH * HEIGHT * sizeof(float));
    cudaMalloc((void**)&d_P, HEIGHT * HEIGHT * sizeof(float));

    cudaMemcpy(d_M, h_M, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil(HEIGHT / 16.0), ceil(HEIGHT / 16.0), 1);
    dim3 dimBlock(16, 16, 1);

    MatricesMultShared << < dimGrid, dimBlock >> > (d_M, d_N, d_P, WIDTH, HEIGHT);

    cudaMemcpy(h_P, d_P, HEIGHT * HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);


	for (int j = 0; j < HEIGHT; j++)
	{
		std::cout << h_P[0][j] << std::endl;
	}

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

    return 0;
}

