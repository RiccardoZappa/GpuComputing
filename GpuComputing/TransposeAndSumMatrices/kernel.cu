
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

const int WIDTH = 500;
const int HEIGHT = 600;
const int BLOCK_DIM = 32;


__global__ void MatricesMultShared(float* d_M, float* d_N, float* d_P, int width, int height)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < height)
    {
        float pvalue;
        for (int k = 0; k < width; k++)
        {
            pvalue += d_M[row + k] * d_N[k * row];
        }
        d_P[row] = pvalue;
    }
}

int main(int argc, char** argv)
{
    float h_M[HEIGHT][WIDTH];
    float h_N[WIDTH][HEIGHT];
    float h_P[WIDTH][WIDTH];

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
    cudaMalloc((void**)d_M, WIDTH * HEIGHT * sizeof(int));
    cudaMalloc((void**)d_N, WIDTH * HEIGHT * sizeof(int));
    cudaMalloc((void**)d_P, WIDTH * WIDTH * sizeof(int));

    cudaMemcpy(d_M, h_M, WIDTH * HEIGHT * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, WIDTH * HEIGHT * sizeof(int), cudaMemcpyHostToDevice);

    int nBlocks = (HEIGHT + BLOCK_DIM - 1) / BLOCK_DIM;

    MatricesMultShared << < nBlocks, BLOCK_DIM >> > (d_M, d_N, d_P, WIDTH, HEIGHT);

    cudaMemcpy(h_P, d_P, WIDTH * WIDTH * sizeof(int), cudaMemcpyDeviceToHost);


    for (int i = 0; i < 1; i++)
    {
        for (int j = 0; j < WIDTH; j++)
        {
            std::cout << h_P[i][j] << std::endl;
        }
    }
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
