
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Utils\GpuTimer.cuh"

constexpr auto FILTER_RADIUS = 2;
using namespace utils;
__constant__ float f [FILTER_RADIUS * 2 + 1][FILTER_RADIUS * 2 + 1];

__global__ void convolution_2D_basic_kernel(float* N, float* P,const int r, int width, int height)
{
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    float Pvalue = 0.0f;
    for (int f_Row = 0; f_Row < 2*r + 1; f_Row++)
    {
	    for(int f_Col=0; f_Col < 2*r + 1; f_Col++)
	    {
            int in_Row = out_row - r + f_Row;
            int in_Col = out_col - r + f_Col;
            if (in_Row >= 0 && in_Row < height && in_Col >= 0 && in_Col < width)
            {
                Pvalue += f[f_Row][f_Col] * N[in_Row * width + in_Col];
            }
	    }
    }
    P[out_row * width + out_col] = Pvalue;
}

int main()
{
	GpuTimer gpu_timer;
    float* f_h;
    // create my filter f_h
    cudaMemcpyToSymbol(f_h, f, (FILTER_RADIUS * 2 + 1) * (FILTER_RADIUS * 2 + 1) * sizeof(float));
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    return 0;
}

