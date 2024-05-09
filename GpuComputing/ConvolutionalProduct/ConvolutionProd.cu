
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Utils\GpuTimer.cuh"

#include <stdio.h>
#include <stdlib.h>
#include "bmpUtils.h"
#include <iostream>

constexpr auto FILTER_RADIUS = 1;
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

/*
 *  Read a 24-bit/pixel BMP file into a 1D linear array.
 *  Allocate memory to store the 1D image and return its pointer
 */
pel* ReadBMPlin(char* fn) {
	static pel* Img;
	FILE* f = fopen(fn, "rb");
	if (f == NULL) {
		printf("\n\n%s NOT FOUND\n\n", fn);
		exit(EXIT_FAILURE);
	}

	pel HeaderInfo[54];
	size_t nByte = fread(HeaderInfo, sizeof(pel), 54, f); // read the 54-byte header

	// extract image height and width from header
	int width = *(int*)&HeaderInfo[18];
	img.width = width;
	int height = *(int*)&HeaderInfo[22];
	img.height = height;
	int RowBytes = (width * 3 + 3) & (~3);  // row is multiple of 4 pixel
	img.rowByte = RowBytes;

	//save header for re-use
	memcpy(img.headInfo, HeaderInfo, 54);
	printf("\n Input File name: %5s  (%d x %d)   File Size=%lu", fn, img.width,
		img.height, IMAGESIZE);

	// allocate memory to store the main image (1 Dimensional array)
	Img = (pel*)malloc(IMAGESIZE);
	if (Img == NULL)
		return Img;      // Cannot allocate memory
	// read the image from disk
	size_t out = fread(Img, sizeof(pel), IMAGESIZE, f);
	fclose(f);
	return Img;
}


int main()
{
	GpuTimer gpu_timer;
    const float f_h[] {(0.0f), (-1.0f), (0.0f), (-1.0f), (4.0f), (-1.0f), (0.0f), (-1.0f), (0.0f)};

	// create my filter f_h
    cudaMemcpyToSymbol(f, f_h, (FILTER_RADIUS * 2 + 1) * (FILTER_RADIUS * 2 + 1) * sizeof(float));
   

    return 0;
}

