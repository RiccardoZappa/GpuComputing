
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Utils\GpuTimer.cuh"

#include <stdio.h>
#include <stdlib.h>
#include "bmpUtils.h"
#include <iostream>

constexpr auto FILTER_RADIUS = 1;
using namespace utils;
__constant__ float F [(FILTER_RADIUS * 2 + 1) * (FILTER_RADIUS * 2 + 1)];


__global__ void convolution_2D_basic_kernel(pel* N, pel* P,const int r, int width, int height,float* f)
{
	int idx = blockIdx.x * 128 + threadIdx.x;
	uint BlockPerRow = (width + 127) / 128;
	uint rows = blockIdx.x / BlockPerRow;
	uint columns = idx - rows * width;
	uint numBytePerRow = (width * 3 + 3) & (~3);
	uint IndexSrc = numBytePerRow * rows + columns * 3;
	float PValue = 0.0f;
	float PValue2 = 0.0f;
	float PValue3 = 0.0f;
	for (int f_Row = 0; f_Row < 2 * r + 1; f_Row++)
	{
		for (int f_Col = 0; f_Col < 2 * r + 1; f_Col++)
		{
			if (idx >= 0 && idx  < width)
			{
				PValue += f[f_Row * 3 + f_Col] * N[IndexSrc];
				PValue2 += f[f_Row * 3 + f_Col] * N[IndexSrc + 1];
				PValue3 += f[f_Row * 3 + f_Col] * N[IndexSrc + 2];
			}
		}
	}
	P[IndexSrc] = PValue;
	P[IndexSrc + 1] = PValue2;
	P[IndexSrc + 2] = PValue3;
}
/*
 *  Kernel that apply grayscale to my image
 */
__global__ void ImgGrayscale(pel* ImgDst, pel* ImgSrc, uint width) {

	int idx = blockIdx.x * 128 + threadIdx.x;
	uint BlockPerRow = (width + 127) / 128;
	uint rows = blockIdx.x / BlockPerRow;
	uint columns = idx - rows * width;
	uint numBytePerRow = (width * 3 + 3) & (~3);
	uint IndexSrc = numBytePerRow * rows + columns * 3;

	ImgDst[IndexSrc] = 0.299*ImgSrc[IndexSrc] + 0.587*ImgSrc[IndexSrc + 1] + 0.114*ImgSrc[IndexSrc + 2];
	ImgDst[IndexSrc + 1] = 0.299*ImgSrc[IndexSrc] + 0.587*ImgSrc[IndexSrc + 1] + 0.114 *ImgSrc[IndexSrc + 2];
	ImgDst[IndexSrc + 2] = 0.299*ImgSrc[IndexSrc] + 0.587*ImgSrc[IndexSrc + 1] + 0.114*ImgSrc[IndexSrc + 2];
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

/*
 *  Write the 1D linear-memory stored image into file
 */
void WriteBMPlin(pel* Img, char* fn) {
	FILE* f = fopen(fn, "wb");
	if (f == NULL) {
		printf("\n\nFILE CREATION ERROR: %s\n\n", fn);
		exit(1);
	}
	//write header
	fwrite(img.headInfo, sizeof(pel), 54, f);
	//write data
	fwrite(Img, sizeof(pel), IMAGESIZE, f);
	printf("\nOutput File name: %5s  (%u x %u)   File Size=%lu", fn, img.width,
		img.height, IMAGESIZE);
	fclose(f);
}

int main()
{
	uint dimBlock = 128, dimGrid;
	char fileName[100] = "C:\\GpuComputing\\GpuComputing\\images\\dog.bmp";
	char fileNameWrite[100] = "C:\\GpuComputing\\GpuComputing\\images\\dogGray.bmp";
	char fileNameWrite2[100] = "C:\\GpuComputing\\GpuComputing\\images\\dogSharpen.bmp";
	pel* imgSrc, * imgDst;		 // Where images are stored in CPU
	pel* imgSrcGPU, *imgDstGPU, *imgHelpGPU;	 // Where images are stored in GPU
	GpuTimer gpuTimer,gpu_Timer2; // to monitor the performance of the gpu operations
	cudaError_t error;

	//float f_h[9];
	//f_h[0] = 0.0f;
	//f_h[1] = -1.0f;
	//f_h[2] = 0.0f;
	//f_h[3] = -1.0f;
	//f_h[4] = 5.0f;
	//f_h[5] = -1.0f;
	//f_h[6] = 0.0f;
	//f_h[7] = -1.0f;
	//f_h[8] = 0.0f;
	float f_h[9];
	f_h[0] = 5.0f;
	f_h[1] = -15.0f;
	f_h[2] = 5.0f;
	f_h[3] = -14.0f;
	f_h[4] = 5.0f;
	f_h[5] = -15.0f;
	f_h[6] = 0.0f;
	f_h[7] = -1.0f;
	f_h[8] = 0.0f;
	// create my filter f_h
    /*cudaError_t error2 = cudaMemcpyToSymbol(f, f_h, (FILTER_RADIUS * 2 + 1) * (FILTER_RADIUS * 2 + 1) * sizeof(float));*/
	/*if (error2 != cudaSuccess)
	{
		printf("Error in cudaMemcpy imgSrc to imgSrcGpu: %d/n", error);
		return -1;
	}*/
	cudaDeviceSynchronize();
	// Create CPU memory to store the input and output images
	imgSrc = ReadBMPlin(fileName); // Read the input image if memory can be allocated
	if (imgSrc == NULL) {
		printf("Cannot allocate memory for the input image...\n");
		exit(EXIT_FAILURE);
	}

	imgDst = (pel*)malloc(IMAGESIZE);
	if (imgDst == NULL) {
		free(imgSrc);
		printf("Cannot allocate memory for the input image...\n");
		exit(EXIT_FAILURE);
	}
	// Allocate GPU buffer for the input and output images
	error = cudaMalloc((void**)&imgSrcGPU, IMAGESIZE * sizeof(pel));
	if (error != cudaSuccess)
	{
		printf("Error in CudaMalloc imgSrcGpu: %d/n", error);
		return -1;
	}

	error = cudaMalloc((void**)&imgDstGPU, IMAGESIZE * sizeof(pel));
	if (error != cudaSuccess)
	{
		printf("Error in CudaMalloc imgDstGpu: %d/n", error);
		return -1;
	}
	error = cudaMalloc((void**)&imgHelpGPU, IMAGESIZE * sizeof(pel));
	if (error != cudaSuccess)
	{
		printf("Error in CudaMalloc imgDstGpu: %d/n", error);
		return -1;
	}
	// Copy input vectors from host memory to GPU buffers.
	error = cudaMemcpy(imgSrcGPU, imgSrc, IMAGESIZE, cudaMemcpyHostToDevice);
	if (error != cudaSuccess)
	{
		printf("Error in cudaMemcpy imgSrc to imgSrcGpu: %d/n", error);
		return -1;
	}

	gpuTimer.Start();
	// invoke kernels (define grid and block sizes)
	int rowBlock = (WIDTH + dimBlock - 1) / dimBlock;
	dimGrid = HEIGHT * rowBlock;

	ImgGrayscale << <dimGrid, dimBlock >> > (imgHelpGPU, imgSrcGPU, WIDTH);
	// Copy output (results) from GPU buffer to host (CPU) memory.
	cudaDeviceSynchronize();
	cudaMemcpy(imgDst, imgHelpGPU, IMAGESIZE, cudaMemcpyDeviceToHost);

	gpuTimer.Stop();
	WriteBMPlin(imgDst, fileNameWrite);
	printf("\nKernel elapsed time %f ms \n\n", gpuTimer.Elapsed());
	error = cudaFree(imgSrcGPU);
	if (error != cudaSuccess)
	{
		printf("Error in CudaFree imgSrcGpu: %d/n", error);
		return -1;
	}
	gpu_Timer2.Start();
	convolution_2D_basic_kernel << < dimGrid, dimBlock >> > (imgDstGPU, imgHelpGPU, 1, WIDTH, HEIGHT,f_h);
	cudaDeviceSynchronize();
	// Copy output (results) from GPU buffer to host (CPU) memory.
	cudaMemcpy(imgDst, imgDstGPU, IMAGESIZE, cudaMemcpyDeviceToHost);
	gpu_Timer2.Stop();
	// Write the flipped image back to disk
	WriteBMPlin(imgDst, fileNameWrite2);
	printf("\nKernel elapsed time %f ms \n\n", gpu_Timer2.Elapsed());

	error = cudaFree(imgDstGPU);
	if (error != cudaSuccess)
	{
		printf("Error in CudaFree imgDstGpu: %d/n", error);
		return -1;
	}
	// Deallocate CPU, GPU memory and destroy events.

	// cuda free vars
	error = cudaFree(imgHelpGPU);
	if (error != cudaSuccess)
	{
		printf("Error in CudaFree imgHelpGpu: %d/n", error);
		return -1;
	}
	free(imgSrc);
	free(imgDst);

    return 0;
}

