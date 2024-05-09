#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils/GpuTimer.cuh"

#include <stdio.h>
#include <stdlib.h>
#include "bmpUtils.h"
#include <iostream>

using namespace utils;

/*
 * Kernel 1D that flips the given image vertically
 * each thread only flips a single pixel (R,G,B)
 */
__global__ void VflipGPU(pel* imgDst, const pel* imgSrc, const uint w, const uint h) {

	int idx = blockIdx.x * 128 + threadIdx.x;
	uint m = (w + 127) / 128; // number of blocks per row. 
	uint r = blockIdx.x / m; // rows, divide the number of blocks for the bocks for row.
	uint c = idx - r * w; // number of columns
	uint s = (w * 3 + 3) & (~3); // num bytes x row (mult. 4)
	uint r1 = h - 1 - r; // dest. row (mirror)
	// ** byte granularity **
	uint p = s * r + 3 * c; // src byte position of the pixel
	uint q = s * r1 + 3 * c; // dst byte position of the pixel
	// swap pixels RGB
	if (idx < w * h)
	{
	imgDst[q] = imgSrc[p]; // R
	imgDst[q + 1] = imgSrc[p + 1]; // G
	imgDst[q + 2] = imgSrc[p + 2]; // B
	}
	
}

/*
 *  Kernel that flips the given image horizontally
 *  each thread only flips a single pixel (R,G,B)
 */
__global__ void HflipGPU(pel* ImgDst, pel* ImgSrc, uint width) {

	int idx = blockIdx.x * 128 + threadIdx.x;
	uint BlockPerRow = (width + 127) / 128;
	uint rows = blockIdx.x / BlockPerRow;
	uint columns = idx - rows * width;
	uint numBytePerRow = (width * 3 + 3) & (~3);
	uint IndexSrc = numBytePerRow * rows + columns * 3;
	uint IndexDst = numBytePerRow * (rows + 1 ) - columns * 3;
	
	ImgDst[IndexSrc] = ImgSrc[IndexDst];
	ImgDst[IndexSrc + 1] = ImgSrc[IndexDst + 1];
	ImgDst[IndexSrc + 2] = ImgSrc[IndexDst + 2];
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

/*
 * MAIN
 */
int main(int argc, char** argv) {
	char flip = 'V';
	uint dimBlock = 128, dimGrid;
	char fileName[100] = "C:\\GpuComputing\\GpuComputing_es1\\images\\dog.bmp";
	char fileNameWrite[100] = "C:\\GpuComputing\\GpuComputing_es1\\images\\dogV.bmp";
	pel* imgSrc, * imgDst;		 // Where images are stored in CPU
	pel* imgSrcGPU, * imgDstGPU;	 // Where images are stored in GPU
	GpuTimer gpuTimer; // to monitor the performance of the gpu operations

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
	cudaError_t error = cudaMalloc((void**)&imgSrcGPU, IMAGESIZE * sizeof(pel));
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
	// Copy input vectors from host memory to GPU buffers.
	error = cudaMemcpy(imgSrcGPU, imgSrc, IMAGESIZE, cudaMemcpyHostToDevice);
	if (error != cudaSuccess)
	{
		printf("Error in cudaMemcpy imgSrc to imgSrcGpu: %d/n", error);
		return -1;
	}
	if ((flip != 'V') && (flip != 'H')) {
		printf("Invalid flip option '%c'. Must be 'V','H'... \n", flip);
		exit(EXIT_FAILURE);
	}
	gpuTimer.Start();
	// invoke kernels (define grid and block sizes)
	int rowBlock = (WIDTH + dimBlock - 1) / dimBlock;
	dimGrid = HEIGHT * rowBlock;
	if (flip == 'V')
	{
		VflipGPU <<<dimGrid, dimBlock >>>(imgDstGPU, imgSrcGPU, WIDTH, HEIGHT);
	}
	if (flip == 'H')
	{
		HflipGPU <<<dimGrid, dimBlock >>>(imgDstGPU, imgSrcGPU, WIDTH);
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	cudaDeviceSynchronize();

	gpuTimer.Stop();
	// Copy output (results) from GPU buffer to host (CPU) memory.
	cudaMemcpy(imgDst, imgDstGPU, IMAGESIZE, cudaMemcpyDeviceToHost);
	// Write the flipped image back to disk
	WriteBMPlin(imgDst, fileNameWrite);
	printf("\nKernel elapsed time %f ms \n\n", gpuTimer.Elapsed());

	// Deallocate CPU, GPU memory and destroy events.

	// cuda free vars
	error = cudaFree(imgSrcGPU);
	if (error != cudaSuccess)
	{
		printf("Error in CudaFree imgSrcGpu: %d/n", error);
		return -1;
	}
	error = cudaFree(imgDstGPU);
	if (error != cudaSuccess)
	{
		printf("Error in CudaFree imgDstGpu: %d/n", error);
		return -1;
	}
	free(imgSrc);
	free(imgDst);
	return (EXIT_SUCCESS);
}



// 0.945248 ms horizontal flip
// 1.357920 ms vertical flip