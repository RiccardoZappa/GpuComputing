#pragma once 
#include "cuda_runtime.h"

namespace utils
{
	class GpuTimer
	{
	public:
		GpuTimer();
		~GpuTimer();
		void Start();
		void Stop();
		float Elapsed();
	private:
		cudaEvent_t start;
		cudaEvent_t stop;
	};
}