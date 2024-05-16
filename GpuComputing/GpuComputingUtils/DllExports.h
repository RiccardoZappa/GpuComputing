#pragma once
#ifdef GPUCOMPUTINGUTILS_EXPORTS
	#define GPUCOMPUTINGUTILS_API __declspec(dllexport)
#else
    #define GPUCOMPUTINGUTILS_API	__declspec(dllimport)
#endif
