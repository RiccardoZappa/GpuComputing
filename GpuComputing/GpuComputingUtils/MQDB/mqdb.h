
#include <cstdio>
#include <cstdlib>
#include <string.h>
#include <math.h>
#include "../DllExports.h"


#define randu() ((float)rand() / (float) RAND_MAX)
#define abs(x) ((x)<0 ? (-x) : (x))

typedef unsigned long ulong;
typedef unsigned int uint;

namespace MQDB
{
	class GPUCOMPUTINGUTILS_API Matrix
	{
	public:
		typedef struct MQDB {
			char desc[100];   // description
			int nBlocks;      // num. of blocks
			int* blkSize;     // block dimensions
			float* elem;       // elements in row-major order
			ulong nElems;     // actual number of elements
		} mqdb;
	public:
		Matrix() = default;
		~Matrix() = default;

		// function prototypes
		int genRandDims(mqdb*, uint, uint, int);
		int genRandDimsUnified(mqdb*, uint, uint, int);
		void fillBlocks(mqdb*, uint, uint, char, float);
		void fillBlocksUnified(mqdb*, uint, uint, char, float);
		mqdb mqdbConst(uint, uint, uint, float);
		void mqdbProd(mqdb, mqdb, mqdb);
		void matProd(mqdb, mqdb, mqdb);
		void checkResult(mqdb, mqdb);
		void mqdbDisplay(mqdb*);
	};

}