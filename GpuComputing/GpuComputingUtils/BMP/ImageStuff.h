#pragma once
#include "../DllExports.h"

typedef unsigned long ulong;
typedef unsigned int uint;
typedef unsigned char pel;

namespace BMP
{
	class GPUCOMPUTINGUTILS_API ImageStuff
	{
	public:
		struct Img_bmp
		{
			int Hpixels;
			int Vpixels;
			unsigned char HeaderInfo[54];
			unsigned long int Hbytes;
		};

		struct Pixel
		{
			unsigned char R;
			unsigned char G;
			unsigned char B;
		};

		ImageStuff() = default;
		~ImageStuff() = default;

		pel** ReadBMP(char*);
		void WriteBMP(pel**, char*);
	private:
		Img_bmp ip;
	};
}