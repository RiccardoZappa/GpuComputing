#pragma once

typedef unsigned long ulong;
typedef unsigned int uint;
typedef unsigned char pel;

namespace BMP
{
	class ImageStuff
	{
	public:
		typedef struct Img_bmp
		{
			int Hpixels;
			int Vpixels;
			unsigned char HeaderInfo[54];
			unsigned long int Hbytes;
		};

		typedef struct Pixel
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