#pragma once
#include "../DllExports.h"

#ifndef EASYPPM_H_
#define EASYPPM_H_

#ifdef __cplusplus
extern "C" {
#endif

#define EASYPPM_NUM_CHANNELS 3
#define EASYPPM_MAX_CHANNEL_VALUE 255

#define PPMBYTE unsigned char

    typedef struct {
        PPMBYTE r;
        PPMBYTE g;
        PPMBYTE b;
    } ppmcolor;

    typedef enum {
        IMAGETYPE_PBM,
        IMAGETYPE_PGM,
        IMAGETYPE_PPM
    } imagetype;

    typedef struct {
        int       width;
        int       height;
        PPMBYTE* image;
        imagetype itype;
    } PPM;

    GPUCOMPUTINGUTILS_API PPM      easyppm_create(int width, int height, imagetype itype);
    GPUCOMPUTINGUTILS_API void     easyppm_clear(PPM* ppm, ppmcolor c);
    GPUCOMPUTINGUTILS_API void     easyppm_set(PPM* ppm, int x, int y, ppmcolor c);
    GPUCOMPUTINGUTILS_API ppmcolor easyppm_get(PPM* ppm, int x, int y);
    GPUCOMPUTINGUTILS_API ppmcolor easyppm_rgb(PPMBYTE r, PPMBYTE g, PPMBYTE b);
    GPUCOMPUTINGUTILS_API ppmcolor easyppm_grey(PPMBYTE gr);
    GPUCOMPUTINGUTILS_API ppmcolor easyppm_black_white(int bw);
    GPUCOMPUTINGUTILS_API void     easyppm_gamma_correct(PPM* ppm, float gamma);
    GPUCOMPUTINGUTILS_API void     easyppm_invert_y(PPM* ppm);
    GPUCOMPUTINGUTILS_API void     easyppm_read(PPM* ppm, const char* path);
    GPUCOMPUTINGUTILS_API void     easyppm_write(PPM* ppm, const char* path);
    GPUCOMPUTINGUTILS_API void     easyppm_destroy(PPM* ppm);

#ifdef __cplusplus
}
#endif

#endif