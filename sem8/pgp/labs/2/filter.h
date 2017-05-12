#ifndef FILTER_H
#define FILTER_H

#include <math.h>
#include "image.h"

__device__ double filterGrayScale(Pixel* pixel);
__global__ void filterSobelKernel(Pixel* pixels, int w, int h);

__constant__ extern int g_filter[6];

#endif
