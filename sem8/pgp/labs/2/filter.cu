#include "filter.h"

__constant__ int g_filter[6];

__device__ double filterGrayScale(Pixel* pixel)
{
	return pixel->x * 0.299 + pixel->y * 0.587 + pixel->z * 0.114;
}

__global__ void filterSobelKernel(Pixel* pixels, int w, int h)
{
	int tY = blockIdx.y * blockDim.y + threadIdx.y;
	int tX = blockIdx.x * blockDim.x + threadIdx.x;
	int offsetY = gridDim.y * blockDim.y;
	int offsetX = gridDim.x * blockDim.x;

	for (int i = tY; i < h; i += offsetY)
	{
		for (int j = tX; j < w; j += offsetX)
		{
			double gx = 0.0;
			double gy = 0.0;
			Pixel pixel;

			for (int k = 0; k < 3; ++k)
			{
				int row = i + k - 1;
				int col0 = j - 1;
				int col2 = j + 1;
				int col = j + k - 1;
				int row0 = i - 1;
				int row2 = i + 1;

				pixel = tex2D(g_tex, col0, row);
				gx += g_filter[k] * filterGrayScale(&pixel);
				pixel = tex2D(g_tex, col2, row);
				gx += g_filter[k + 3] * filterGrayScale(&pixel);
				pixel = tex2D(g_tex, col, row0);
				gy += g_filter[k] * filterGrayScale(&pixel);
				pixel = tex2D(g_tex, col, row2);
				gy += g_filter[k + 3] * filterGrayScale(&pixel);
			}

			Byte gm = (Byte)min((int)sqrt(gx * gx + gy * gy), (int)0xFF);
			int offset = i * w + j;

			pixels[offset].x = gm;
			pixels[offset].y = gm;
			pixels[offset].z = gm;
			pixels[offset].w = 0;
		}
	}
}
