#include "filter.h"

int main(void)
{
	char inputFileName[256];
	char outputFileName[256];
	Image inputImage;
	Image outputImage;
	Pixel* dPixels;
	int filter[] = {-1, -2, -1, 1, 2, 1};

	scanf("%s", inputFileName);
	scanf("%s", outputFileName);

	imageReadFromFile(&inputImage, inputFileName);
	imageCreate(&outputImage, inputImage.width, inputImage.height);
	imageCreateTexture(&inputImage);

	int size = sizeof(Pixel) * inputImage.width * inputImage.height;

	ERR(cudaMalloc(&dPixels, size));
	ERR(cudaMemcpyToSymbol(g_filter, filter, sizeof(filter)));

	dim3 gridSize(32, 32);
	dim3 blockSize(32, 32);

	filterSobelKernel<<<gridSize, blockSize>>>(dPixels, inputImage.width, inputImage.height);

	ERR(cudaMemcpy(outputImage.pixels, dPixels, size, cudaMemcpyDeviceToHost));
	ERR(cudaFree(dPixels));

	imageWriteToFile(&outputImage, outputFileName);
	imageDeleteTexture();
	imageDelete(&outputImage);
	imageDelete(&inputImage);

	return 0;
}
