#include "classifier.h"

int main(void)
{
	char inputFileName[256];
	char outputFileName[256];
	Classifier classifier;
	Image image;
	Pixel* dPixels;

	scanf("%s", inputFileName);
	scanf("%s", outputFileName);

	imageReadFromFile(&image, inputFileName);

	classifierCreate(&classifier);
	classifierCalc(&classifier, &image);
	classifierCopyToConstant(&classifier);
	classifierDelete(&classifier);

	ERR(cudaMalloc(&dPixels, imageSize(&image)));
	ERR(cudaMemcpy(dPixels, image.pixels, imageSize(&image), cudaMemcpyHostToDevice));

	dim3 gridSize(32, 32);
	dim3 blockSize(32, 32);

	classifierMahalanobisKernel<<<gridSize, blockSize>>>(dPixels, image.width, image.height);

	ERR(cudaMemcpy(image.pixels, dPixels, imageSize(&image), cudaMemcpyDeviceToHost));
	ERR(cudaFree(dPixels));

	imageWriteToFile(&image, outputFileName);
	imageDelete(&image);

	return 0;
}
