#include "classifier.h"

__constant__ int g_nc;
__constant__ Matrix13 g_avgs[32];
__constant__ Matrix33 g_covs[32];

void classifierCreate(Classifier* classifier)
{
	int nc;

	scanf("%d", &nc);

	classifier->size = nc;
	classifier->classes = (Class*)malloc(sizeof(Class) * nc);
	classifier->avgs = (Matrix13*)malloc(sizeof(Matrix13) * nc);
	classifier->covs = (Matrix33*)malloc(sizeof(Matrix33) * nc);

	for (int i = 0; i < nc; ++i)
	{
		int np;
		Class* cls = &classifier->classes[i];

		scanf("%d", &np);

		cls->size = np;
		cls->coords = (Coord*)malloc(sizeof(Coord) * np);

		for (int j = 0; j < np; ++j)
		{
			Coord* coord = &cls->coords[j];

			scanf("%d %d", &coord->x, &coord->y);
		}
	}
}

void classifierDelete(Classifier* classifier)
{
	for (int i = 0; i < classifier->size; ++i)
		free(classifier->classes[i].coords);

	free(classifier->covs);
	free(classifier->avgs);
	free(classifier->classes);
}

void classifierCalc(Classifier* classifier, Image* image)
{
	for (int i = 0; i < classifier->size; ++i)
	{
		Matrix13 matPixel;
		Matrix13* avg = &classifier->avgs[i];
		Matrix33* cov = &classifier->covs[i];
		Class* cls = &classifier->classes[i];

		matrixZeros(avg);
		matrixZeros(cov);

		for (int j = 0; j < cls->size; ++j)
		{
			Coord* coords = &classifier->classes[i].coords[j];
			Pixel pixel = imageGetPixel(image, coords->y, coords->x);

			matrixFromPixel(&matPixel, &pixel);
			matrixAdd(avg, &matPixel, avg);
		}

		matrixMul(avg, 1.0 / cls->size, avg);

		for (int j = 0; j < cls->size; ++j)
		{
			Coord* coords = &classifier->classes[i].coords[j];
			Pixel pixel = imageGetPixel(image, coords->y, coords->x);
			Matrix13 matDiff;
			Matrix33 matCross;

			matrixFromPixel(&matPixel, &pixel);
			matrixSub(&matPixel, avg, &matDiff);
			matrixMul(&matDiff, &matDiff, &matCross);
			matrixAdd(cov, &matCross, cov);
		}

		if (cls->size > 1)
			matrixMul(cov, 1.0 / (cls->size - 1), cov);
	}
}

void classifierCopyToConstant(Classifier* classifier)
{
	ERR(cudaMemcpyToSymbol(g_nc, &classifier->size, sizeof(classifier->size)));

	for (int i = 0; i < classifier->size; ++i)
	{
		int size13 = sizeof(Matrix13);
		int size33 = sizeof(Matrix33);

		matrixInverse(&classifier->covs[i], &classifier->covs[i]);

		ERR(cudaMemcpyToSymbol(g_avgs, &classifier->avgs[i], size13, i * size13));
		ERR(cudaMemcpyToSymbol(g_covs, &classifier->covs[i], size33, i * size33))
	}
}

__device__ double classifierFunc(Pixel* pixel, int classInd)
{
	Matrix13 matPixel;
	Matrix13 matDiff;
	Matrix13 matRes;

	matrixFromPixel(&matPixel, pixel);
	matrixSub(&matPixel, &g_avgs[classInd], &matDiff);
	matrixMul(&matDiff, &g_covs[classInd], &matRes);

	return -matrixMul(&matRes, &matDiff);
}

__global__ void classifierMahalanobisKernel(Pixel* pixels, int w, int h)
{
	int tY = blockIdx.y * blockDim.y + threadIdx.y;
	int tX = blockIdx.x * blockDim.x + threadIdx.x;
	int offsetY = gridDim.y * blockDim.y;
	int offsetX = gridDim.x * blockDim.x;

	for (int i = tY; i < h; i += offsetY)
	{
		for (int j = tX; j < w; j += offsetX)
		{
			Pixel* pixel = &pixels[i * w + j];
			double maxVal = classifierFunc(pixel, 0);
			int maxInd = 0;

			for (int c = 1; c < g_nc; ++c)
			{
				double tmpVal = classifierFunc(pixel, c);

				if (tmpVal > maxVal)
				{
					maxVal = tmpVal;
					maxInd = c;
				}
			}

			pixel->w = (Byte)maxInd;
		}
	}
}
