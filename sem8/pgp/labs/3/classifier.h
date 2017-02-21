#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include "image.h"
#include "matrix.h"
#include "err.h"

typedef struct _Coord
{
	int x;
	int y;
} Coord;

typedef struct _Class
{
	int size;
	Coord* coords;
} Class;

typedef struct _Classifier
{
	int size;
	Class* classes;
	Matrix13* avgs;
	Matrix33* covs;
} Classifier;

void classifierCreate(Classifier* classifier);
void classifierDelete(Classifier* classifier);
void classifierCalc(Classifier* classifier, Image* image);
void classifierCopyToConstant(Classifier* classifier);

__device__ double classifierFunc(Pixel* pixel, int classInd);
__global__ void classifierMahalanobisKernel(Pixel* pixels, int w, int h);

__constant__ extern int g_nc;
__constant__ extern Matrix13 g_avgs[32];
__constant__ extern Matrix33 g_covs[32];

#endif
