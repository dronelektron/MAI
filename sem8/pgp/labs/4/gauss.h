#ifndef GAUSS_H
#define GAUSS_H

#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "err.h"

typedef struct _Gauss
{
	int m;
	int n;
	int rank;
	double* mat;
	double* res;
} Gauss;

void gaussCreate(Gauss* gauss);
void gaussDelete(Gauss* gauss);
void gaussSolve(Gauss* gauss);
void gaussBackward(Gauss* gauss);
void gaussPrintResult(Gauss* gauss);
__host__ __device__ int gaussOffset(int row, int col, int m);

__global__ void swapKernel(double* mat, int m, int n, int row1, int row2);
__global__ void ratioKernel(double* mat, double* ratio, int m, int row, int col);
__global__ void transformKernel(double* mat, double* ratio, int m, int n, int row, int col);

#endif
