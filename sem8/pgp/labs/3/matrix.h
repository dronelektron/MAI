#ifndef MATRIX_H
#define MATRIX_H

#include "types.h"

typedef struct _Matrix13
{
	double cells[3];
} Matrix13;

typedef struct _Matrix33
{
	double cells[3][3];
} Matrix33;

__host__ __device__ void matrixZeros(Matrix13* mat);
__host__ __device__ void matrixZeros(Matrix33* mat);
__host__ __device__ void matrixIdentity(Matrix33* mat);
__host__ __device__ void matrixFromPixel(Matrix13* matrix, Pixel* pixel);
__host__ __device__ void matrixInverse(Matrix33* inMat, Matrix33* outMat);
__host__ __device__ void matrixAdd(Matrix13* inMatA, Matrix13* inMatB, Matrix13* outMat);
__host__ __device__ void matrixAdd(Matrix33* inMatA, Matrix33* inMatB, Matrix33* outMat);
__host__ __device__ void matrixSub(Matrix13* inMatA, Matrix13* inMatB, Matrix13* outMat);
__host__ __device__ void matrixMul(Matrix13* inMatA, Matrix13* inMatB, Matrix33* outMat);
__host__ __device__ void matrixMul(Matrix13* inMatA, Matrix33* inMatB, Matrix13* outMat);
__host__ __device__ void matrixMul(Matrix33* inMatA, Matrix33* inMatB, Matrix33* outMat);
__host__ __device__ void matrixMul(Matrix13* inMat, double value, Matrix13* outMat);
__host__ __device__ void matrixMul(Matrix33* inMat, double value, Matrix33* outMat);
__host__ __device__ double matrixMul(Matrix13* inMatA, Matrix13* inMatB);

#endif
