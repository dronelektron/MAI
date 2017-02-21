#include "matrix.h"

__host__ __device__ void matrixZeros(Matrix13* mat)
{
	for (int i = 0; i < 3; ++i)
		mat->cells[i] = 0.0;
}

__host__ __device__ void matrixZeros(Matrix33* mat)
{
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			mat->cells[i][j] = 0.0;
}

__host__ __device__ void matrixIdentity(Matrix33* mat)
{
	matrixZeros(mat);

	for (int i = 0; i < 3; ++i)
		mat->cells[i][i] = 1.0;
}

__host__ __device__ void matrixFromPixel(Matrix13* matrix, Pixel* pixel)
{
	matrix->cells[0] = pixel->x;
	matrix->cells[1] = pixel->y;
	matrix->cells[2] = pixel->z;
}

__host__ __device__ void matrixInverse(Matrix33* inMat, Matrix33* outMat)
{
	Matrix33 matA = *inMat;
	Matrix33 matE;

	matrixIdentity(&matE);

	for (int j = 0; j < 3; ++j)
	{
		for (int i = 0; i < 3; ++i)
		{
			if (i == j)
			{
				double ratio = matA.cells[j][j];

				for (int k = 0; k < 3; ++k)
				{
					matA.cells[i][k] /= ratio;
					matE.cells[i][k] /= ratio;
				}
			}
			else
			{
				double ratio = matA.cells[i][j] / matA.cells[j][j];

				for (int k = 0; k < 3; ++k)
				{
					matA.cells[i][k] -= matA.cells[j][k] * ratio;
					matE.cells[i][k] -= matE.cells[j][k] * ratio;
				}
			}
		}
	}

	*outMat = matE;
}

__host__ __device__ void matrixAdd(Matrix13* inMatA, Matrix13* inMatB, Matrix13* outMat)
{
	for (int i = 0; i < 3; ++i)
		outMat->cells[i] = inMatA->cells[i] + inMatB->cells[i];
}

__host__ __device__ void matrixAdd(Matrix33* inMatA, Matrix33* inMatB, Matrix33* outMat)
{
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			outMat->cells[i][j] = inMatA->cells[i][j] + inMatB->cells[i][j];
}

__host__ __device__ void matrixSub(Matrix13* inMatA, Matrix13* inMatB, Matrix13* outMat)
{
	for (int i = 0; i < 3; ++i)
		outMat->cells[i] = inMatA->cells[i] - inMatB->cells[i];
}

__host__ __device__ void matrixMul(Matrix13* inMatA, Matrix13* inMatB, Matrix33* outMat)
{
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			outMat->cells[i][j] = inMatA->cells[i] * inMatB->cells[j];
}

__host__ __device__ void matrixMul(Matrix13* inMatA, Matrix33* inMatB, Matrix13* outMat)
{
	matrixZeros(outMat);

	for (int j = 0; j < 3; ++j)
		for (int k = 0; k < 3; ++k)
			outMat->cells[j] += inMatA->cells[k] * inMatB->cells[k][j];
}

__host__ __device__ void matrixMul(Matrix33* inMatA, Matrix33* inMatB, Matrix33* outMat)
{
	matrixZeros(outMat);

	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			for (int k = 0; k < 3; ++k)
				outMat->cells[i][j] += inMatA->cells[i][k] * inMatB->cells[k][j];
}

__host__ __device__ void matrixMul(Matrix13* inMat, double value, Matrix13* outMat)
{
	for (int i = 0; i < 3; ++i)
		outMat->cells[i] = inMat->cells[i] * value;
}

__host__ __device__ void matrixMul(Matrix33* inMat, double value, Matrix33* outMat)
{
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			outMat->cells[i][j] = inMat->cells[i][j] * value;
}

__host__ __device__ double matrixMul(Matrix13* inMatA, Matrix13* inMatB)
{
	double res = 0.0;

	for (int k = 0; k < 3; ++k)
		res += inMatA->cells[k] * inMatB->cells[k];

	return res;
}
