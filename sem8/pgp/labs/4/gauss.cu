#include "gauss.h"

void gaussCreate(Gauss* gauss)
{
	scanf("%d %d", &gauss->m, &gauss->n);

	gauss->rank = 0;
	gauss->mat = (double*)malloc(sizeof(double) * gauss->m * (gauss->n + 1));
	gauss->res = (double*)malloc(sizeof(double) * gauss->n);

	for (int i = 0; i < gauss->m; ++i)
		for (int j = 0; j < gauss->n; ++j)
			scanf("%lf", &gauss->mat[gaussOffset(i, j, gauss->m)]);

	for (int i = 0; i < gauss->m; ++i)
		scanf("%lf", &gauss->mat[gaussOffset(i, gauss->n, gauss->m)]);
}

void gaussDelete(Gauss* gauss)
{
	free(gauss->mat);
	free(gauss->res);
}

void gaussSolve(Gauss* gauss)
{
	int m = gauss->m;
	int n = gauss->n;
	double* dRatio;
	double* dMat;

	int matSize = sizeof(double) * m * (n + 1);

	ERR(cudaMalloc(&dRatio, sizeof(double) * m));
	ERR(cudaMalloc(&dMat, matSize));
	ERR(cudaMemcpy(dMat, gauss->mat, matSize, cudaMemcpyHostToDevice));

	thrust::device_ptr<double> dPtrBase = thrust::device_pointer_cast(dMat);

	for (int j = 0; j < n; ++j)
	{
		thrust::pair<thrust::device_ptr<double>, thrust::device_ptr<double> > dPtrMax = thrust::minmax_element(
			dPtrBase + gaussOffset(gauss->rank, j, m),
			dPtrBase + gaussOffset(m, j, m)
		);

		double maxVal1 = fabs(dPtrMax.first[0]);
		double maxVal2 = fabs(dPtrMax.second[0]);
		int maxInd = -1;

		if (maxVal1 > maxVal2)
			maxInd = (dPtrMax.first - dPtrBase) % m;
		else
			maxInd = (dPtrMax.second - dPtrBase) % m;

		if (fmax(maxVal1, maxVal2) < 1e-7)
		{
			gauss->res[j] = 0.0;

			continue;
		}
		else
			gauss->res[j] = 1.0;

		swapKernel<<<64, 128>>>(dMat, m, n, gauss->rank, maxInd);
		ratioKernel<<<64, 128>>>(dMat, dRatio, m, gauss->rank, j);
		transformKernel<<<dim3(64, 64), dim3(32, 32)>>>(dMat, dRatio, m, n, gauss->rank, j);

		++gauss->rank;

		if (gauss->rank == m)
			break;
	}

	ERR(cudaMemcpy(gauss->mat, dMat, matSize, cudaMemcpyDeviceToHost));
	ERR(cudaFree(dMat));
	ERR(cudaFree(dRatio));

	gaussBackward(gauss);
}

void gaussBackward(Gauss* gauss)
{
	int m = gauss->m;
	int n = gauss->n;
	int rank = gauss->rank - 1;
	double* mat = gauss->mat;
	double* res = gauss->res;

	for (int j = n - 1; j >= 0; --j)
	{
		if (fabs(res[j]) < 1e-7)
			continue;

		double sum = 0.0;

		for (int k = j + 1; k < n; ++k)
			sum += mat[gaussOffset(rank, k, m)] * res[k];

		res[j] = (mat[gaussOffset(rank, n, m)] - sum) / mat[gaussOffset(rank, j, m)];
		--rank;
	}
}

void gaussPrintResult(Gauss* gauss)
{
	for (int j = 0; j < gauss->n; ++j)
		printf("%.10e ", gauss->res[j]);

	printf("\n");
}

__host__ __device__ int gaussOffset(int row, int col, int m)
{
	return col * m + row;
}

__global__ void swapKernel(double* mat, int m, int n, int row1, int row2)
{
	int tX = blockDim.x * blockIdx.x + threadIdx.x + row1;
	int offsetX = gridDim.x * blockDim.x;

	while (tX <= n)
	{
		int offset1 = gaussOffset(row1, tX, m);
		int offset2 = gaussOffset(row2, tX, m);
		double tmp = mat[offset1];

		mat[offset1] = mat[offset2];
		mat[offset2] = tmp;
		tX += offsetX;
	}
}

__global__ void ratioKernel(double* mat, double* ratio, int m, int row, int col)
{
	int tX = blockDim.x * blockIdx.x + threadIdx.x + row + 1;
	int offsetX = gridDim.x * blockDim.x;

	while (tX < m)
	{
		ratio[tX] = mat[gaussOffset(tX, col, m)] / mat[gaussOffset(row, col, m)];
		tX += offsetX;
	}
}

__global__ void transformKernel(double* mat, double* ratio, int m, int n, int row, int col)
{
	int tX = blockDim.x * blockIdx.x + threadIdx.x + col;
	int tY = blockDim.y * blockIdx.y + threadIdx.y + row + 1;
	int offsetX = gridDim.x * blockDim.x;
	int offsetY = gridDim.y * blockDim.y;

	for (int j = tX; j <= n; j += offsetX)
		for (int i = tY; i < m; i += offsetY)
			mat[gaussOffset(i, j, m)] -= mat[gaussOffset(row, j, m)] * ratio[i];
}
