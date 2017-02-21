#include <stdio.h>
#include <stdlib.h>

#define ERR(call) \
{ \
	cudaError_t err = call; \
	\
	if (err != cudaSuccess) \
	{ \
		fprintf(stderr, "ERROR: CUDA failed in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
		exit(0); \
	} \
}

__global__ void kernel(double* dA, double* dB, double* dC, int n);

int main(void)
{
	int n;
	int size;
	double* hA;
	double* hB;
	double* hC;
	double* dA;
	double* dB;
	double* dC;

	scanf("%d", &n);

	size = sizeof(double) * n;
	hA = (double*)malloc(size);
	hB = (double*)malloc(size);
	hC = (double*)malloc(size);

	for (int i = 0; i < n; ++i)
		scanf("%lf", &hA[i]);

	for (int i = 0; i < n; ++i)
		scanf("%lf", &hB[i]);

	ERR(cudaMalloc(&dA, size));
	ERR(cudaMalloc(&dB, size));
	ERR(cudaMalloc(&dC, size));
	ERR(cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice));
	ERR(cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice));

	kernel<<<256, 256>>>(dA, dB, dC, n);

	ERR(cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost));

	for (int i = 0; i < n; ++i)
		printf("%.10e ", hC[i]);

	printf("\n");

	ERR(cudaFree(dC));
	ERR(cudaFree(dB));
	ERR(cudaFree(dA));
	free(hC);
	free(hB);
	free(hA);

	return 0;
}

__global__ void kernel(double* dA, double* dB, double* dC, int n)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = gridDim.x * blockDim.x;

	while (idx < n)
	{
		dC[idx] = dA[idx] * dB[idx];
		idx += offset;
	}
}
