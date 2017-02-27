#include "array.h"

void arrayRead(Array* arr)
{
	freopen(NULL, "rb", stdin);
	fread(&arr->size, sizeof(arr->size), 1, stdin);

	arr->data = (Byte*)malloc(sizeof(Byte) * arr->size);

	fread(arr->data, sizeof(Byte), arr->size, stdin);
	fclose(stdin);
}

void arrayWrite(Array* arr)
{
	freopen(NULL, "wb", stdout);
	fwrite(arr->data, sizeof(Byte), arr->size, stdout);
	free(arr->data);
	fclose(stdout);
}

void arraySort(Array* arr)
{
	Byte* dArr;
	int* dHist;
	int* dPrefix;

	int histSize = sizeof(int) * 256;

	ERR(cudaMalloc(&dArr, arr->size));
	ERR(cudaMalloc(&dHist, histSize));
	ERR(cudaMalloc(&dPrefix, histSize));
	ERR(cudaMemcpy(dArr, arr->data, arr->size, cudaMemcpyHostToDevice));
	ERR(cudaMemset(dHist, 0, histSize));

	histogramKernel<<<32, 32>>>(dArr, dHist, arr->size);

// BENCHMARK

	cudaEvent_t start;
	cudaEvent_t stop;
	float delta;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	scanKernel<<<1, 256>>>(dHist, dPrefix);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&delta, start, stop);

	fprintf(stderr, "Time: %f\n", delta);

// BENCHMARK END

// TEST
/*
	int hist[256];

	ERR(cudaMemcpy(&hist, dHist, histSize, cudaMemcpyDeviceToHost));

	fprintf(stderr, "Histogram:\n");

	for (int i = 0; i < 256; ++i)
		fprintf(stderr, "%d: %d\n", i, hist[i]);
*/
	int prefix[256];

	ERR(cudaMemcpy(&prefix, dPrefix, histSize, cudaMemcpyDeviceToHost));
	
	fprintf(stderr, "Prefix:\n");

	for (int i = 0; i < 256; ++i)
		fprintf(stderr, "%d: %d\n", i, prefix[i]);

// END TEST
	arrangementKernel<<<1, 256>>>(dArr, dHist, dPrefix);

	ERR(cudaMemcpy(arr->data, dArr, arr->size, cudaMemcpyDeviceToHost));
	ERR(cudaFree(dArr));
	ERR(cudaFree(dHist));
	ERR(cudaFree(dPrefix));
}

__global__ void histogramKernel(Byte* arr, int* hist, int arrCount)
{
	__shared__ int localHist[256];

	int tIdLocal = threadIdx.x;
	int tIdGlobal = blockDim.x * blockIdx.x + tIdLocal;
	int offsetX = gridDim.x * blockDim.x;

	for (int i = tIdGlobal; i < arrCount; i += offsetX)
		atomicAdd(&localHist[arr[i]], 1);

	__syncthreads();

	for (int i = tIdLocal; i < 256; i += blockDim.x)
		atomicAdd(&hist[i], localHist[i]);
}

__global__ void scanKernel(int* hist, int* prefix)
{
	__shared__ int temp[256];

	int tId = threadIdx.x;
	int offset = 1;

	temp[tId] = hist[tId];

	for (int d = 256 >> 1; d > 0; d >>= 1)
	{
		__syncthreads();

		if (tId < d)
		{
			int index1 = offset * (tId * 2 + 1) - 1;
			int index2 = offset * (tId * 2 + 2) - 1;

			temp[index2] += temp[index1];
		}

		offset <<= 1;
	}

	if (tId == 255)
		temp[255] = 0;

	for (int d = 1; d < 256; d <<= 1)
	{
		offset >>= 1;

		__syncthreads();

		if (tId < d)
		{
			int index1 = offset * (tId * 2 + 1) - 1;
			int index2 = offset * (tId * 2 + 2) - 1;
			int t = temp[index1];

			temp[index1] = temp[index2];
			temp[index2] += t;
		}
	}

	__syncthreads(); 
	
	if (tId < 255)
		prefix[tId] = temp[tId + 1];
	else
		prefix[tId] = temp[tId] + hist[255];
}

__global__ void arrangementKernel(Byte* arr, int* hist, int* prefix)
{
	int tId = threadIdx.x;

	while (hist[tId] > 0)
	{
		arr[prefix[tId] - 1] = (Byte)tId;
		--prefix[tId];
		--hist[tId];
	}
}
