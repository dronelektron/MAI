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
	scanKernel<<<1, 256>>>(dHist, dPrefix);
	arrangementKernel<<<8, 32>>>(dArr, dHist, dPrefix);

	ERR(cudaMemcpy(arr->data, dArr, arr->size, cudaMemcpyDeviceToHost));
	ERR(cudaFree(dArr));
	ERR(cudaFree(dHist));
	ERR(cudaFree(dPrefix));
}

__device__ int arrayConflictFree(int index)
{
	return index + (index >> 5);
}

__global__ void histogramKernel(Byte* arr, int* hist, int arrCount)
{
	__shared__ int temp[256];

	int tIdLocal = threadIdx.x;
	int tIdGlobal = blockDim.x * blockIdx.x + tIdLocal;
	int offsetX = gridDim.x * blockDim.x;

	for (int i = tIdGlobal; i < arrCount; i += offsetX)
		atomicAdd(&temp[arr[i]], 1);

	__syncthreads();

	for (int i = tIdLocal; i < 256; i += blockDim.x)
		atomicAdd(&hist[i], temp[i]);
}

__global__ void scanKernel(int* hist, int* prefix)
{
	__shared__ int temp[256 + 6];

	int tId = threadIdx.x;
	int offset = 1;

	temp[arrayConflictFree(tId)] = hist[tId];

	for (int d = 256 >> 1; d > 0; d >>= 1)
	{
		__syncthreads();

		if (tId < d)
		{
			int index1 = arrayConflictFree(offset * (tId * 2 + 1) - 1);
			int index2 = arrayConflictFree(offset * (tId * 2 + 2) - 1);

			temp[index2] += temp[index1];
		}

		offset <<= 1;
	}

	if (tId == 255)
		temp[arrayConflictFree(255)] = 0;

	for (int d = 1; d < 256; d <<= 1)
	{
		offset >>= 1;

		__syncthreads();

		if (tId < d)
		{
			int index1 = arrayConflictFree(offset * (tId * 2 + 1) - 1);
			int index2 = arrayConflictFree(offset * (tId * 2 + 2) - 1);
			int t = temp[index1];

			temp[index1] = temp[index2];
			temp[index2] += t;
		}
	}

	__syncthreads(); 
	
	if (tId < 255)
		prefix[tId] = temp[arrayConflictFree(tId + 1)];
	else
		prefix[tId] = temp[arrayConflictFree(tId)] + hist[255];
}

__global__ void arrangementKernel(Byte* arr, int* hist, int* prefix)
{
	__shared__ int temp[256 * 2];

	int tId = blockDim.x * blockIdx.x + threadIdx.x;

	temp[tId] = hist[tId];
	temp[tId + 256] = prefix[tId];	

	while (temp[tId] > 0)
	{
		arr[temp[tId + 256] - 1] = (Byte)tId;
		--temp[tId + 256];
		--temp[tId];
	}
}
