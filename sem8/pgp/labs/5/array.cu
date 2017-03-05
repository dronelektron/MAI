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
	Byte* dArrSrc;
	Byte* dArrRes;
	int* dHist;
	int* dPrefix;

	int histSize = sizeof(int) * BLOCK_SIZE;

	ERR(cudaMalloc(&dArrSrc, arr->size));
	ERR(cudaMalloc(&dArrRes, arr->size));
	ERR(cudaMalloc(&dHist, histSize));
	ERR(cudaMalloc(&dPrefix, histSize));
	ERR(cudaMemcpy(dArrSrc, arr->data, arr->size, cudaMemcpyHostToDevice));
	ERR(cudaMemset(dHist, 0, histSize));

	histogramKernel<<<32, 32>>>(dArrSrc, dHist, arr->size);
	scanKernel<<<1, BLOCK_SIZE>>>(dHist, dPrefix);
	arrangementKernel<<<32, 32>>>(dArrSrc, dArrRes, dPrefix, arr->size);

	ERR(cudaGetLastError());
	ERR(cudaMemcpy(arr->data, dArrRes, arr->size, cudaMemcpyDeviceToHost));
	ERR(cudaFree(dArrSrc));
	ERR(cudaFree(dArrRes));
	ERR(cudaFree(dHist));
	ERR(cudaFree(dPrefix));
}

__device__ int conflictFree(int index)
{
	return index + (index >> LOG2_BANKS);
}

__global__ void histogramKernel(Byte* arr, int* hist, int arrCount)
{
	__shared__ int temp[BLOCK_SIZE];

	int tIdLocal = threadIdx.x;
	int tIdGlobal = blockDim.x * blockIdx.x + tIdLocal;
	int offsetX = gridDim.x * blockDim.x;

	for (int i = tIdGlobal; i < arrCount; i += offsetX)
		atomicAdd(temp + arr[i], 1);

	__syncthreads();

	for (int i = tIdLocal; i < BLOCK_SIZE; i += blockDim.x)
		atomicAdd(hist + i, temp[i]);
}

__global__ void scanKernel(int* hist, int* prefix)
{
	__shared__ int temp[BLOCK_SIZE + BLOCK_SIZE / 32];

	int tId = threadIdx.x;
	int offset = 1;

	temp[conflictFree(tId)] = hist[tId];

	for (int d = BLOCK_SIZE >> 1; d > 0; d >>= 1)
	{
		__syncthreads();

		if (tId < d)
		{
			int index1 = conflictFree(offset * (tId * 2 + 1) - 1);
			int index2 = conflictFree(offset * (tId * 2 + 2) - 1);

			temp[index2] += temp[index1];
		}

		offset <<= 1;
	}

	if (tId == BLOCK_SIZE - 1)
		temp[conflictFree(BLOCK_SIZE - 1)] = 0;

	for (int d = 1; d < BLOCK_SIZE; d <<= 1)
	{
		offset >>= 1;

		__syncthreads();

		if (tId < d)
		{
			int index1 = conflictFree(offset * (tId * 2 + 1) - 1);
			int index2 = conflictFree(offset * (tId * 2 + 2) - 1);
			int t = temp[index1];

			temp[index1] = temp[index2];
			temp[index2] += t;
		}
	}

	__syncthreads(); 
	
	if (tId < BLOCK_SIZE - 1)
		prefix[tId] = temp[conflictFree(tId + 1)];
	else
		prefix[tId] = temp[conflictFree(tId)] + hist[BLOCK_SIZE - 1];
}

__global__ void arrangementKernel(Byte* arrSrc, Byte* arrRes, int* prefix, int arrCount)
{
	int tId = blockDim.x * blockIdx.x + threadIdx.x;
	int offsetX = gridDim.x * blockDim.x;

	for (int i = tId; i < arrCount; i += offsetX)
	{
		int pos = atomicSub(prefix + arrSrc[i], 1) - 1;

		arrRes[pos] = arrSrc[i];
	}
}
