#include <stdio.h>

int main(void)
{
	int deviceCoint;
	cudaDeviceProp devProp;

	cudaGetDeviceCount(&deviceCoint);

	printf("Found %d devices\n", deviceCoint);

	for (int device = 0; device < deviceCoint; ++device)
	{
		cudaGetDeviceProperties(&devProp, device);

		printf("Device: %d\n", device);
		printf("Compute capability: %d.%d\n", devProp.major, devProp.minor);
		printf("Name: %s\n", devProp.name);
		printf("Total global memory: %li\n", devProp.totalGlobalMem);
		printf("Shared memory per block: %li\n", devProp.sharedMemPerBlock);
		printf("Registers per block: %d\n", devProp.regsPerBlock);
		printf("Warp size: %d\n", devProp.warpSize);
		printf("Max threads per block: %d\n", devProp.maxThreadsPerBlock);
		printf("Total constant memory: %li\n", devProp.totalConstMem);
		printf("Clock rate: %d\n", devProp.clockRate);
		printf("Texture alignment: %lu\n", devProp.textureAlignment);
		printf("Device overlap: %d\n", devProp.deviceOverlap);
		printf("Multiprocessor count: %d\n", devProp.multiProcessorCount);
		printf("Max threads dim: %d %d %d\n", devProp.maxThreadsDim[0], devProp.maxThreadsDim[1], devProp.maxThreadsDim[2]);
		printf("Max grid size: %d %d %d\n", devProp.maxGridSize[0], devProp.maxGridSize[1], devProp.maxGridSize[2]);
	}

	return 0;
}
