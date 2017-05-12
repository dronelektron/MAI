#ifndef ERR_H
#define ERR_H

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

#endif
