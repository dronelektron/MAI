#include "gauss.h"

int main(void)
{
	Gauss gauss;
/*
	cudaEvent_t start;
	cudaEvent_t stop;
	float delta;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
*/
	gaussCreate(&gauss);
	gaussSolve(&gauss);	
	gaussPrintResult(&gauss);
	gaussDelete(&gauss);
/*
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&delta, start, stop);

	printf("Time: %f\n", delta);
*/
	return 0;
}
