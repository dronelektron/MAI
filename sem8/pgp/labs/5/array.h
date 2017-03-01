#ifndef ARRAY_H
#define ARRAY_H

#include <stdio.h>
#include <stdlib.h>
#include "err.h"

typedef unsigned char Byte;

typedef struct _Array
{
	int size;
	Byte* data;
} Array;

void arrayRead(Array* arr);
void arrayWrite(Array* arr);
void arraySort(Array* arr);
__device__ int arrayConflictFree(int index);

__global__ void histogramKernel(Byte* arr, int* hist, int arrCount);
__global__ void scanKernel(int* hist, int* prefix);
__global__ void arrangementKernel(Byte* arr, int* hist, int* prefix);

#endif
