#ifndef ALLOCATOR_2_H
#define ALLOCATOR_2_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef unsigned char* PBYTE_A2;

typedef enum _kMemStateA2
{
	LINK = -1,
	FREE,
} kMemState;

typedef struct _BlockA2
{
	struct _BlockA2* prev;
	struct _BlockA2* next;
} BlockA2;

typedef struct _PageInfoA2
{
	BlockA2* begin;
	int size;
	size_t count;
} PageInfoA2;

static const size_t PAGE_SIZE_A2 = 1024;
static size_t gPagesCntA2 = 0;
static void* gHeapA2 = NULL;
static PageInfoA2* gPagesInfoA2 = NULL;

size_t getPageCountBySize(size_t size);
void splitPageToBlocksA2(size_t pageIndex, size_t size);
void linkPagesA2(size_t pageIndex, size_t count);
void unlinkPagesA2(size_t pageIndex);
int initAllocatorA2(size_t size);
void destroyAllocatorA2();
void* mallocA2(size_t size);
void freeA2(void* ptr);

#endif
