#ifndef ALLOCATOR_2_H
#define ALLOCATOR_2_H

#include <stdio.h>
#include <stdlib.h>

typedef unsigned char* PBYTE_MKK;

typedef enum _kMemState
{
	FREE = 0
} kMemState;

typedef struct _BlockMKK
{
	struct _BlockMKK* next;
} BlockMKK;

static const size_t PAGE_SIZE_MKK = 4096;
static void* gHeapMKK = NULL;
static size_t* gMemsizeMKK = NULL;
static BlockMKK** gListMKK = NULL;
static size_t gPagesCntMKK = 0;
static size_t gPowCntMKK = 0;
static size_t gPowIndexMin = 0;
static size_t gReqMKK = 0;
static size_t gTotMKK = 0;

int initMKK(size_t size);
void destroyMKK();
void* mallocMKK(size_t size);
void freeMKK(void* ptr);
BlockMKK* allocPageMKK(size_t size);
void freePageMKK(BlockMKK* block);
void splitPageMMK(BlockMKK* block, size_t powIndex);
size_t powOfSizeMKK(size_t size);
size_t getPagesCountMKK(size_t size);
size_t getPageIndexMKK(BlockMKK* block);
size_t getReqMKK();
size_t getTotMKK();

#endif
