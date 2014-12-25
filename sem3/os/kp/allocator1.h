#ifndef ALLOCATOR_1_H
#define ALLOCATOR_1_H

#include <stdio.h>
#include <stdlib.h>

typedef unsigned char* PBYTE_A1;

typedef struct _BlockA1
{
	size_t size;
	struct _BlockA1* prev;
	struct _BlockA1* next;
} BlockA1;

static BlockA1* gBeginA1;
static BlockA1* gFreeA1;
static size_t gSizeA1;
static size_t gReqA1 = 0;
static size_t gTotA1 = 0;

int initAllocatorA1(size_t size);
void destroyAllocatorA1();
void deallocBlockA1(BlockA1* left, BlockA1* block, BlockA1* right);
void* allocBlockA1(BlockA1* block, size_t size);
void* mallocA1(size_t size);
void freeA1(void* ptr);
size_t getReqA1();
size_t getTotA1();

#endif
