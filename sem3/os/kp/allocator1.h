#ifndef ALLOCATOR_1_H
#define ALLOCATOR_1_H

#include <stdio.h>
#include <stdlib.h>

typedef unsigned char* PBYTE_LIST;

typedef struct _BlockList
{
	size_t size;
	struct _BlockList* prev;
	struct _BlockList* next;
} BlockList;

static BlockList* gBeginList;
static BlockList* gFreeList;
static size_t gSizeList;
static size_t gReqList = 0;
static size_t gTotList = 0;

int initList(size_t size);
void destroyList();
void* allocBlockList(BlockList* block, size_t size);
void* mallocList(size_t size);
void freeList(void* ptr);
size_t getReqList();
size_t getTotList();

#endif
