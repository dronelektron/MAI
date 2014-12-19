#include "allocator2.h"

void splitPageToBlocksA2(size_t pageIndex, size_t size)
{
	size_t i;
	size_t cnt = PAGE_SIZE_A2 / size;
	BlockA2* cur = (BlockA2*)((PBYTE_A2)gHeapA2 + pageIndex * PAGE_SIZE_A2);
	BlockA2* next = NULL;

	cur->prev = NULL;
	gPagesInfoA2[pageIndex].begin = cur;
	gPagesInfoA2[pageIndex].size = size;
	gPagesInfoA2[pageIndex].count = cnt;

	for (i = 1; i < cnt; ++i)
	{
		next = (BlockA2*)((PBYTE_A2)cur + size);
		cur->next = next;
		next->prev = cur;
		cur = cur->next;
	}

	cur->next = NULL;
}

int initAllocatorA2(size_t size)
{
	size_t i;

	gPagesCntA2 = size / PAGE_SIZE_A2;
	gPagesCntA2 += (size_t)(size % PAGE_SIZE_A2 != 0);
	gHeapA2 = malloc(gPagesCntA2 * PAGE_SIZE_A2);

	if (gHeapA2 == NULL)
		return 0;

	gPagesInfoA2 = (PageInfoA2*)malloc(sizeof(PageInfoA2) * gPagesCntA2);

	if (gPagesInfoA2 == NULL)
		return 0;

	for (i = 0; i < gPagesCntA2; ++i)
	{
		gPagesInfoA2[i].begin = NULL;
		gPagesInfoA2[i].size = FREE;
		gPagesInfoA2[i].count = 0;
	}

	return 1;
}

void destroyAllocatorA2()
{
	free(gHeapA2);
	free(gPagesInfoA2);
}

void* mallocA2(size_t size)
{
	size_t i;
	size_t freePage = -1;
	size_t sizeA = sizeof(BlockA2);
	size_t sizeB = (size_t)pow(2.0, ceil(log(size) / log(2.0)));
	BlockA2* cur = NULL;
	
	size = sizeA > sizeB ? sizeA : sizeB;

	for (i = 0; i < gPagesCntA2; ++i)
	{
		if ((gPagesInfoA2[i].size == size && gPagesInfoA2[i].count > 0) || gPagesInfoA2[i].size == FREE)
		{
			freePage = i;

			break;
		}
	}

	if (freePage == -1)
		return NULL;

	if (gPagesInfoA2[freePage].size == FREE)
		splitPageToBlocksA2(freePage, size);
	
	cur = gPagesInfoA2[freePage].begin;

	if (cur->next != NULL)
		cur->next->prev = NULL;

	gPagesInfoA2[freePage].begin = cur->next;
	--gPagesInfoA2[freePage].count;

	return (void*)cur;
}

void freeA2(void* ptr)
{
	BlockA2* left = NULL;
	BlockA2* right = NULL;
	BlockA2* cur = NULL;
	BlockA2* block = (BlockA2*)ptr;
	size_t pageIndex = (ptr - gHeapA2) / PAGE_SIZE_A2;
	size_t blockSize = gPagesInfoA2[pageIndex].size;
	
	if (gPagesInfoA2[pageIndex].count == 0)
	{
		block->prev = NULL;
		block->next = NULL;
		gPagesInfoA2[pageIndex].begin = block;
		gPagesInfoA2[pageIndex].count = 1;
	}
	else
	{
		cur = gPagesInfoA2[pageIndex].begin;

		while (cur != NULL)
		{
			if ((BlockA2*)((PBYTE_A2)cur + blockSize) <= block)
				left = cur;

			if ((BlockA2*)((PBYTE_A2)block + blockSize) <= cur)
			{
				right = cur;

				break;
			}

			cur = cur->next;
		}

		if (left != NULL)
			left->next = block;
		else
			gPagesInfoA2[pageIndex].begin = block;

		if (right != NULL)
			right->prev = block;

		block->prev = left;
		block->next = right;

		++gPagesInfoA2[pageIndex].count;
	}

	if (gPagesInfoA2[pageIndex].size * gPagesInfoA2[pageIndex].count == PAGE_SIZE_A2)
	{
		gPagesInfoA2[pageIndex].begin = NULL;
		gPagesInfoA2[pageIndex].size = FREE;
		gPagesInfoA2[pageIndex].count = 0;
	}
}
