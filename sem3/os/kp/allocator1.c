#include "allocator1.h"

int initList(size_t size)
{
	if (size < sizeof(BlockList))
		size = sizeof(BlockList);
	
	gBeginList = (BlockList*)malloc(size);
	
	if (gBeginList == NULL)
		return 0;

	gBeginList->size = size;
	gBeginList->prev = NULL;
	gBeginList->next = NULL;
	gFreeList = gBeginList;
	gSizeList = size;

	return 1;
}

void destroyList()
{
	free(gBeginList);
}

void* allocBlockList(BlockList* block, size_t size)
{
	BlockList* nextBlock = NULL;

	if (block->size >= size + sizeof(BlockList))
	{
		nextBlock = (BlockList*)((PBYTE_LIST)block + size);
		nextBlock->size = block->size - size;
		nextBlock->prev = block->prev;
		nextBlock->next = block->next;
		block->size = size;

		if (block->prev != NULL)
			block->prev->next = nextBlock;

		if (block->next != NULL)
			block->next->prev = nextBlock;

		if (block == gFreeList)
			gFreeList = nextBlock;
	}
	else
	{
		if (block->prev != NULL)
			block->prev->next = block->next;

		if (block->next != NULL)
			block->next->prev = block->prev;

		if (block == gFreeList)
			gFreeList = block->next;
	}

	return (void*)((PBYTE_LIST)block + sizeof(size_t));
}

void* mallocList(size_t size)
{
	size_t minSize = gSizeList;
	size_t oldSize = size;
	BlockList* minBlock = gFreeList;
	BlockList* cur = gFreeList;

	size += sizeof(size_t);

	if (size < sizeof(BlockList))
		size = sizeof(BlockList);

	while (cur != NULL)
	{
		if (cur->size < minSize && cur->size >= size)
		{
			minSize = cur->size;
			minBlock = cur;
		}

		cur = cur->next;
	}

	if (gFreeList == NULL || minBlock->size < size)
		return NULL;

	gReqList += oldSize;
	gTotList += size;
	
	return allocBlockList(minBlock, size);
}

void freeList(void* ptr)
{
	BlockList* block = (BlockList*)((PBYTE_LIST)ptr - sizeof(size_t));
	BlockList* cur = gFreeList;
	BlockList* leftBlock = NULL;
	BlockList* rightBlock = NULL;

	while (cur != NULL)
	{
		if ((BlockList*)((PBYTE_LIST)cur + cur->size) <= block)
			leftBlock = cur;

		if ((BlockList*)((PBYTE_LIST)block + block->size) <= cur)
		{
			rightBlock = cur;

			break;
		}

		cur = cur->next;
	}
	
	if (leftBlock != NULL)
		leftBlock->next = block;
	else
		gFreeList = block;
	
	if (rightBlock != NULL)
		rightBlock->prev = block;

	block->prev = leftBlock;
	block->next = rightBlock;
	cur = gFreeList;

	while (cur != NULL)
	{
		if ((BlockList*)((PBYTE_LIST)cur + cur->size) == cur->next)
		{
			cur->size += cur->next->size;
			cur->next = cur->next->next;

			if (cur->next != NULL)
				cur->next->prev = cur;

			continue;
		}

		cur = cur->next;
	}
}

size_t getReqList()
{
	return gReqList;
}

size_t getTotList()
{
	return gTotList;
}
