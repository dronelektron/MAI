#include "allocator1.h"

int initAllocatorA1(size_t size)
{
	if (size < sizeof(BlockA1))
		size = sizeof(BlockA1);
	
	gBeginA1 = (BlockA1*)malloc(size);
	
	if (gBeginA1 == NULL)
		return 0;

	gBeginA1->size = size;
	gBeginA1->prev = NULL;
	gBeginA1->next = NULL;
	gFreeA1 = gBeginA1;
	gSizeA1 = size;

	return 1;
}

void destroyAllocatorA1()
{
	free(gBeginA1);
}

void deallocBlockA1(BlockA1* left, BlockA1* block, BlockA1* right)
{
	if (left == NULL && right == NULL)
	{
		gFreeA1 = block;
		gFreeA1->prev = NULL;
		gFreeA1->next = NULL;
	}
	else if (left != NULL && right == NULL)
	{
		left->next = block;
		block->prev = left;
		block->next = NULL;
	}
	else if (left == NULL && right != NULL)
	{
		block->prev = NULL;
		block->next = right;
		right->prev = block;
		gFreeA1 = block;
	}
	else
	{
		left->next = block;
		right->prev = block;
		block->prev = left;
		block->next = right;
	}
}

void* allocBlockA1(BlockA1* block, size_t size)
{
	BlockA1* nextBlock = NULL;

	if (block->size >= size + sizeof(BlockA1))
	{
		nextBlock = (BlockA1*)((PBYTE_A1)block + size);
		nextBlock->size = block->size - size;
		nextBlock->prev = block->prev;
		nextBlock->next = block->next;
		block->size = size;

		if (block->prev != NULL)
			block->prev->next = nextBlock;

		if (block->next != NULL)
			block->next->prev = nextBlock;

		if (block == gFreeA1)
			gFreeA1 = nextBlock;
	}
	else
	{
		if (block->prev != NULL)
			block->prev->next = block->next;

		if (block->next != NULL)
			block->next->prev = block->prev;

		if (block == gFreeA1)
			gFreeA1 = block->next;
	}

	return (void*)((PBYTE_A1)block + sizeof(size_t));
}

void* mallocA1(size_t size)
{
	size_t minSize = gSizeA1;
	size_t oldSize = size;
	BlockA1* minBlock = gFreeA1;
	BlockA1* cur = gFreeA1;

	size += sizeof(size_t);

	if (size < sizeof(BlockA1))
		size = sizeof(BlockA1);

	while (cur != NULL)
	{
		if (cur->size < minSize && cur->size >= size)
		{
			minSize = cur->size;
			minBlock = cur;
		}

		cur = cur->next;
	}

	if (gFreeA1 == NULL || minBlock->size < size)
		return NULL;

	gReqA1 += oldSize;
	gTotA1 += size;

	//printf("Request: %zu bytes\n", oldSize);
	//printf("Allocated: %zu bytes\n", size);
	
	return allocBlockA1(minBlock, size);
}

void freeA1(void* ptr)
{
	BlockA1* block = (BlockA1*)((PBYTE_A1)ptr - sizeof(size_t));
	BlockA1* cur = gFreeA1;
	BlockA1* leftBlock = NULL;
	BlockA1* rightBlock = NULL;

	while (cur != NULL)
	{
		if ((BlockA1*)((PBYTE_A1)cur + cur->size) <= block)
			leftBlock = cur;

		if ((BlockA1*)((PBYTE_A1)block + block->size) <= cur)
		{
			rightBlock = cur;

			break;
		}

		cur = cur->next;
	}
	
	deallocBlockA1(leftBlock, block, rightBlock);

	cur = gFreeA1;

	while (cur != NULL)
	{
		if ((BlockA1*)((PBYTE_A1)cur + cur->size) == cur->next)
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

size_t getReqA1()
{
	return gReqA1;
}

size_t getTotA1()
{
	return gTotA1;
}
