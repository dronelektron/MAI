#include "allocator1.h"

int initAllocatorA1(size_t size)
{
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

void insertBlockA1(BlockA1* left, BlockA1* block, BlockA1* right)
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

void* eraseBlockA1(BlockA1* block, size_t size)
{
	BlockA1* nextBlock = NULL;

	if (block->size >= size + MIN_BLOCK_SIZE_A1)
	{
		nextBlock = (BlockA1*)((unsigned char*)block + size);
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

	return (void*)((unsigned char*)block + sizeof(size_t));
}

void* mallocA1(size_t size)
{
	size_t minSize = gSizeA1;
	BlockA1* minBlock = gFreeA1;
	BlockA1* cur = gFreeA1;
	
	size += sizeof(size_t);

	if (size < MIN_BLOCK_SIZE_A1)
		size = MIN_BLOCK_SIZE_A1;

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
	
	return eraseBlockA1(minBlock, size);
}

void freeA1(void* ptr)
{
	BlockA1* block = (BlockA1*)((unsigned char*)ptr - sizeof(size_t));
	BlockA1* cur = gFreeA1;
	BlockA1* leftBlock = NULL;
	BlockA1* rightBlock = NULL;

	while (cur != NULL)
	{
		if ((BlockA1*)((unsigned char*)cur + cur->size) <= block)
			leftBlock = cur;

		if ((BlockA1*)((unsigned char*)block + block->size) <= cur)
		{
			rightBlock = cur;

			break;
		}

		cur = cur->next;
	}
	
	insertBlockA1(leftBlock, block, rightBlock);

	cur = gFreeA1;

	while (cur != NULL)
	{
		if ((BlockA1*)((unsigned char*)cur + cur->size) == cur->next)
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
