#include "allocator2.h"

int initMKK(size_t size)
{
	size_t i;
	BlockMKK* block = NULL;
	
	gPagesCntMKK = getPagesCountMKK(size);
	gPowCntMKK = powOfSizeMKK(PAGE_SIZE_MKK);
	gPowIndexMin = powOfSizeMKK(sizeof(BlockMKK));
	gHeapMKK = malloc(gPagesCntMKK * PAGE_SIZE_MKK);
	gMemsizeMKK = (size_t*)malloc(sizeof(size_t) * gPagesCntMKK);
	gListMKK = (BlockMKK**)malloc(sizeof(BlockMKK*) * gPowCntMKK);

	if (gHeapMKK == NULL || gMemsizeMKK == NULL || gListMKK == NULL)
		return 0;
	
	gMemsizeMKK[FREE] = FREE;
	gListMKK[FREE] = (BlockMKK*)gHeapMKK;
	block = gListMKK[FREE];

	for (i = 1; i < gPagesCntMKK; ++i)
	{
		gMemsizeMKK[i] = FREE;
		block->next = (BlockMKK*)((PBYTE_MKK)block + PAGE_SIZE_MKK);
		block = block->next;
	}

	block->next = NULL;

	for (i = 1; i < gPowCntMKK; ++i)
		gListMKK[i] = NULL;

	return 1;
}

void destroyMKK()
{
	free(gHeapMKK);
	free(gMemsizeMKK);
	free(gListMKK);
}

void* mallocMKK(size_t size)
{
	size_t powIndex = powOfSizeMKK(size);
	size_t oldSize = size;
	BlockMKK* block = NULL;

	if (powIndex < gPowIndexMin)
		powIndex = gPowIndexMin;

	size = 1 << powIndex;
	
	if (size < PAGE_SIZE_MKK)
	{
		if (gListMKK[powIndex] == NULL)
		{
			block = allocPageMKK(size);

			if (block == NULL)
				return NULL;

			splitPageMMK(block, powIndex);
		}
		
		block = gListMKK[powIndex];
		gListMKK[powIndex] = block->next;

		gReqMKK += oldSize;
		gTotMKK += size;

		return (void*)block;
	}
	else
	{
		gReqMKK += oldSize;
		gTotMKK += size;

		return allocPageMKK(size);
	}
}

void freeMKK(void* ptr)
{
	size_t pageIndex = getPageIndexMKK((BlockMKK*)ptr);
	size_t powIndex = powOfSizeMKK(gMemsizeMKK[pageIndex]);
	BlockMKK* block = (BlockMKK*)ptr;
	
	if (gMemsizeMKK[pageIndex] < PAGE_SIZE_MKK)
	{
		block->next = gListMKK[powIndex];
		gListMKK[powIndex] = block;
	}
	else
		freePageMKK(block);
}

BlockMKK* allocPageMKK(size_t size)
{
	size_t cnt = 0;
	size_t pageIndex = 0;
	size_t prevIndex = getPageIndexMKK(gListMKK[FREE]);
	size_t pages = getPagesCountMKK(size);
	BlockMKK* cur = gListMKK[FREE];
	BlockMKK* prev = NULL;
	BlockMKK* page = NULL;

	while (cur != NULL)
	{
		pageIndex = getPageIndexMKK(cur);

		if (pageIndex - prevIndex <= 1)
		{
			if (page == NULL)
				page = cur;

			++cnt;
		}
		else
		{
			page = cur;
			cnt = 1;
		}

		if (cnt == pages)
			break;

		prev = cur;
		cur = cur->next;
		prevIndex = pageIndex;
	}

	if (cnt < pages)
		page = NULL;

	if (page != NULL)
	{
		pageIndex = getPageIndexMKK(page);
		gMemsizeMKK[pageIndex] = size;
		cur = (BlockMKK*)((PBYTE_MKK)page + (pages - 1) * PAGE_SIZE_MKK);
		
		if (prev != NULL)	
			prev->next = cur->next;
		else
			gListMKK[FREE] = cur->next;
	}

	return page;
}

void freePageMKK(BlockMKK* block)
{
	size_t i;
	size_t pageIndex = getPageIndexMKK(block);
	size_t blockCnt = gMemsizeMKK[pageIndex] / PAGE_SIZE_MKK;
	BlockMKK* left = NULL;
	BlockMKK* right = NULL;
	BlockMKK* cur = block;

	while (cur != NULL)
	{
		if (cur < block)
			left = cur;
		else if (cur > block)
		{
			right = cur;

			break;
		}

		cur = cur->next;
	}

	for (i = 1; i < blockCnt; ++i)
	{
		block->next = (BlockMKK*)((PBYTE_MKK)block + PAGE_SIZE_MKK);
		block = block->next;
	}

	block->next = right;

	if (left != NULL)
		left->next = block;
	else
		gListMKK[FREE] = block;
}

void splitPageMMK(BlockMKK* block, size_t powIndex)
{
	size_t i;
	size_t pageIndex = getPageIndexMKK(block);
	size_t blockSize = 1 << powIndex;
	size_t blockCnt = PAGE_SIZE_MKK / blockSize;

	gListMKK[powIndex] = block;
	gMemsizeMKK[pageIndex] = blockSize;
	
	for (i = 1; i < blockCnt; ++i)
	{
		block->next = (BlockMKK*)((PBYTE_MKK)block + blockSize);
		block = block->next;
	}

	block->next = NULL;
}

size_t powOfSizeMKK(size_t size)
{
	size_t p = 0;

	while (size > ((size_t)1 << p))
		++p;

	return p;
}

size_t getPagesCountMKK(size_t size)
{
	return size / PAGE_SIZE_MKK + (size_t)(size % PAGE_SIZE_MKK != 0);
}

size_t getPageIndexMKK(BlockMKK* block)
{
	return (size_t)((PBYTE_MKK)block - (PBYTE_MKK)gHeapMKK) / PAGE_SIZE_MKK;
}

size_t getReqMKK()
{
	return gReqMKK;
}

size_t getTotMKK()
{
	return gTotMKK;
}
