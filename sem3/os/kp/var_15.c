#include <stdio.h>
#include <time.h>
#include "allocator1.h"
#include "allocator2.h"

size_t parseSize(const char* str);

int main(int argc, char* argv[])
{
	const size_t N = 100;
	size_t i;
	size_t j;
	size_t k;
	size_t arg;
	clock_t time1;
	clock_t time2;
	void* addr[N];
	size_t bytes[N];
	size_t delSeq[N];

	srand((unsigned int)time(0));

	if (argc < 2)
	{
		printf("Usage: %s <HEAP_SIZE>\n", argv[0]);

		return 0;
	}

	arg = parseSize(argv[1]);

	if (arg == 0)
	{
		printf("Error. Invalid size\n");

		return 0;
	}

	if (arg < MIN_BLOCK_SIZE_A1)
	{
		printf("Error. Heap size must be at least %zu bytes\n", MIN_BLOCK_SIZE_A1);

		return 0;
	}

	if (!initAllocatorA1(arg))
	{
		printf("Error. No memory\n");

		return 0;
	}

	if (!initAllocatorA2(arg))
	{
		printf("Error. No memory\n");

		return 0;
	}
	
	for (i = 0; i < N; ++i)
	{
		bytes[i] = 1 + rand() % 256;
		delSeq[i] = i;
	}

	for (i = 0; i < N; ++i)
	{
		j = rand() % N;
		k = rand() % N;

		arg = delSeq[j];
		delSeq[j] = delSeq[k];
		delSeq[k] = arg;
	}

	// ALLOCATOR 1 TIMER
	time1 = clock();

	for (i = 0; i < N; ++i)
		addr[i] = mallocA1(bytes[i]);
	
	time2 = clock();

	printf("[1] Alloc time: %lf\n", (double)(time2 - time1) / CLOCKS_PER_SEC);

	for (i = 0; i < N; ++i)
	{
		if (addr[delSeq[i]] == NULL)
			continue;
		
		freeA1(addr[delSeq[i]]);
	}
	
	time1 = clock();

	printf("[1] Free time: %lf\n", (double)(time1 - time2) / CLOCKS_PER_SEC);

	// ALLOCATOR 2 TIMER
	time1 = clock();

	for (i = 0; i < N; ++i)
		addr[i] = mallocA2(bytes[i]);
	
	time2 = clock();

	printf("[2] Alloc time: %lf\n", (double)(time2 - time1) / CLOCKS_PER_SEC);

	for (i = 0; i < N; ++i)
	{
		if (addr[delSeq[i]] == NULL)
			continue;

		freeA2(addr[delSeq[i]]);
	}
	
	time1 = clock();

	printf("[2] Free time: %lf\n", (double)(time1 - time2) / CLOCKS_PER_SEC);

	destroyAllocatorA1();
	destroyAllocatorA2();

	return 0;
}

size_t parseSize(const char* str)
{
	size_t size = 0;

	while (*str != '\0')
	{
		if (*str < '0' || *str > '9')
			return 0;

		size = size * 10 + *str - '0';
		++str;
	}

	return size;
}
