#include <stdio.h>
#include <time.h>
#include "allocator1.h"
//#include "allocator2.h"

size_t parseSize(const char* str);

int main(int argc, char* argv[])
{
	const size_t N = 10000;
	size_t i;
	size_t j;
	size_t k;
	size_t arg;
	clock_t time1;
	clock_t time2;
	void* ptr;
	void* arr[N];

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
	
	time1 = clock();

	for (i = 0; i < N; ++i)
	{
		arg = 1 + rand() % 256;
		arr[i] = mallocA1(arg);
	}
	
	time2 = clock();

	printf("Alloc time: %lf\n", (double)(time2 - time1) / CLOCKS_PER_SEC);

	for (i = 0; i < N; ++i)
	{
		j = rand() % N;
		k = rand() % N;

		ptr = arr[j];
		arr[j] = arr[k];
		arr[k] = ptr;
	}

	for (i = 0; i < N; ++i)
	{
		if (arr[i] == NULL)
			continue;
		
		freeA1(arr[i]);
	}
	
	time1 = clock();

	printf("Free time: %lf\n", (double)(time1 - time2) / CLOCKS_PER_SEC);
	
	destroyAllocatorA1();

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
