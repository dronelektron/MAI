#include <stdio.h>
#include <time.h>
#include "allocator1.h"
#include "allocator2.h"

size_t parseSize(const char* str);

int main(int argc, char* argv[])
{
	const size_t N = 10000;
	const size_t MAX_BYTES = 9000;
	size_t i;
	size_t j;
	size_t k;
	size_t arg;
	size_t req = 0;
	size_t tot = 0;
	size_t bytes[N];
	size_t delSeq[N];
	void* addr[N];
	clock_t time1;
	clock_t time2;
	clock_t time3;
	clock_t time4;

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

	if (!initAllocatorA1(arg))
	{
		printf("Error. No memory\n");

		return 0;
	}

	if (!initMKK(arg))
	{
		printf("Error. No memory\n");

		return 0;
	}
	
	for (i = 0; i < N; ++i)
	{
		bytes[i] = 1 + rand() % MAX_BYTES;
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

	printf("--------------------------------\n");
	printf("Alloc requests: %zu\n", N);
	printf("Bytes: 1 to %zu\n", MAX_BYTES);
	printf("--------------------------------\n");
	printf("Allocator #1 (LIST):\n");
	printf("--------------------------------\n");

	time1 = clock();

	for (i = 0; i < N; ++i)
	{
		//time3 = clock();
		addr[i] = mallocA1(bytes[i]);
		//time4 = clock();

		//printf("(%lf sec)\n\n", (double)(time4 - time3) / CLOCKS_PER_SEC);
	}
	
	time2 = clock();

	//printf("--------------------------------\n");
	printf("Alloc time: %lf\n", (double)(time2 - time1) / CLOCKS_PER_SEC);

	req = getReqA1();
	tot = getTotA1();

	for (i = 0; i < N; ++i)
	{
		if (addr[delSeq[i]] == NULL)
			continue;
		
		freeA1(addr[delSeq[i]]);
	}
	
	time1 = clock();
	
	printf("Free time: %lf\n", (double)(time1 - time2) / CLOCKS_PER_SEC);
	printf("Usage factor: %lf\n", (double)req / tot);
	printf("--------------------------------\n");
	printf("Allocator #2 (MKK):\n");
	printf("--------------------------------\n");

	time1 = clock();
	
	for (i = 0; i < N; ++i)
	{
		//time3 = clock();
		addr[i] = mallocMKK(bytes[i]);
		//time4 = clock();

		//printf("(%lf sec)\n\n", (double)(time4 - time3) / CLOCKS_PER_SEC);
	}
	
	time2 = clock();

	//printf("--------------------------------\n");
	printf("Alloc time: %lf\n", (double)(time2 - time1) / CLOCKS_PER_SEC);

	req = getReqMKK();
	tot = getTotMKK();

	for (i = 0; i < N; ++i)
	{
		if (addr[delSeq[i]] == NULL)
			continue;

		freeMKK(addr[delSeq[i]]);
	}
	
	time1 = clock();

	printf("Free time: %lf\n", (double)(time1 - time2) / CLOCKS_PER_SEC);
	printf("Usage factor: %lf\n", (double)req / tot);
	printf("--------------------------------\n");
	
	destroyAllocatorA1();
	destroyMKK();
	
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
