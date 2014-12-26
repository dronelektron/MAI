#include <stdio.h>
#include <time.h>
#include "allocator1.h"
#include "allocator2.h"

size_t parseSize(const char* str);

typedef struct _Req
{
	void* addr;
	size_t bytes;
} Req;

int main(int argc, char* argv[])
{
	const size_t N = 1000;
	const size_t MAX_BYTES = 5000;
	clock_t time1;
	clock_t time2;
	clock_t time3;
	clock_t time4;
	size_t i;
	size_t j;
	size_t k;
	size_t arg;
	size_t req = 0;
	size_t tot = 0;
	size_t* delSeq = (size_t*)malloc(sizeof(size_t) * N);
	Req* reqs = (Req*)malloc(sizeof(Req) * N);
	FILE* allocLogA1 = fopen("alloc_1_data.txt", "w");
	FILE* allocLogA2 = fopen("alloc_2_data.txt", "w");
	FILE* freeLogA1 = fopen("free_1_data.txt", "w");
	FILE* freeLogA2 = fopen("free_2_data.txt", "w");
	FILE* factorLogA1 = fopen("factor_1_data.txt", "w");
	FILE* factorLogA2 = fopen("factor_2_data.txt", "w");

	srand((unsigned int)time(0));

	if (argc < 2)
	{
		printf("Usage: %s <HEAP_SIZE>\n", argv[0]);

		return 0;
	}

	arg = parseSize(argv[1]);

	if (!initList(arg))
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
		reqs[i].bytes = 1 + rand() % MAX_BYTES;
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
		time3 = clock();
		reqs[i].addr = mallocList(reqs[i].bytes);
		time4 = clock();

		if (i % 100 == 0)
		{
			fprintf(allocLogA1, "%zu\n", (size_t)(1000000 * (double)(time4 - time3) / CLOCKS_PER_SEC));
			fprintf(factorLogA1, "%zu\t%zu\n", getReqList(), getTotList());
		}
	}
	
	time2 = clock();

	printf("Alloc time: %lf\n", (double)(time2 - time1) / CLOCKS_PER_SEC);

	req = getReqList();
	tot = getTotList();

	for (i = 0; i < N; ++i)
	{
		if (reqs[delSeq[i]].addr == NULL)
			continue;
		
		time3 = clock();
		freeList(reqs[delSeq[i]].addr);
		time4 = clock();

		if (i % 100 == 0)
			fprintf(freeLogA1, "%zu\n", (size_t)(1000000 * (double)(time4 - time3) / CLOCKS_PER_SEC));
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
		time3 = clock();
		reqs[i].addr = mallocMKK(reqs[i].bytes);
		time4 = clock();

		if (i % 100 == 0)
		{
			fprintf(allocLogA2, "%zu\n", (size_t)(1000000 * (double)(time4 - time3) / CLOCKS_PER_SEC));
			fprintf(factorLogA2, "%zu\t%zu\n", getReqMKK(), getTotMKK());
		}
	}
	
	time2 = clock();

	printf("Alloc time: %lf\n", (double)(time2 - time1) / CLOCKS_PER_SEC);

	req = getReqMKK();
	tot = getTotMKK();

	for (i = 0; i < N; ++i)
	{
		if (reqs[delSeq[i]].addr == NULL)
			continue;

		time3 = clock();
		freeMKK(reqs[delSeq[i]].addr);
		time4 = clock();

		if (i % 100 == 0)
			fprintf(freeLogA2, "%zu\n", (size_t)(1000000 * (double)(time4 - time3) / CLOCKS_PER_SEC));
	}
	
	time1 = clock();

	printf("Free time: %lf\n", (double)(time1 - time2) / CLOCKS_PER_SEC);
	printf("Usage factor: %lf\n", (double)req / tot);
	printf("--------------------------------\n");
	
	destroyList();
	destroyMKK();
	
	fclose(allocLogA1);
	fclose(allocLogA2);
	fclose(freeLogA1);
	fclose(freeLogA2);
	fclose(factorLogA1);
	fclose(factorLogA2);

	free(reqs);
	free(delSeq);

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
