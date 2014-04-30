#include <stdio.h>
#include <time.h>
#include "sort.h"

int main(void)
{
	const int N = 10;
	int i, rnd;
	Udt udt;
	
	srand((unsigned int)time(0));

	udtCreate(&udt, N);

	printf("Сгенерированный дек:\n");

	for (i = 0; i < N; i++)
	{
		rnd = -10 + rand() % (10 - (-10) + 1);
		
		udtPushBack(&udt, rnd);

		printf("%d ", rnd);
	}

	printf("\n");

	udtQuickSort(&udt);

	printf("Отсортированный дек\n");

	udtPrint(&udt);

	printf("\n");
	
	udtDestroy(&udt);

	return 0;
}
