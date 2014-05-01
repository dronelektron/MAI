#include <stdio.h>
#include <time.h>
#include "sort.h"

int randomAB(int a, int b);

int main(void)
{
	const int N = 20;
	int i, rnd;
	Udt udt;
	
	srand((unsigned int)time(0));

	udtCreate(&udt, N);

	printf("Сгенерированный дек:\n");

	for (i = 0; i < N; i++)
	{
		rnd = randomAB(-10, 10);
		
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

int randomAB(int a, int b)
{
	return a + rand() % (b - a + 1);
}
