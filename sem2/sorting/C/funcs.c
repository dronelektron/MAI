#include "funcs.h"

void generateArray(int *a, const int n, int from, int to)
{
	int i;

	srand((unsigned int)time(0));

	for (i = 0; i < n; i++)
		a[i] = randomAB(from, to);
}

void shuffleArray(int *a, const int n)
{
	int k, i, j;

	for (k = 0; k < n; k++)
	{
		i = randomAB(0, n - 1);
		j = randomAB(0, n - 1);

		swap(&a[i], &a[j]);
	}
}

void printArray(int *a, const int n)
{
	int i;

	for (i = 0; i < n; i++)
		printf("%d ", a[i]);

	printf("\n");
}

void swap(int *a, int *b)
{
	int c = *a;

	*a = *b;
	*b = c;
}

int randomAB(int a, int b)
{
	return a + rand() % (b - a + 1);
}
