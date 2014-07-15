#include "sort1.h"

void sort1(int *a, const int n)
{
	int i, j;
	int min;

	for (i = 0; i < n - 1; i++)
	{
		min = i;

		for (j = i + 1; j < n; j++)
			if (a[j] < a[min])
				min = j;

		swap(&a[i], &a[min]);
	}
}
