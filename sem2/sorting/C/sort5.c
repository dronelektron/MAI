#include "sort5.h"

void sort5(int *a, const int n)
{
	int i, j;

	for (i = 1; i < n; i++)
	{
		j = i;

		while (j > 0 && a[j] < a[j - 1])
		{
			swap(&a[j], &a[j - 1]);

			j--;
		}
	}
}
