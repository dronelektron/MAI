#include "sort3.h"

void sort3(int *a, const int n)
{
	int i, j;

	for (i = 0; i < n; i++)
		for (j = 0; j < n - i - 1; j++)
			if (a[j] > a[j + 1])
				swap(&a[j], &a[j + 1]);
}
