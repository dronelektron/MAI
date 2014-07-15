#include "sort7.h"

void sort7(int *a, const int n)
{
	int i, j, k, h;

	for (h = n - 1; h > 0; h /= 2)
	{
		for (i = h; i < n; i++)
		{
			k = a[i];

			for (j = i; j >= h && k < a[j - h]; j -= h)
				a[j] = a[j - h];

			a[j] = k;
		}
	}
}
