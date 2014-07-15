#include "sort6.h"

void sort6(int *a, const int n)
{
	int i, j;
	int m, left, right;
	int x;

	for (i = 1; i < n; i++)
	{
		x = a[i];
		left = 0;
		right = i;

		while (left < right)
		{
			m = (left + right) / 2;

			if (a[m] <= x)
				left = m + 1;
			else
				right = m;
		}

		for (j = i; j > right + 1; j--)
			a[j] = a[j - 1];

		a[right] = x;
	}
}
