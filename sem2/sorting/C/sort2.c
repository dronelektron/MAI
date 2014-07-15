#include "sort2.h"

void sort2(int *a, const int n)
{
	int i, j;
	int b[n];
	int count[n];

	for (i = 0; i < n; i++)
		count[i] = 0;

	for (i = 0; i < n - 1; i++)
		for (j = i + 1; j < n; j++)
			if (a[i] < a[j])
				count[j]++;
			else
				count[i]++;

	for (i = 0; i < n; i++)
		b[count[i]] = a[i];

	for (i = 0; i < n; i++)
		a[i] = b[i];
}
