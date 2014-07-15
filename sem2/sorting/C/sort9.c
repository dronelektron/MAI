#include "sort9.h"

void sort9(int *a, const int n)
{
	int i;

	for (i = (n - 2) / 2; i >= 0; i--)
		sift(a, i, n);

	for (i = n - 1; i > 0; i--)
	{
		swap(&a[0], &a[i]);
		sift(a, 0, i);
	}
}

void sift(int *a, int start, int end)
{
	int root = start;
	int child = root * 2 + 1;

	while (child < end)
	{
		if (child + 1 < end && a[child] < a[child + 1])
			child++;

		if (a[root] >= a[child])
			break;

		swap(&a[root], &a[child]);

		root = child;
		child = root * 2 + 1;
	}
}
