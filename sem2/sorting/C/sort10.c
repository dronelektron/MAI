#include "sort10.h"

void sort10(int *a, const int n)
{
	mergeSort(a, 0, n - 1);
}

void mergeSort(int *a, const int left, const int right)
{
	int middle = (left + right) / 2;

	if (left < right)
	{
		mergeSort(a, left, middle);
		mergeSort(a, middle + 1, right);
		merge(a, left, right);
	}
}

void merge(int *a, const int left, const int right)
{
	int tmp[right - left + 1];
	int middle = (left + right) / 2;
	int i, pos = 0, pos1 = left, pos2 = middle + 1;

	while (pos1 <= middle && pos2 <= right)
		if (a[pos1] <= a[pos2])
			tmp[pos++] = a[pos1++];
		else
			tmp[pos++] = a[pos2++];

	while (pos1 <= middle)
		tmp[pos++] = a[pos1++];

	while (pos2 <= right)
		tmp[pos++] = a[pos2++];

	for (i = left; i <= right; i++)
		a[i] = tmp[i - left];
}
