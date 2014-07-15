#include "sort4.h"

void sort4(int *a, const int n)
{
	int i;
	int left = 0;
	int right = n - 1;

	while (left < right)
	{
		for (i = right; i > left; i--)
			if (a[i] < a[i - 1])
				swap(&a[i], &a[i - 1]);

		left++;

		for (i = left; i < right; i++)
			if (a[i] > a[i + 1])
				swap(&a[i], &a[i + 1]);

		right--;
	}
}
