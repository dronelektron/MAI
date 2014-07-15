#include "sort11.h"

void sort11(int *a, const int n)
{
	int p = a[(n - 1) / 2];
	int *left = a;
	int *right = a + n - 1;

	if (n < 2)
		return;

	while (left <= right)
	{
		if (*left < p)
		{
			left++;

			continue;
		}

		if (*right > p)
		{
			right--;

			continue;
		}

		swap(left, right);

		left++;
		right--;
	}

	sort11(a, right - a + 1);
	sort11(left, a + n - left);
}
