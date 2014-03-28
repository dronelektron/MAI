#include <stdio.h>
#include <stdlib.h>

int main(void)
{
	int i, n, max, *pArr = NULL;

	do
	{
		printf("Input n: ");
		scanf("%d", &n);
	} while (n <= 0);

	//pArr = (int *)malloc(sizeof(int) * n);
	pArr = (int)malloc(sizeof(int) * n);

	for (i = 0; i < n; i++)
	{
		printf("a[%d] = ", i);
		scanf("%d", &pArr[i]);
	}

	max = pArr[0];

	for (int i = 1; i < n; i++)
		if (pArr[i] > max) max = pArr[i];

	free(pArr);

	printf("Maximum = %d\n", max);

	return 0;
}
