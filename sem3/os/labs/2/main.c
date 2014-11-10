#include <stdio.h>
#include <stdlib.h>

extern void bubbleSortAsm(int* p, int size);

int main(void)
{
	int i;
	int n;
	int* arr = NULL;
	
	printf("Input n: ");
	scanf("%d", &n);
	
	if (n == 0)
	{
		printf("Error. Size of array must be greater than 0\n");
		
		return 0;
	}

	arr = (int*)malloc(sizeof(int) * n);

	for (i = 0; i < n; ++i)
	{
		printf("arr[%d] = ", i);
		scanf("%d", &arr[i]);
	}
	
	bubbleSortAsm(arr, n);
	
	printf("Sorted:\n");

	for (i = 0; i < n; ++i)
	{
		printf("arr[%d] = %d\n", i, arr[i]);
	}

	free(arr);
}
