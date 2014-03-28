/*
Лабороторная работа 14
Студента группы: ...
Ф.И.О.
*/

#include <stdio.h>

void swap(int *a, int *b);

int main(void)
{
	const int N = 7;
	int n, a[N][N], i, j, k;

	while (1)
	{
		printf("Введите n (1 - %d. Завершить - 0): ", N);
		scanf("%d", &n);
		
		if (n == 0) break;

		if (n < 1 || n > N)
		{
			printf("Недопустимое значение n\n");

			continue;
		}

		for (i = 0; i < n; i++)
			for (j = 0; j < n; j++)
				scanf("%d", &a[i][j]);

		printf("Исходная матрица:\n");

		for (i = 0; i < n; i++)
		{
			for (j = 0; j < n; j++) printf("%d ", a[i][j]);

			printf("\n");
		}

		printf("Линейный вид: \n");

		for (k = 0; k < 2 * n - 1; k++)
		{
			if (k <= n - 1)
			{
				j = 0;
				i = k;
			}
			else
			{
				j = k - n + 1;
				i = n - 1;
			}
			
			if (k & 1) for (; i >= 0 && j < n; i--, j++) printf("%d ", a[n - i - 1][n - j - 1]);
			else
			{
				swap(&i, &j);

				for (; j >= 0 && i < n; i++, j--) printf("%d ", a[n - i - 1][n - j - 1]);
			}
		}

		printf("\n");
	}

	return 0;
}

void swap(int *a, int *b)
{
	int c = *a;
	*a = *b;
	*b = c;
}
