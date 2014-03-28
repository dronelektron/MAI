/*
Лабороторная работа 15
Выполнил: 
Группа: 80-107Б
*/

#include <stdio.h>

int main(void)
{
	const int N = 8;
	int a[N][N], n, i, j, max;

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

		for (i = 0; i < n; i++)
		{
			max = a[i][0];

			for (j = 0; j < n; j++)
				if (a[i][j] > max) max = a[i][j];

			a[i][i] = max;
			a[i][n - i - 1] = max;
		}

		printf("Результат:\n");

		for (i = 0; i < n; i++)
		{
			for (j = 0; j < n; j++) printf("%d ", a[i][j]);

			printf("\n");
		}
	}

	return 0;
}
