/*
Лабороторная работа 15
Выполнил: 
Группа: 80-107Б
*/

#include <stdio.h>

int main(void)
{
	const int N = 8;
	int a[N][N], n, i, j, k, m, min, removed;

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

		min = a[0][0];

		for (i = 0; i < n; i++)
			for (j = 0; j < n; j++)
				if (a[i][j] < min) min = a[i][j];

		removed = 0;

		for (i = 0; i < n; i++)
		{
			for (j = 0; j < n; j++)
			{
				if (a[i][j] == min)
				{
					for (m = i; m < n - removed; m++)
						for (k = 0; k < n; k++) a[m][k] = (m < n - removed - 1 ? a[m + 1][k] : 0);

					removed++;
					i--;

					break;
				}
			}
		}

		printf("Результат:\n");

		if (n - removed == 0) printf("Все строки были удалены, так как они содержали минимальный элемент\n");
		else
		for (i = 0; i < n - removed; i++)
		{
			for (j = 0; j < n; j++) printf("%d ", a[i][j]);

			printf("\n");
		}
	}

	return 0;
}
