/*
Лабороторная работа 15
Выполнил: 
Группа: 80-107Б
*/

#include <stdio.h>

int main(void)
{
	const int N = 8;
	int a[N][N], n, i, j, m, isCopy, removed;

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

		removed = 0;

		for (i = 1; i < n - removed; i++)
		{
			for (m = 0; m < i; m++)
			{
				isCopy = 1;

				for (j = 0; j < n; j++)
				{
					if (a[i][j] != a[m][j])
					{
						isCopy = 0;

						break;
					}
				}

				if (isCopy)
				{
					for (m = i; m < n; m++)
						for (j = 0; j < n; j++)
							a[m][j] = m < n - 1 ? a[m + 1][j] : 0;

					i--;
					removed++;

					break;
				}
			}
		}

		printf("Результат:\n");

		for (i = 0; i < n - removed; i++)
		{
			for (j = 0; j < n; j++) printf("%d ", a[i][j]);

			printf("\n");
		}
	}

	return 0;
}
