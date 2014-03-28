/*
Лабороторная работа 15
Студента группы: 80-107Б
Барковского Андрея
*/

#include <stdio.h>

int main(void)
{
	const int N = 8;
	int a[N][N], i, j, n, sumMin, sumMax, sumTmp, jMin, jMax;

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

		sumMin = 0;

		for (i = 0; i < n; i++)
		{
			for (j = 0; j < n; j++)
			{
				scanf("%d", &a[i][j]);

				if (j == 0) sumMin += a[i][j];
			}
		}

		printf("Исходная матрица:\n");

		for (i = 0; i < n; i++)
		{
			for (j = 0; j < n; j++) printf("%d ", a[i][j]);

			printf("\n");
		}

		sumMax = sumMin;
		jMax = jMin = 0;

		for (j = 1; j < n; j++)
		{
			sumTmp = 0;

			for (i = 0; i < n; i++) sumTmp += a[i][j];

			if (sumTmp < sumMin)
			{
				sumMin = sumTmp;
				jMin = j;
			}
			else if (sumTmp > sumMax)
			{
				sumMax = sumTmp;
				jMax = j;
			}
		}

		if (jMin != jMax)
		{
			for (i = 0; i < n; i++)
			{
				sumTmp = a[i][jMin];
				a[i][jMin] = a[i][jMax];
				a[i][jMax] = sumTmp;
			}
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
