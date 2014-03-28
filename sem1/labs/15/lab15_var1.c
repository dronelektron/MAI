#include <stdio.h>

int main(void)
{
	const int N = 8;
	int a[N][N], t[N][N], b[N][N], i, j, k, n, tmp;

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
		{
			for (j = 0; j < n; j++)
			{
				scanf("%d", &a[i][j]);

				t[j][i] = a[i][j];
			}
		}

		printf("Исходная матрица:\n");

		for (i = 0; i < n; i++)
		{
			for (j = 0; j < n; j++) printf("%d ", a[i][j]);

			printf("\n");
		}

		for (i = 0; i < n; i++)
		{
			for (j = 0; j < n; j++)
			{
				tmp = 0;

				for (k = 0; k < n; k++) tmp += a[i][k] * t[k][j];

				b[i][j] = tmp;
			}
		}

		printf("Результат умножения:\n");

		for (i = 0; i < n; i++)
		{
			for (j = 0; j < n; j++) printf("%d ", b[i][j]);

			printf("\n");
		}
	}

	return 0;
}
