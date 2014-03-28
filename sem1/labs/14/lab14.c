/*
Лабороторная работа 14
Студента группы: 80-107Б
Барковского Андрея
*/

#include <stdio.h>

typedef enum _kDir
{
	RIGHT = 0, UP, LEFT, DOWN
} kDir;

int main(void)
{
	const int N = 7;
	int n, a[N][N], i, j, k, sw, iMin, jMin;
	kDir dir;

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

		i = n - 1;
		j = n - 2;
		jMin = iMin = j;
		sw = 0;
		dir = RIGHT;

		printf("Линейный вид: ");

		for (k = 0; k < n * n; k++)
		{
			switch (dir)
			{
				case RIGHT:
				{
					if (j < n - 1) j++;
					else
					{
						sw++;
						i--;
						dir = UP;
					}
				}
				break;

				case UP:
				{
					if (i > iMin) i--;
					else
					{
						iMin--;

						if (++sw == 2)
						{
							j--;
							dir = LEFT;
						}
						else
						{
							j++;
							dir = RIGHT;
						}
					}
				}
				break;

				case LEFT:
				{
					if (j > jMin) j--;
					else
					{
						jMin--;

						if (++sw == 3)
						{
							i++;
							dir = DOWN;
						}
						else
						{
							i--;
							dir = UP;
						}
					}
				}
				break;

				case DOWN:
				{
					if (i < n - 1) i++;
					else
					{
						sw++;
						j--;
						dir = LEFT;
					}
				}
				break;
			}

			if (sw > 6) sw = 1;

			printf("%d ", a[i][j]);
		}

		printf("\n");
	}

	return 0;
}
