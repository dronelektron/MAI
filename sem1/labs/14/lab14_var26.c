/*
Лабороторная работа 14
Студента группы: 
Ф.И.О.
*/

#include <stdio.h>

typedef enum _kDir
{
	LEFT = 0,
	UP,
	RIGHT,
	DOWN
} kDir;

int main(void)
{
	const int N = 7;
	int n, a[N][N], i, j, k, leftInd, rightInd, upInd, downInd;
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

		dir = LEFT;
		leftInd = 0;
		rightInd = n - 1;
		upInd = 0;
		downInd = n - 1;
		i = n - 1;
		j = n - 1;

		printf("Линейный вид: ");

		for (k = 0; k < n * n; k++)
		{
			printf("%d ", a[i][j]);

			switch (dir)
			{
				case LEFT:
				{
					if (j > leftInd) j--;
					else
					{
						i--;
						downInd--;
						dir = UP;
					}
				}
				break;

				case UP:
				{
					if (i > upInd) i--;
					else
					{
						j++;
						leftInd++;
						dir = RIGHT;
					}
				}
				break;

				case RIGHT:
				{
					if (j < rightInd) j++;
					else
					{
						i++;
						upInd++;
						dir = DOWN;
					}
				}
				break;

				case DOWN:
				{
					if (i < downInd) i++;
					else
					{
						j--;
						rightInd--;
						dir = LEFT;
					}
				}
				break;
			}
		}

		printf("\n");
	}

	return 0;
}
