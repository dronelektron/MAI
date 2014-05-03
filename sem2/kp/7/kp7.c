#include <stdio.h>
#include <math.h>
#include "vector.h"

typedef enum _kInd
{
	END = -3,
	COMP,
	EMPTY
} kInd;

double complexModule(const Comp c);
Comp complexDivide(const Comp c1, const Comp c2);
void printSourceMatrix(const int *rows, const Vector *cols, const int m, const int n);
void printInnerMatrix(const int *rows, const Vector *cols, const int m);

int main(void)
{
	const int N = 100;
	int m, n, i, j, isEmptyRow, isMaxInRow, row[N];
	Comp maxComp, tmpComp;
	Vector col;
	Item tmpItem;

	vectorCreate(&col, 0);

	for (i = 0; i < N; i++)
		row[i] = END;

	printf("Введите количество строк: ");
	scanf("%d", &m);
	printf("Введите количество столбцов: ");
	scanf("%d", &n);

	if (m < 1 || m > N)
	{
		printf("Количество строк должно быть в диапозоне от 1 до %d\n", N);

		return 0;
	}

	if (n < 1 || n > N)
	{
		printf("Количество столбцов должно быть в диапозоне от 1 до %d\n", N);

		return 0;
	}

	for (i = 0; i < m; i++)
	{
		isEmptyRow = 1;

		for (j = 0; j < n; j++)
		{
			printf("Введите действительную часть и мнимую ячейки [%d][%d]: ", i, j);
			scanf("%lf %lf", &tmpComp.a, &tmpComp.b);

			if (complexModule(tmpComp) == 0.0)
				continue;

			isEmptyRow = 0;

			if (row[i] == END)
				row[i] = vectorSize(&col);

			tmpItem.ind = j;

			vectorPushBack(&col, tmpItem);

			tmpItem.ind = COMP;
			tmpItem.c = tmpComp;

			vectorPushBack(&col, tmpItem);
		}

		if (isEmptyRow)
			row[i] = EMPTY;
		else
		{
			tmpItem.ind = EMPTY;

			vectorPushBack(&col, tmpItem);
		}
	}

	tmpItem.ind = END;

	vectorPushBack(&col, tmpItem);

	printf("Обычное представление:\n");
	printSourceMatrix(row, &col, m, n);
	printf("Внутреннее представление\n");
	printInnerMatrix(row, &col, m);

	maxComp.a = 0.0;
	maxComp.b = 0.0;

	for (i = 0; i < m; i++)
	{
		if (row[i] == EMPTY)
			continue;

		for (j = row[i]; j < vectorLoad(&col, j).ind != END && vectorLoad(&col, j).ind != EMPTY; j++)
		{
			if (vectorLoad(&col, j).ind != COMP)
				continue;

			if (complexModule(vectorLoad(&col, j).c) > complexModule(maxComp))
				maxComp = vectorLoad(&col, j).c;
		}
	}

	printf("Максимальное комплексное число по модулю: (%.2lf, %.2lf), модуль равен: %.2lf\n", maxComp.a, maxComp.b, complexModule(maxComp));

	if (maxComp.a == 0.0 && maxComp.b == 0)
	{
		printf("Делить на него нельзя, так как его модуль равен нулю\n");

		return 0;
	}

	for (i = 0; i < m; i++)
	{
		isMaxInRow = 0;

		if (row[i] == EMPTY)
			continue;

		for (j = row[i]; j < vectorLoad(&col, j).ind != END && vectorLoad(&col, j).ind != EMPTY; j++)
		{
			if (vectorLoad(&col, j).ind != COMP)
				continue;

			if (complexModule(vectorLoad(&col, j).c) == complexModule(maxComp))
			{
				isMaxInRow = 1;

				break;
			}
		}

		if (!isMaxInRow)
			continue;

		for (j = row[i]; j < vectorLoad(&col, j).ind != END && vectorLoad(&col, j).ind != EMPTY; j++)
		{
			if (vectorLoad(&col, j).ind == COMP)
			{
				tmpItem = vectorLoad(&col, j);
				tmpItem.c = complexDivide(vectorLoad(&col, j).c, maxComp);

				vectorSave(&col, j, tmpItem);
			}
			else if (vectorLoad(&col, j).ind != EMPTY)
				continue;
		}
	}

	printf("Обычное представление после модификации:\n");
	printSourceMatrix(row, &col, m, n);
	printf("Внутреннее представление после модификации:\n");
	printInnerMatrix(row, &col, m);

	vectorDestroy(&col);

	return 0;
}

double complexModule(const Comp c)
{
	return sqrt(pow(c.a, 2.0) + pow(c.b, 2.0));
}

Comp complexDivide(const Comp c1, const Comp c2)
{
	const double znam = pow(c2.a, 2.0) + pow(c2.b, 2.0);
	Comp res;

	res.a = (double)(c1.a * c2.a + c1.b * c2.b) / znam;
	res.b = (double)(c2.a * c1.b - c2.b * c1.a) / znam;

	return res;
}

void printSourceMatrix(const int *rows, const Vector *cols, const int m, const int n)
{
	int i, j, k;

	for (i = 0; i < m; i++)
	{
		if (rows[i] == EMPTY)
		{
			for (j = 0; j < n; j++)
				printf("(%.2lf, %.2lf) ", 0.0, 0.0);

			printf("\n");

			continue;
		}

		k = 0;
		j = rows[i];

		while (k < n)
		{
			if (vectorLoad(cols, j).ind == EMPTY)
			{
				printf("(%.2lf, %.2lf) ", 0.0, 0.0);

				k++;

				continue;
			}

			while (k < vectorLoad(cols, j).ind)
			{
				printf("(%.2lf, %.2lf) ", 0.0, 0.0);

				k++;
			}

			printf("(%.2lf, %.2lf) ", vectorLoad(cols, j + 1).c.a, vectorLoad(cols, j + 1).c.b);
			
			j += 2;
			k++;
		}

		printf("\n");
	}
}

void printInnerMatrix(const int *rows, const Vector *cols, const int m)
{
	int i, j;

	printf("Массив M:\n");

	for (i = 0; i < m; i++)
		printf("%d ", rows[i]);

	printf("\nМассив A:\n");

	if (vectorLoad(cols, 0).ind == END)
	{
		printf("Пуст\n");

		return;
	}

	for (i = 0; vectorLoad(cols, i).ind != END; i++)
		if (vectorLoad(cols, i).ind == COMP)
			printf("(%.2lf, %.2lf) ", vectorLoad(cols, i).c.a, vectorLoad(cols, i).c.b);
		else
			printf("%d ", vectorLoad(cols, i).ind);

	printf("\n");
}
