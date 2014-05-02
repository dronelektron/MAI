#include <stdio.h>
#include <math.h>

typedef enum _kInd
{
	END = -3,
	COMP,
	EMPTY
} kInd;

typedef struct _Comp
{
	double a;
	double b;
} Comp;

typedef struct _Item
{
	int ind;
	Comp c;
} Item;

double complexModule(const Comp c);
Comp complexDivide(const Comp c1, const Comp c2);
void printSourceMatrix(const int *rows, const Item *cols, const int m, const int n);
void printInnerMatrix(const int *rows, const Item *cols, const int m);

int main(void)
{
	const int maxM = 100, maxN = 100;
	int m, n, i, j, isEmptyRow, isMaxInRow, colCnt = 0, row[maxM];
	Item col[3 * maxN];
	Comp maxComp, tmpComp;

	for (i = 0; i < maxM; i++)
		row[i] = END;

	for (i = 0; i < 3 * maxN; i++)
		col[i].ind = END;

	printf("Введите количество строк: ");
	scanf("%d", &m);
	printf("Введите количество столбцов: ");
	scanf("%d", &n);

	if (m < 1 || m > maxM)
	{
		printf("Количество строк должно быть в диапозоне от 1 до %d\n", maxM);

		return 0;
	}

	if (n < 1 || n > maxN)
	{
		printf("Количество столбцов должно быть в диапозоне от 1 до %d\n", maxN);

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
				row[i] = colCnt;

			col[colCnt++].ind = j;
			col[colCnt].ind = COMP;
			col[colCnt++].c = tmpComp;
		}

		if (isEmptyRow)
			row[i] = EMPTY;
		else
			col[colCnt++].ind = EMPTY;
	}

	printf("Обычное представление:\n");
	printSourceMatrix(row, col, m, n);
	printf("Внутреннее представление\n");
	printInnerMatrix(row, col, m);

	maxComp.a = 0.0;
	maxComp.b = 0.0;

	for (i = 0; i < m; i++)
	{
		for (j = row[i]; j < col[j].ind != END && col[j].ind != EMPTY; j++)
		{
			if (col[j].ind != COMP)
				continue;

			if (complexModule(col[j].c) > complexModule(maxComp))
				maxComp = col[j].c;
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

		for (j = row[i]; j < col[j].ind != END && col[j].ind != EMPTY; j++)
		{
			if (col[j].ind != COMP)
				continue;

			if (complexModule(col[j].c) == complexModule(maxComp))
			{
				isMaxInRow = 1;

				break;
			}
		}

		if (!isMaxInRow)
			continue;

		for (j = row[i]; j < col[j].ind != END && col[j].ind != EMPTY; j++)
		{
			if (col[j].ind == COMP)
				col[j].c = complexDivide(col[j].c, maxComp);
			else if (col[j].ind != EMPTY)
				continue;
		}
	}

	printf("Обычное представление после модификации:\n");
	printSourceMatrix(row, col, m, n);
	printf("Внутреннее представление после модификации:\n");
	printInnerMatrix(row, col, m);

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

void printSourceMatrix(const int *rows, const Item *cols, const int m, const int n)
{
	int i, j, z, k;

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
			if (cols[j].ind == EMPTY)
			{
				printf("(%.2lf, %.2lf) ", 0.0, 0.0);

				k++;

				continue;
			}

			while (k < cols[j].ind)
			{
				printf("(%.2lf, %.2lf) ", 0.0, 0.0);

				k++;
			}

			printf("(%.2lf, %.2lf) ", cols[j + 1].c.a, cols[j + 1].c.b);
			
			j += 2;
			k++;
		}

		printf("\n");
	}
}

void printInnerMatrix(const int *rows, const Item *cols, const int m)
{
	int i, j, k = 0;

	printf("Массив M:\n");

	for (i = 0; i < m; i++)
	{
		printf("%d ", rows[i]);

		if (rows[i] != -1)
			k++;
	}

	printf("\nМассив A:\n");

	if (k == 0)
	{
		printf("Пуст\n");

		return;
	}

	for (i = 0; cols[i].ind != END; i++)
		if (cols[i].ind == COMP)
			printf("(%.2lf, %.2lf) ", cols[i].c.a, cols[i].c.b);
		else
			printf("%d ", cols[i].ind);

	printf("\n");
}
