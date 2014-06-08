#include <stdio.h>
#include <math.h>
#include "vector.h"

typedef enum _kInd
{
	END = -3,
	COMP,
	EMPTY
} kInd;

typedef struct _Cell
{
	Vector *v;
	int ind;
	int row;
	int col;
	Comp data;
} Cell;

double complexModule(const Comp c);
Comp complexDivide(const Comp c1, const Comp c2);

Cell cellFirst(Vector *v);
void cellNext(Cell *cell);

void printSourceMatrix(Vector *v, const int m, const int n);
void printInnerMatrix(const Vector *v);

int main(void)
{
	const int N = 100;
	int m, n, i, j, isRowBegin, lastInd, cnt, maxCols[N];
	Vector v;
	Comp tmpComp, maxComp;
	Item tmpItem;
	Cell cell;

	for (i = 0; i < N; i++)
		maxCols[i] = 0;

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

	vectorCreate(&v, 1);

	tmpItem.ind = EMPTY;

	vectorPushBack(&v, tmpItem);

	for (i = 0; i < m; i++)
	{
		isRowBegin = 0;

		for (j = 0; j < n; j++)
		{
			printf("Введите действительную и мнимую части ячейки [%d][%d]: ", i, j);
			scanf("%lf %lf", &tmpComp.a, &tmpComp.b);

			if (tmpComp.a == 0.0 && tmpComp.b == 0.0)
				continue;

			if (!isRowBegin)
			{
				isRowBegin = 1;

				tmpItem.ind = i;

				vectorPushBack(&v, tmpItem);
			}

			tmpItem.ind = j;

			vectorPushBack(&v, tmpItem);

			tmpItem.c = tmpComp;
			tmpItem.ind = COMP;

			vectorPushBack(&v, tmpItem);
		}

		if (isRowBegin)
		{
			tmpItem.ind = EMPTY;

			vectorPushBack(&v, tmpItem);
		}
	}

	tmpItem.ind = END;

	vectorPushBack(&v, tmpItem);

	printf("Обычное представление:\n");
	printSourceMatrix(&v, m, n);
	printf("Внутреннее представление\n");
	printInnerMatrix(&v);

	maxComp.a = 0.0;
	maxComp.b = 0.0;

	cell = cellFirst(&v);

	while (cell.row != END)
	{
		if (complexModule(cell.data) > complexModule(maxComp))
			maxComp = cell.data;

		cellNext(&cell);
	}

	printf("Максимальное комплексное число по модулю: (%.2lf, %.2lf), модуль равен: %.2lf\n", maxComp.a, maxComp.b, complexModule(maxComp));

	if (maxComp.a == 0.0 && maxComp.b == 0)
	{
		printf("Делить на него нельзя, так как его модуль равен нулю\n");

		return 0;
	}

	lastInd = 0;
	cnt = 0;

	cell = cellFirst(&v);

	while (cell.row != END)
	{
		if (complexModule(cell.data) == complexModule(maxComp))
		{
			maxCols[cell.col] = 1;
			lastInd = cell.col;
			cnt++;
		}

		cellNext(&cell);
	}

	if (cnt > 1)
		for (i = lastInd - 1; i >= 0; i--)
			if (maxCols[i])
			{
				lastInd = i;

				break;
			}

	cell = cellFirst(&v);

	while (cell.row != END)
	{
		if (cell.col == lastInd)
		{
			tmpItem = vectorLoad(&v, cell.ind + 1);
			tmpItem.c = complexDivide(cell.data, maxComp);

			vectorSave(&v, cell.ind + 1, tmpItem);
		}

		cellNext(&cell);
	}

	printf("Обычное представление после преобразования:\n");
	printSourceMatrix(&v, m, n);
	printf("Внутреннее представление после преобразования:\n");
	printInnerMatrix(&v);

	vectorDestroy(&v);

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

Cell cellFirst(Vector *v)
{
	Cell res;

	res.v = v;
	res.ind = 2;
	res.row = END;
	res.col = EMPTY;
	res.data.a = 0.0;
	res.data.b = 0.0;

	if (vectorLoad(v, res.ind - 1).ind != END)
	{
		res.row = vectorLoad(v, res.ind - 1).ind;
		res.col = vectorLoad(v, res.ind).ind;
		res.data = vectorLoad(v, res.ind + 1).c;
	}

	return res;
}

void cellNext(Cell *cell)
{
	int c1, c2;

	if (cell->row == END)
		return;

	cell->ind += 2;
	c1 = vectorLoad(cell->v, cell->ind).ind;
	c2 = vectorLoad(cell->v, cell->ind + 1).ind;

	if (c1 > EMPTY && c2 == COMP)
	{
		cell->col = vectorLoad(cell->v, cell->ind).ind;
		cell->data = vectorLoad(cell->v, cell->ind + 1).c;
	}
	else if (c1 == EMPTY && c2 > EMPTY)
	{
		cell->row = vectorLoad(cell->v, cell->ind + 1).ind;
		cell->col = vectorLoad(cell->v, cell->ind + 2).ind;
		cell->data = vectorLoad(cell->v, cell->ind + 3).c;
		cell->ind += 2;
	}
	else
	{
		cell->row = END;
		cell->col = EMPTY;
	}
}

void printSourceMatrix(Vector *v, const int m, const int n)
{
	int i, j;
	Cell cell = cellFirst(v);
	
	for (i = 0; i < m; i++)
	{
		for (j = 0; j < n; j++)
		{
			if (i == cell.row && j == cell.col)
			{
				printf("(%.2lf, %.2lf) ", cell.data.a, cell.data.b);

				cellNext(&cell);
			}
			else
				printf("(%.2lf, %.2lf) ", 0.0, 0.0);
		}

		printf("\n");
	}
}

void printInnerMatrix(const Vector *v)
{
	int i;
	Item item;
	
	for (i = 0; i < vectorSize(v); i++)
	{
		item = vectorLoad(v, i);

		if (item.ind == COMP)
			printf("(%.2lf, %.2lf) ", item.c.a, item.c.b);
		else
			printf("%d ", item.ind);
	}

	printf("\n");
}
