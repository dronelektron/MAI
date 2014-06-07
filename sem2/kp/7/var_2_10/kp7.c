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

Comp mulComplexOnNum(Comp c, const double num);

Cell cellFirst(Vector *v);
void cellNext(Cell *cell);

void printSourceMatrix(Vector *v, const int n);
void printInnerMatrix(const Vector *v);
void printMatrochlen(Vector *v, const int n, const double a, const double b);

int main(void)
{
	const int N = 100;
	int n, i, j, isRowBegin;
	double a, b;
	Vector v;
	Comp tmpComp;
	Item tmpItem;
	Cell cell;

	printf("Введите размерность матрицы: ");
	scanf("%d", &n);

	if (n < 1 || n > N)
	{
		printf("Размерность матрицы должна быть в диапозоне от 1 до %d\n", N);

		return 0;
	}

	vectorCreate(&v, 1);

	tmpItem.ind = EMPTY;

	vectorPushBack(&v, tmpItem);

	for (i = 0; i < n; i++)
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

	printf("Введите a: ");
	scanf("%lf", &a);
	printf("Введите b: ");
	scanf("%lf", &b);

	printf("Обычное представление:\n");
	printSourceMatrix(&v, n);
	printf("Внутреннее представление\n");
	printInnerMatrix(&v);

	printMatrochlen(&v, n, a, b);

	vectorDestroy(&v);

	return 0;
}

Comp mulComplexOnNum(Comp c, const double num)
{
	c.a *= num;
	c.b *= num;

	return c;
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

void printSourceMatrix(Vector *v, const int n)
{
	int i, j;
	Cell cell = cellFirst(v);
	
	for (i = 0; i < n; i++)
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

void printMatrochlen(Vector *v, const int n, const double a, const double b)
{
	int i, j;
	Cell cell = cellFirst(v);
	Comp comp;
	
	printf("Матрочлен:\n");

	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			if (i == j)
			{
				if (cell.row == cell.col && cell.row == i)
				{
					comp = mulComplexOnNum(cell.data, a);
					comp.a += b;

					printf("(%.2lf, %.2lf) ", comp.a, comp.b);

					cellNext(&cell);
				}
				else
					printf("(%.2lf, %.2lf) ", b, 0.0);
			}
			else if (i == cell.row && j == cell.col)
			{
				comp = mulComplexOnNum(cell.data, a);

				printf("(%.2lf, %.2lf) ", comp.a, comp.b);

				cellNext(&cell);
			}
			else
				printf("(%.2lf, %.2lf) ", 0.0, 0.0);
		}

		printf("\n");
	}
}
