#include <stdio.h>

int abs(int a);
int sign(int a);
int max(int a, int b);

int main(void)
{
	const int i0 = -1, j0 = -1, l0 = -9, x0 = 10, y0 = 10, rIn = 5, rOut = 10;
	int iPrev = i0, jPrev = j0, lPrev = l0, f = 0, i, j, l, k = 0;

	if (((i0 - x0) * (i0 - x0) + (j0 - y0) * (j0 - y0) <= rOut * rOut) && ((i0 - x0) * (i0 - x0) + (j0 - y0) * (j0 - y0) >= rIn * rIn))
	{
		printf("Точка попала в кольцо, ее координаты (%d, %d), параметр движения l = %d и итерация k = %d\n", i0, j0, l0, k);

		return 0;
	}

	for (; k < 50; k++)
	{
		i = max(jPrev - k, lPrev - k) % 30 + max(iPrev + lPrev, jPrev + k) % 20;
		j = (abs(iPrev - lPrev) * sign(jPrev + k) + abs(iPrev - k) * (jPrev + k)) % 20;
		l = ((iPrev + k) * (jPrev - k) * (lPrev + k)) % 25;

		iPrev = i; jPrev = j; lPrev = l;

		if (((i - x0) * (i - x0) + (j - y0) * (j - y0) <= rOut * rOut) && ((i - x0) * (i - x0) + (j - y0) * (j - y0) >= rIn * rIn))
		{
			printf("Точка попала в кольцо, ее координаты (%d, %d), параметр движения l = %d и итерация k = %d\n", i, j, l, k + 1);

			f = 1;

			break;
		}
	}

	if (!f) printf("Точка не попала в кольцо за отведенное время, последние координаты точки (%d, %d), параметр движения l = %d и итерация k = %d\n", i, j, l, k);

	return 0;
}

int abs(int a) { return (a >= 0 ? a : -a); }
int sign(int a) { return (a >= 0 ? 1 : -1); }
int max(int a, int b) { return (a >= b ? a : b); }
