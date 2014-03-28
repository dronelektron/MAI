#include <stdio.h>

int abs(int a);
int sign(int a);
int min(int a, int b);
int max(int a, int b);

int main(void)
{
	const int i0 = -22, j0 = 29, l0 = 4, r = 10, x1 = -10, y1 = -10, x2 = -20, y2 = -20;
	int iPrev = i0, jPrev = j0, lPrev = l0, f = 0, i = iPrev, j = jPrev, l = lPrev, k;

	for (k = 0; k < 50; k++)
	{
		if (((x1 - iPrev) * (x1 - iPrev) + (y1 - jPrev) * (y1 - jPrev) <= r * r) &&
			((x2 - iPrev) * (x2 - iPrev) + (y2 - jPrev) * (y2 - jPrev) <= r * r))
		{
			printf("Точка попала в лунку. Координаты точки (%d, %d), параметр движения l = %d и итерация k = %d\n", i, j, l, k + 1);

			f = 1;

			break;
		}

		i = sign(min(iPrev, jPrev)) * max((iPrev + k) % 20, (jPrev + lPrev) % 20);
		j = abs(max(iPrev, jPrev)) - k * min(jPrev, lPrev);
		l = (k - lPrev) / (((iPrev + jPrev + lPrev) * (iPrev + jPrev + lPrev)) % 5 + 1);

		iPrev = i; jPrev = j; lPrev = l;
	}

	if (!f) printf("Точка не попала в лунку за отведенное время, последние координаты точки (%d, %d), параметр движения l = %d и итерация k = %d\n", i, j, l, k);

	return 0;
}

int abs(int a) { return (a >= 0 ? a : -a); }
int sign(int a) { return (a >= 0 ? 1 : -1); }
int min(int a, int b) { return (a <= b ? a : b); }
int max(int a, int b) { return (a >= b ? a : b); }
