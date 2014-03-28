#include <stdio.h>
#include <math.h>

double epsylone();

int main(void)
{
	const double a = 0.1, b = 0.6;
	double tayVal, realVal, tmp, eps = epsylone(), x = a;
	int n, k, i, step;

	printf("Введите n: ");
	scanf("%d", &n);
	printf("Введите k: ");
	scanf("%d", &k);

	printf("Eps = %1.30f\n", eps);
	printf("+-------+------------------------------+------------------------------+----------------+\n");
	printf("|   x   |     част. сумма для ряда     |       значения функции       | число итераций |\n");
	printf("+-------+------------------------------+------------------------------+----------------+\n");
	
	for (i = 0; i <= n; i++)
	{
		step = 0;
		tayVal = 0.0;
		realVal = (1.0 + pow(x, 2.0)) * atan(x) / 2.0 - x / 2.0;

		while (1)
		{
			step++;
			tmp = pow(-1.0, step + 1) * pow(x, 2.0 * step + 1.0) / (4.0 * pow(step, 2.0) - 1.0);
			tayVal += tmp;
			
			if (fabs(realVal - tayVal) < eps * k || step == 100) break;
		}

		printf("|%7.2f|%30.20f|%30.20f|%16d|\n", x, tayVal, realVal, step);

		x += (b - a) / n;
	}

	printf("+-------+------------------------------+------------------------------+----------------+\n");

	return 0;
}

double epsylone()
{
	double eps = 1.0;
	
	while (eps / 2.0 + 1.0 > 1.0) eps /= 2.0;

	return eps;
}
