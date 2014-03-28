#include <stdio.h>
#include <math.h>

double f1(double x);
double f1_it(double x);
double f1_pr(double x);

double f2(double x);
double f2_it(double x);
double f2_pr(double x);

double dichotomy(double (*f)(double x), double eps, double a, double b);
double iteration(double (*f)(double x), double eps, double a, double b);
double newton(double (*f)(double x), double (*f1)(double x), double eps, double a, double b);

int main(void)
{
	double a1 = 0.0, b1 = 1.0;
	double a2 = 1.0, b2 = 3.0;
	double eps = 0.000001;

	double d1 = dichotomy(f1, eps, a1, b1);
	double i1 = iteration(f1_it, eps, a1, b1);
	double n1 = newton(f1, f1_pr, eps, a1, b1);

	double d2 = dichotomy(f2, eps, a2, b2);
	double i2 = iteration(f2_it, eps, a2, b2);
	double n2 = newton(f2, f2_pr, eps, a2, b2);

	printf("Точность: %.6f\n", eps);
	printf("+-----------+----------+---------------+-----------------------+-----------+----------+---------+\n");
	printf("| Уравнение | Отрезок  | Базовый метод | Прибл. значение корня | Дихотомии | Итераций | Ньютона |\n");
	printf("+-----------+----------+---------------+-----------------------+-----------+----------+---------+\n");
	printf("|     1     |  [0, 1]  |    Ньютона    |         0.8814        |%.9f|%.8f|%.7f|\n", d1, i1, n1);
	printf("+-----------+----------+---------------+-----------------------+-----------+----------+---------+\n");
	printf("|     2     |  [1, 3]  |    Дихотомии  |         1.3749        |%.9f|    -     |%.7f|\n", d2, n2);
	printf("+-----------+----------+---------------+-----------------------+-----------+----------+---------+\n");

	return 0;
}

double f1(double x)
{
	return (exp(x) - exp(-x) - 2.0);
}

double f1_it(double x)
{
	return (log(exp(-x) + 2.0));
}

double f1_pr(double x)
{
	return (exp(x) + exp(-x));
}

double f2(double x)
{
	return (sin(log(x)) - cos(log(x)) + 2.0 * log(x));
}

double f2_it(double x)
{
	// f'(x) >= 1 - нет итерационного метода
	return 0.0;
}

double f2_pr(double x)
{
	return ((cos(log(x)) + sin(log(x)) + 2.0) / 2.0);
}

double dichotomy(double (*f)(double x), double eps, double a, double b)
{
	double infimum = a;
	double supremum = b;
	double tmp;

	while (fabs(infimum - supremum) >= eps)
	{
		tmp = (infimum + supremum) / 2.0;

		if (f(infimum) * f(tmp) > 0.0) infimum = tmp;
		else supremum = tmp;
	}

	return ((infimum + supremum) / 2.0);
}

double iteration(double (*f)(double x), double eps, double a, double b)
{
	double x1 = a;
	double x2 = (a + b) / 2.0;

	while (fabs(x2 - x1) >= eps)
	{
		x1 = x2;
		x2 = f(x1);
	}

	return x1;
}

double newton(double (*f)(double x), double (*f1)(double x), double eps, double a, double b)
{
	double x1 = a;
	double x2 = (a + b) / 2.0;

	while (fabs(x2 - x1) >= eps)
	{
		x1 = x2;
		x2 = x1 - f(x1) / f1(x1);
	}

	return x1;
}
