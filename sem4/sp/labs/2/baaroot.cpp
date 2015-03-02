#include "mlisp.h"

extern double dx;
extern double tolerance;

double dx = 1e-5;
double tolerance = 1e-5;

double fixed__point(double first__guess);
double __baa__try(double guess);
double close__enough(double x, double y);
double newton__transform(double x);
double deriv(double x);
double fun(double x);
double root(double first__guess);

int main()
{
	display("baa variant 1");
	newline();
	display(root(4));
	newline();

	return 0;	
}

double fixed__point(double first__guess)
{
	{
		return __baa__try(first__guess);
	}
}

double __baa__try(double guess)
{
	{
		double next = newton__transform(guess);

		display("+");

		return close__enough(guess, next) ? next : __baa__try(next);
	}
}

double close__enough(double x, double y)
{
	{
		return abs(x - y) < tolerance;
	}
}

double newton__transform(double x)
{
	{
		return (x - fun(x) / deriv(x));
	}
}

double deriv(double x)
{
	{
		return (fun(x + dx) - fun(x)) / dx;
	}
}

double fun(double x)
{
	{
		double z = x - (double)101 / 102;

		return exp(z) + log(z) - 10 * z;
	}
}

double root(double first__guess)
{
	{
		double temp = fixed__point(first__guess);
		
		newline();
		display("first-guess=\t");
		display(first__guess);
		newline();
		display("discrepancy=\t");
		display(fun(temp));
		newline();
		display("root=\t\t");

		return temp;
	}
}
