#include "mlisp.h"

double baaFib(double n);
double fib__iter(double a, double b, double n);

int main()
{
	display("baaFib(47)=");
	newline();
	display(baaFib(47));
	newline();

	display("baaFib(5)=");
	newline();
	display(baaFib(5));
	newline();

	return 0;
}

double baaFib(double n)
{
	{
		return fib__iter(1, 0, n);
	}
}

double fib__iter(double a, double b, double n)
{
	{
		return n == 0 ? b : fib__iter(a + b, a, n - 1);
	}
}
