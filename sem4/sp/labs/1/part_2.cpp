#include "mlisp.h"

double count__change(double amount);
double cc(double amount, double kinds__of__coins);
double first__denomination(double kinds__of__coins);

int main()
{
	display("baa variant 5:");
	newline();
	display(count__change(100));
	newline();

	return 0;
}

double count__change(double amount)
{
	{
		return cc(amount, 5);
	}
}

double cc(double amount, double kinds__of__coins)
{
	{
		return amount == 0 ? 1 :
			(amount < 0) || (kinds__of__coins == 0) ? 0 :
			cc(amount, kinds__of__coins - 1) +
			cc(amount - first__denomination(kinds__of__coins), kinds__of__coins);
	}
}

double first__denomination(double kinds__of__coins)
{
	{
		return kinds__of__coins == 1 ? 2 :
			kinds__of__coins == 2 ? 3 :
			kinds__of__coins == 3 ? 5 :
			kinds__of__coins == 4 ? 25 :
			kinds__of__coins == 5 ? 50 :
			_infinity;
	}
}
