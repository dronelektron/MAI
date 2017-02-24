#include "gauss.h"

int main(void)
{
	Gauss gauss;

	gaussCreate(&gauss);
	gaussSolve(&gauss);	
	gaussPrintResult(&gauss);
	gaussDelete(&gauss);

	return 0;
}
