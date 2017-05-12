#include "array.h"

int main(void)
{
	Array arr;

	arrayRead(&arr);
	arraySort(&arr);
	arrayWrite(&arr);

	return 0;
}
