#include "sort.h"

void udtSwap(Udt *udt1, Udt *udt2)
{
	Udt tmp;

	tmp = *udt1;
	*udt1 = *udt2;
	*udt2 = tmp;
}

void udtSort(Udt *udt)
{
	const int cap = udtCapacity(udt);
	UDT_TYPE max;
	Udt second;
	Udt third;
	Udt tmp;

	if (cap < 2)
		return;

	udtCreate(&second, cap);
	udtCreate(&third, cap);
	udtCreate(&tmp, cap);

	max = udtFront(udt);

	udtPop(udt);
	udtPush(&second, max);

	while (!udtEmpty(udt))
	{
		if (udtFront(udt)._key > max._key)
		{
			max = udtFront(udt);

			udtPop(udt);
			udtPush(&second, max);
		}
		else
		{
			while (udtFront(&second)._key < udtFront(udt)._key)
			{
				udtPush(&third, udtFront(&second));
				udtPop(&second);
			}

			udtPush(&third, udtFront(udt));
			udtPop(udt);

			while (!udtEmpty(&second))
			{
				udtPush(&third, udtFront(&second));
				udtPop(&second);
			}

			udtSwap(&third, &second);
		}
	}

	udtSwap(&second, udt);

	udtDestroy(&second);
	udtDestroy(&third);
	udtDestroy(&tmp);
}
