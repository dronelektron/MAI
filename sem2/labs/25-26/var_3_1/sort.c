#include "sort.h"

void udtSwap(Udt *udt1, Udt *udt2)
{
	Udt tmp;

	tmp = *udt1;
	*udt1 = *udt2;
	*udt2 = tmp;
}

void udtSelectionSort(Udt *udt)
{
	const int cap = udtCapacity(udt);
	Udt sorted, tmp;
	UDT_TYPE item;

	if (cap < 2)
		return;

	udtCreate(&sorted, cap);
	udtCreate(&tmp, cap);

	while (!udtEmpty(udt))
	{
		udtPushFront(&tmp, udtTopFront(udt));
		udtPopFront(udt);

		while (!udtEmpty(udt))
		{
			item = udtTopFront(udt);

			udtPopFront(udt);

			if (item._key < udtTopFront(&tmp)._key)
				udtPushFront(&tmp, item);
			else
				udtPushBack(&tmp, item);
		}

		udtPushBack(&sorted, udtTopFront(&tmp));
		udtPopFront(&tmp);
		
		udtSwap(udt, &tmp);
	}

	udtSwap(udt, &sorted);

	udtDestroy(&sorted);
	udtDestroy(&tmp);
}
