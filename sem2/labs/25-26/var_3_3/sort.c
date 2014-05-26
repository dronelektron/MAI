#include "sort.h"

void udtConcat(Udt *udt, Udt *udt1, Udt *udt2, UDT_TYPE pivot)
{
	while (!udtEmpty(udt1))
	{
		udtPushBack(udt, udtTopFront(udt1));
		udtPopFront(udt1);
	}

	udtPushBack(udt, pivot);

	while (!udtEmpty(udt2))
	{
		udtPushBack(udt, udtTopFront(udt2));
		udtPopFront(udt2);
	}
}

void udtQuickSort(Udt *udt)
{
	const int size = udtSize(udt);
	UDT_TYPE pivot, top;
	Udt left, right;

	if (size <= 1)
		return;

	pivot = udtTopFront(udt);

	udtPopFront(udt);

	udtCreate(&left, size - 1);
	udtCreate(&right, size - 1);

	while (!udtEmpty(udt))
	{
		top = udtTopFront(udt);

		udtPopFront(udt);

		if (top._key < pivot._key)
			udtPushBack(&left, top);
		else
			udtPushBack(&right, top);
	}

	udtQuickSort(&left);
	udtQuickSort(&right);
	udtConcat(udt, &left, &right, pivot);
	udtDestroy(&left);
	udtDestroy(&right);
}
