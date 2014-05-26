#include "sort.h"

void udtMergeSort(Udt *udt)
{
	const int size = udtSize(udt);
	Udt left, right;
	
	if (size <= 1)
		return;
	
	udtCreate(&left, size);
	udtCreate(&right, size);
	
	udtPartition(udt, &left, &right);
	udtMergeSort(&left);
	udtMergeSort(&right);
	udtMerge(udt, &left, &right);

	udtDestroy(&left);
	udtDestroy(&right);
}

void udtPartition(Udt *udt, Udt *udt1, Udt *udt2)
{
	const int size = udtSize(udt);
	int i;
	UDT_TYPE cur;
	
	for (i = 0; i < size / 2; i++)
	{	
		udtPushBack(udt1, udtTopFront(udt));
		udtPopFront(udt);
	}

	for (; i < size; i++)
	{
		udtPushBack(udt2, udtTopFront(udt));
		udtPopFront(udt);
	}
}

void udtMerge(Udt *udt, Udt *udt1, Udt *udt2)
{
	while (!udtEmpty(udt1) && !udtEmpty(udt2))
	{
		if (udtTopFront(udt1)._key <= udtTopFront(udt2)._key)
		{
			udtPushBack(udt, udtTopFront(udt1));
			udtPopFront(udt1);
		}
		else
		{
			udtPushBack(udt, udtTopFront(udt2));
			udtPopFront(udt2);
		}
	}
	
	while (!udtEmpty(udt1))
	{
		udtPushBack(udt, udtTopFront(udt1));
		udtPopFront(udt1);
	}
	
	while (!udtEmpty(udt2))
	{
		udtPushBack(udt, udtTopFront(udt2));
		udtPopFront(udt2);
	}
}
