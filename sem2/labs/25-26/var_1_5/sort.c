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
	Udt tmp;
	UDT_TYPE cur;
	
	udtCreate(&tmp,	size);
	
	while (!udtEmpty(udt))
	{
		udtPush(&tmp, udtTop(udt));
		udtPop(udt);
	}
	
	for (i = 0; i < size / 2; i++)
	{
		cur = udtTop(&tmp);
		
		udtPush(udt1, cur);
		udtPop(&tmp);
	}
	
	for (; i < size; i++)
	{
		cur = udtTop(&tmp);
		
		udtPush(udt2, cur);
		udtPop(&tmp);
	}
	
	udtDestroy(&tmp);
}

void udtMerge(Udt *udt, Udt *udt1, Udt *udt2)
{
	Udt tmp1, tmp2;
	
	udtCreate(&tmp1, udtSize(udt1));
	udtCreate(&tmp2, udtSize(udt2));
	
	while (!udtEmpty(udt1))
	{
		udtPush(&tmp1, udtTop(udt1));
		udtPop(udt1);
	}

	while (!udtEmpty(udt2))
	{
		udtPush(&tmp2, udtTop(udt2));
		udtPop(udt2);
	}
	
	while (!udtEmpty(&tmp1) && !udtEmpty(&tmp2))
	{
		if (udtTop(&tmp1)._key <= udtTop(&tmp2)._key)
		{
			udtPush(udt, udtTop(&tmp1));
			udtPop(&tmp1);
		}
		else
		{
			udtPush(udt, udtTop(&tmp2));
			udtPop(&tmp2);
		}
	}
	
	while (!udtEmpty(&tmp1))
	{
		udtPush(udt, udtTop(&tmp1));
		udtPop(&tmp1);
	}
	
	while (!udtEmpty(&tmp2))
	{
		udtPush(udt, udtTop(&tmp2));
		udtPop(&tmp2);
	}
	
	udtDestroy(&tmp1);
	udtDestroy(&tmp2);
}
