#include "udt.h"

void udtCreate(Udt *udt, const int capacity)
{
	int i;

	if (capacity <= 0)
		return;

	udt->_data = (UDT_TYPE *)malloc(sizeof(UDT_TYPE) * capacity);

	for (i = 0; i < capacity; i++)
		udt->_data[i] = 0;

	udt->_capacity = capacity;
	udt->_size = 0;
	udt->_first = (capacity == 1 ? 0 : 1);
	udt->_last = 0;
}

int udtPushFront(Udt *udt, const UDT_TYPE value)
{
	const int pos = (udt->_first + udt->_capacity - 1) % udt->_capacity;

	if (udt->_size == udt->_capacity)
		return 0;

	udt->_data[pos] = value;
	udt->_first = pos;
	udt->_size++;

	return 1;
}

int udtPushBack(Udt *udt, const UDT_TYPE value)
{
	const int pos = (udt->_last + udt->_capacity + 1) % udt->_capacity;

	if (udt->_size == udt->_capacity)
		return 0;

	udt->_data[pos] = value;
	udt->_last = pos;
	udt->_size++;

	return 1;
}

void udtPopFront(Udt *udt)
{
	const int pos = (udt->_first + udt->_capacity + 1) % udt->_capacity;

	if (udt->_size == 0)
		return;

	udt->_data[udt->_first] = 0;
	udt->_first = pos;
	udt->_size--;
}

void udtPopBack(Udt *udt)
{
	const int pos = (udt->_last + udt->_capacity - 1) % udt->_capacity;

	if (udt->_size == 0)
		return;

	udt->_data[udt->_last] = 0;
	udt->_last = pos;
	udt->_size--;
}

UDT_TYPE udtTopFront(const Udt *udt)
{
	return udt->_data[udt->_first];
}

UDT_TYPE udtTopBack(const Udt *udt)
{
	return udt->_data[udt->_last];
}

int udtSize(const Udt *udt)
{
	return udt->_size;
}

int udtEmpty(const Udt *udt)
{
	return udt->_size == 0;
}

void udtPrint(Udt *udt)
{
	while (!udtEmpty(udt))
	{
		printf("%d ", udtTopFront(udt));

		udtPopFront(udt);
	}
}

void udtDestroy(Udt *udt)
{
	if (udt->_data != NULL)
	{
		free(udt->_data);

		udt->_data = NULL;
	}

	udt->_capacity = 0;
	udt->_size = 0;
	udt->_first = 0;
	udt->_last = 0;
}
