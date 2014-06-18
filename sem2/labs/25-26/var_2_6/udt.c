#include "udt.h"

void udtCreate(Udt *udt, const int size)
{
	if (size <= 1)
		return;

	udt->_data = (UDT_TYPE *)malloc(sizeof(UDT_TYPE) * size);
	udt->_first = 0;
	udt->_size = 0;
	udt->_capacity = size;
}

int udtSize(const Udt *udt)
{
	return udt->_size;
}

int udtCapacity(const Udt *udt)
{
	return udt->_capacity;
}

int udtEmpty(const Udt *udt)
{
	return udt->_size == 0;
}

int udtPush(Udt *udt, const UDT_TYPE value)
{
	if (udt->_size == udt->_capacity)
		return 0;

	udt->_data[(udt->_first + udt->_size) % udt->_capacity] = value;
	udt->_size++;

	return 1;
}

UDT_TYPE udtFront(const Udt *udt)
{
	return udt->_data[udt->_first];
}

void udtPop(Udt *udt)
{
	if (udt->_size == 0)
		return;

	udt->_first = (udt->_first + 1) % udt->_capacity;
	udt->_size--;
}

void udtPrint(Udt *udt)
{
	int i;
	UDT_TYPE tmp;

	for (i = 0; i < udtSize(udt); i++)
	{
		tmp = udt->_data[(udt->_first + i) % udt->_capacity];

		printf("%lf %s\n", tmp._key, tmp._val);
	}
}

void udtDestroy(Udt *udt)
{
	if (udt->_data == NULL)
		return;

	free(udt->_data);

	udt->_data = NULL;
}
