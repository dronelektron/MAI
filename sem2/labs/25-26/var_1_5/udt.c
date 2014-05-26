#include "udt.h"

void udtCreate(Udt *udt, const int capacity)
{
	int i;
	UDT_TYPE item;

	item._key = 0.0f;
	item._str[0] = '\0';

	if (capacity <= 0)
		return;

	udt->_data = (UDT_TYPE *)malloc(sizeof(UDT_TYPE) * capacity);

	for (i = 0; i < capacity; i++)
		udt->_data[i] = item;

	udt->_capacity = capacity;
	udt->_size = 0;
}

int udtPush(Udt *udt, const UDT_TYPE value)
{
	if (udt->_size == udt->_capacity)
		return 0;
	
	udt->_data[udt->_size++] = value;

	return 1;
}

void udtPop(Udt *udt)
{
	if (udt->_size == 0)
		return;
	
	udt->_size--;
}

UDT_TYPE udtTop(const Udt *udt)
{
	return udt->_data[udt->_size - 1];
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
	int i;
	Item item;

	printf("+-------+------------+------------------------------+\n");
	printf("| Номер |    Ключ    |            Строка            |\n");
	printf("+-------+------------+------------------------------+\n");

	for (i = 0; i < udtSize(udt); i++)
	{
		item = udt->_data[i];

		printf("|%7d|%12.2f|%30s|\n", i + 1, item._key, item._str);
	}

	printf("+-------+------------+------------------------------+\n");
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
}
