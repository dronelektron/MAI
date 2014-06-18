#ifndef UDT_H
#define UDT_H

#include <stdio.h>
#include <stdlib.h>

typedef struct _Item
{
	double _key;
	char _val[81];
} Item;

typedef Item UDT_TYPE;

typedef struct _Udt
{
	UDT_TYPE *_data;
	int _first;
	int _size;
	int _capacity;
} Udt;

void udtCreate(Udt *udt, const int size);
int udtSize(const Udt *udt);
int udtCapacity(const Udt *udt);
int udtEmpty(const Udt *udt);
int udtPush(Udt *udt, const UDT_TYPE value);
UDT_TYPE udtFront(const Udt *udt);
void udtPop(Udt *udt);
void udtPrint(Udt *udt);
void udtDestroy(Udt *udt);

#endif
