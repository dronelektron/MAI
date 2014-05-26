#ifndef UDT_H
#define UDT_H

#include <stdio.h>
#include <stdlib.h>

typedef struct _Item
{
	float _key;
	char _str[31];
} Item;

typedef Item UDT_TYPE;

typedef struct _Udt
{
	UDT_TYPE *_data;
	int _capacity;
	int _size;
} Udt;

void udtCreate(Udt *udt, const int capacity);
int udtPush(Udt *udt, const UDT_TYPE value);
void udtPop(Udt *udt);
UDT_TYPE udtTop(const Udt *udt);
int udtSize(const Udt *udt);
int udtEmpty(const Udt *udt);
void udtPrint(Udt *udt);
void udtDestroy(Udt *udt);

#endif
