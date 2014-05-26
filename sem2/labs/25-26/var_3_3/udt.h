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
	int _first;
	int _last;
} Udt;

void udtCreate(Udt *udt, const int capacity);
int udtPushFront(Udt *udt, const UDT_TYPE value);
int udtPushBack(Udt *udt, const UDT_TYPE value);
void udtPopFront(Udt *udt);
void udtPopBack(Udt *udt);
UDT_TYPE udtTopFront(const Udt *udt);
UDT_TYPE udtTopBack(const Udt *udt);
int udtSize(const Udt *udt);
int udtEmpty(const Udt *udt);
void udtPrint(Udt *udt);
void udtDestroy(Udt *udt);

#endif
