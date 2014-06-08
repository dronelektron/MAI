#ifndef VECTOR_H
#define VECTOR_H

#include <stdlib.h>

typedef struct _Comp
{
	double a;
	double b;
} Comp;

typedef struct _Item
{
	int ind;
	Comp c;
} Item;

typedef Item VECTOR_TYPE;

typedef struct _Vector
{
	VECTOR_TYPE *_data;
	int _size;
	int _capacity;
} Vector;

void vectorCreate(Vector *v, const int size);
int vectorEmpty(const Vector *v);
int vectorSize(const Vector *v);
int vectorCapacity(const Vector *v);
VECTOR_TYPE vectorLoad(const Vector *v, const int index);
void vectorSave(Vector *v, const int index, const VECTOR_TYPE value);
int vectorPushBack(Vector *v, const VECTOR_TYPE value);
void vectorResize(Vector *v, const int size);
//int vectorEqual(const Vector *v1, const Vector *v2);
void vectorDestroy(Vector *v);

#endif
