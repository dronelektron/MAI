#include "vector.h"

void vectorCreate(Vector *v, const int size)
{
	v->_size = size;
	v->_data = (int *)malloc(sizeof(int) * v->_size);
}

int vectorEmpty(const Vector *v)
{
	return v->_size == 0;
}

int vectorSize(const Vector *v)
{
	return v->_size;
}

VECTOR_TYPE vectorLoad(const Vector *v, const int index)
{
	return v->_data[index];
}

void vectorSave(Vector *v, const int index, const VECTOR_TYPE value)
{
	v->_data[index] = value;
}

void vectorResize(Vector *v, const int size)
{
	v->_size = size;
	v->_data = (int *)realloc(v->_data, sizeof(int) * v->_size);
}

int vectorEqual(const Vector *v1, const Vector *v2)
{
	int i;

	if (v1->_size != v2->_size)
		return 0;

	for (i = 0; i < v1->_size; i++)
		if (v1->_data[i] != v2->_data[i])
			return 0;

	return 1;
}

void vectorDestroy(Vector *v)
{
	v->_size = 0;

	free(v->_data);
}
