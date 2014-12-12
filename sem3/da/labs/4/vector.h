#ifndef VECTOR_H
#define VECTOR_H

#include <exception>
#include <new>

template<class T>
struct TVector
{
	T* begin;
	int cap;
	int size;
};

template<class T>
void VectorInit(TVector<T>& v)
{
	try
	{
		v.begin = new T[1];
	}
	catch (const std::bad_alloc& e)
	{
		printf("ERROR: No memory\n");

		std::exit(0);
	}

	v.cap = 1;
	v.size = 0;
}

template<class T>
void VectorPushBack(TVector<T>& v, const T& val)
{
	if (v.size == v.cap)
	{
		v.cap *= 2;
		T* v2 = NULL;

		try
		{
			v2 = new T[v.cap];
		}
		catch (const std::bad_alloc& e)
		{
			printf("ERROR: No memory\n");

			std::exit(0);
		}

		for (int i = 0; i < v.size; ++i)
		{
			v2[i] = v.begin[i];
		}

		delete[] v.begin;

		v.begin = v2;
	}

	v.begin[v.size++] = val;
}

template<class T>
void VectorDestroy(TVector<T>& v)
{
	delete[] v.begin;
}

#endif
