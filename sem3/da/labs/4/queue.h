#ifndef QUEUE_H
#define QUEUE_H

#include <exception>
#include <new>

template<class T>
struct TQueue
{
	T* begin;
	int cap;
	int size;
	int offset;
};

template<class T>
void QueueInit(TQueue<T>& q, int size)
{
	try
	{
		q.begin = new T[size];
	}
	catch (const std::bad_alloc& e)
	{
		printf("ERROR: No memory\n");

		std::exit(0);
	}

	q.cap = size;
	q.size = 0;
	q.offset = 0;
}

template<class T>
void QueuePush(TQueue<T>& q, const T& val)
{
	q.begin[(q.offset + q.size++) % q.cap] = val;
}

template<class T>
void QueuePop(TQueue<T>& q)
{
	q.offset = (q.offset + 1) % q.cap;
	--q.size;
}

template<class T>
void QueueDestroy(TQueue<T>& q)
{
	delete[] q.begin;
}

#endif
