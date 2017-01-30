#ifndef QUEUE_H
#define QUEUE_H

#include <iostream>
#include <thread>
#include <future>
#include <functional>
#include "queue_item.h"
#include "iterator.h"

template <class T>
class Queue
{
public:
	Queue();
	~Queue();

	void push(const std::shared_ptr<T>& item);
	void pop();
	unsigned int size() const;
	std::shared_ptr<T> front() const;

	Iterator<QueueItem<T>, T> begin() const;
	Iterator<QueueItem<T>, T> end() const;

	void sort();
	void sortParallel();

	template <class K>
	friend std::ostream& operator << (std::ostream& os, const Queue<K>& queue);

private:
	std::shared_ptr<QueueItem<T>> m_front;
	std::shared_ptr<QueueItem<T>> m_end;
	unsigned int m_size;

	void sortHelper(Queue<T>& q, bool isParallel);
	std::future<void> sortParallelHelper(Queue<T>& q);
};

#include "queue_impl.cpp"

#endif
