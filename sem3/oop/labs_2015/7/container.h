#ifndef CONTAINTER_H
#define CONTAINTER_H

#include <memory>
#include <cstring>
#include "queue.h"
#include "list.h"
#include "criteria.h"

template <class T>
class Container
{
public:
	void add(const std::shared_ptr<T>& item);
	void erase(const Criteria<T>& criteria);
	//void print() const;

	template <class K>
	friend std::ostream& operator << (std::ostream& os, const Container<K>& container);

private:
	Queue<List<T>> m_container;
};

#include "container_impl.cpp"

#endif
