#ifndef LIST_H
#define LIST_H

#include <iostream>
#include "list_item.h"
#include "iterator.h"

template <class T>
class List
{
public:
	List();
	~List();

	void add(const std::shared_ptr<T>& item);
	void erase(const Iterator<ListItem<T>, T>& it);
	unsigned int size() const;
	Iterator<ListItem<T>, T> get(unsigned int index) const;

	Iterator<ListItem<T>, T> begin() const;
	Iterator<ListItem<T>, T> end() const;

	template <class K>
	friend std::ostream& operator << (std::ostream& os, const List<K>& list);

private:
	std::shared_ptr<ListItem<T>> m_begin;
	std::shared_ptr<ListItem<T>> m_end;
	unsigned int m_size;
};

#include "list_impl.cpp"

#endif
