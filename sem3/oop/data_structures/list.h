#ifndef LIST_H
#define LIST_H

#include <cstdlib>

namespace ds
{
	template<class T>
	class List
	{
	public:
		struct ListItem
		{
			T data;
			ListItem* prev;
			ListItem* next;
		};

		class iterator
		{
		public:
			iterator();
			iterator(ListItem* item);

			ListItem* getCur();

			iterator& operator++();
			iterator& operator--();
			bool operator!=(const iterator& it);
			T& operator*();
			T* operator->();

		private:
			ListItem* _cur;
		};

		List();
		List(const List& list);
		~List();

		void insert(iterator it, const T& val = T());
		iterator erase(iterator it);
		void clear();
		size_t size() const;
		size_t empty() const;

		List& operator=(const List& list);

		iterator begin();
		iterator end();

	private:
		ListItem* _begin;
		ListItem* _end;
		size_t _size;

		void _copy(const List& list);
	};
}

template<class T>
ds::List<T>::List()
{
	_begin = new ListItem;
	_begin->next = _begin;
	_begin->prev = _begin;
	_end = _begin;
	_size = 0;
}

template<class T>
ds::List<T>::List(const List& list)
{
	_begin = new ListItem;
	_begin->next = _begin;
	_begin->prev = _begin;
	_end = _begin;
	_size = 0;

	if (this != &list)
		_copy(list);
}

template<class T>
ds::List<T>::~List()
{
	clear();

	delete _begin;
}

template<class T>
void ds::List<T>::insert(iterator it, const T& val)
{
	ListItem* cur = it.getCur();
	ListItem* tmp = new ListItem;

	tmp->data = val;
	cur->prev->next = tmp;
	tmp->next = cur;
	tmp->prev = cur->prev;
	cur->prev = tmp;
	_begin = _end->next;
	_size++;
}

template<class T>     
typename ds::List<T>::iterator ds::List<T>::erase(iterator it)
{
	ListItem* tmp = it.getCur();
	tmp->prev->next = tmp->next;
	tmp->next->prev = tmp->prev;
	_begin = _end->next;
	_size--;

	iterator nextIt(tmp->next);

	delete tmp;

	return nextIt;
}

template<class T>
void ds::List<T>::clear()
{
	for (size_t i = 0; i < _size; ++i)
	{
		ListItem* cur = _begin;
		_begin = _begin->next;

		delete cur;
	}

	_size = 0;
}

template<class T>
size_t ds::List<T>::size() const
{
	return _size;
}

template<class T>
size_t ds::List<T>::empty() const
{
	return _size == 0;
}

template<class T>
ds::List<T>& ds::List<T>::operator=(const List& list)
{
	if (this != &list)
	{
		clear();
		_copy(list);
	}

	return *this;
}

template<class T>
void ds::List<T>::_copy(const List& list)
{
	ListItem* item = list._end->prev;
	ListItem* cur = _end;
	
	for (size_t i = 0; i < list._size; ++i)
	{
		ListItem* tmp = new ListItem;
		tmp->data = item->data;
		tmp->next = cur;
		cur->prev = tmp;

		cur = tmp;
		item = item->prev;
	}

	cur->prev = _end;
	_end->next = cur;
	_begin = cur;
	_size = list._size;
}

template<class T>
typename ds::List<T>::iterator ds::List<T>::begin()
{
	return iterator(_begin);
}

template<class T>
typename ds::List<T>::iterator ds::List<T>::end()
{
	return iterator(_end);
}

template<class T>
ds::List<T>::iterator::iterator()
{
	_cur = NULL;
}

template<class T>
ds::List<T>::iterator::iterator(ListItem* item)
{
	_cur = item;
}

template<class T>
typename ds::List<T>::ListItem* ds::List<T>::iterator::getCur()
{
	return _cur;
}

template<class T>
typename ds::List<T>::iterator& ds::List<T>::iterator::operator++()
{
	_cur = _cur->next;

	return *this;
}

template<class T>
typename ds::List<T>::iterator& ds::List<T>::iterator::operator--()
{
	_cur = _cur->prev;

	return *this;
}

template<class T>
bool ds::List<T>::iterator::operator!=(const iterator& it)
{
	return _cur != it._cur;
}

template<class T>
T& ds::List<T>::iterator::operator*()
{
	return _cur->data;
}

template<class T>
T* ds::List<T>::iterator::operator->()
{
	return &_cur->data;
}

#endif
