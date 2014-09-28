#ifndef STACK_H
#define STACK_H

#include <cstdlib>

namespace ds
{
	template<class T>
	class Stack
	{
	public:
		struct Item
		{
			T data;
			Item* prev;
		};

		Stack();
		Stack(const Stack& s);
		~Stack();

		void push(const T& val);
		void pop();
		void clear();
		T& top();
		size_t size() const;
		bool empty() const;

		Stack& operator=(const Stack& s);

	private:
		Item* _top;
		size_t _size;

		void _copy(const Stack& s);
	};
}

template<class T>
ds::Stack<T>::Stack()
{
	_top = NULL;
	_size = 0;
}

template<class T>
ds::Stack<T>::Stack(const Stack& s)
{
	_top = NULL;
	_size = 0;

	if (this != &s)
		_copy(s);
}

template<class T>
ds::Stack<T>::~Stack()
{
	clear();
}

template<class T>
void ds::Stack<T>::push(const T& val)
{
	Item* item = new Item;

	item->data = val;
	item->prev = _top;
	_top = item;
	_size++;
}

template<class T>
void ds::Stack<T>::pop()
{
	Item* item = _top;

	_top = _top->prev;
	_size--;

	delete item;
}

template<class T>
void ds::Stack<T>::clear()
{
	while (_top != NULL)
	{
		Item* item = _top;

		_top = _top->prev;

		delete item;
	}

	_top = NULL;
	_size = 0;
}

template<class T>
T& ds::Stack<T>::top()
{
	return _top->data;
}

template<class T>
size_t ds::Stack<T>::size() const
{
	return _size;
}

template<class T>
bool ds::Stack<T>::empty() const
{
	return _size == 0;
}

template<class T>
ds::Stack<T>& ds::Stack<T>::operator=(const Stack& s)
{
	if (this != &s)
		_copy(s);

	return *this;
}

template<class T>
void ds::Stack<T>::_copy(const Stack& s)
{
	Item* item = s._top;
	Item* term = new Item;
	_top = term;

	term->prev = NULL;

	while (item != NULL)
	{
		_top->prev = new Item;
		_top->prev->data = item->data;
		_top->prev->prev = NULL;
		_top = _top->prev;
		item = item->prev;
	}

	_top = term->prev;
	_size = s._size;

	delete term;
}

#endif
