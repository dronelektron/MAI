#ifndef QUEUE_H
#define QUEUE_H

namespace ds
{
	template<class T>
	class Queue
	{
	public:
		struct QueueItem
		{
			T data;
			QueueItem* next;
		};

		class iterator
		{
		public:
			iterator();
			iterator(QueueItem* item);

			iterator& operator++();
			bool operator!=(const iterator& it);
			T& operator*();
			T* operator->();

		private:
			QueueItem* _cur;
		};

		Queue();
		Queue(const Queue& q);
		~Queue();

		void push(const T& val);
		void pop();
		void clear();
		T& front();
		size_t size() const;
		size_t empty() const;

		Queue& operator=(const Queue& q);

		iterator begin();
		iterator end();

	private:
		QueueItem* _begin;
		QueueItem* _end;
		size_t _size;

		void _copy(const Queue& q);
	};
}

template<class T>
ds::Queue<T>::Queue()
{
	_begin = new QueueItem;
	_begin->next = nullptr;
	_end = _begin;
	_size = 0;
}

template<class T>
ds::Queue<T>::Queue(const Queue& q)
{
	_begin = new QueueItem;
	_end = _begin;
	_size = 0;

	if (this != &q)
		_copy(q);
}

template<class T>
ds::Queue<T>::~Queue()
{
	clear();

	delete _begin;
}

template<class T>
void ds::Queue<T>::push(const T& val)
{
	QueueItem* item = new QueueItem;

	item->next = nullptr;

	_end->data = val;
	_end->next = item;
	_end = item;
	_size++;
}

template<class T>
void ds::Queue<T>::pop()
{
	if (_size == 0)
		return;

	QueueItem* item = _begin;
	_begin = _begin->next;

	delete item;

	_size--;
}

template<class T>
void ds::Queue<T>::clear()
{
	while (_begin->next != nullptr)
	{
		QueueItem* item = _begin;
		_begin = _begin->next;

		delete item;
	}

	_size = 0;
}

template<class T>
T& ds::Queue<T>::front()
{
	return _begin->data;
}

template<class T>
size_t ds::Queue<T>::size() const
{
	return _size;
}

template<class T>
size_t ds::Queue<T>::empty() const
{
	return _size == 0;
}

template<class T>
ds::Queue<T>& ds::Queue<T>::operator=(const Queue& q)
{
	if (this != &q)
	{
		clear();
		_copy(q);
	}

	return *this;
}

template<class T>
void ds::Queue<T>::_copy(const Queue& q)
{
	QueueItem* item = q._begin;

	while (item->next != nullptr)
	{
		QueueItem* tmpItem = new QueueItem;

		tmpItem->next = nullptr;

		_end->data = item->data;
		_end->next = tmpItem;
		_end = tmpItem;

		item = item->next;
	}

	_size = q._size;
}

template<class T>
typename ds::Queue<T>::iterator ds::Queue<T>::begin()
{
	return iterator(_begin);
}

template<class T>
typename ds::Queue<T>::iterator ds::Queue<T>::end()
{
	return iterator(_end);
}

template<class T>
ds::Queue<T>::iterator::iterator()
{
	_cur = nullptr;
}

template<class T>
ds::Queue<T>::iterator::iterator(QueueItem* item)
{
	_cur = item;
}

template<class T>
typename ds::Queue<T>::iterator& ds::Queue<T>::iterator::operator++()
{
	_cur = _cur->next;

	return *this;
}

template<class T>
bool ds::Queue<T>::iterator::operator!=(const iterator& it)
{
	return _cur != it._cur;
}

template<class T>
T& ds::Queue<T>::iterator::operator*()
{
	return _cur->data;
}

template<class T>
T* ds::Queue<T>::iterator::operator->()
{
	return &_cur->data;
}

#endif
