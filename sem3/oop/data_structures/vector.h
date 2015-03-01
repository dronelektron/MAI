#ifndef VECTOR_H
#define VECTOR_H

namespace ds
{
	template<class T>
	class Vector
	{
	public:
		Vector();
		Vector(const Vector& v);
		Vector(size_t n, const T& val = T());
		~Vector();
		
		void push_back(const T& val);
		void erase(size_t index);
		void resize(size_t n, const T& val = T());
		void clear();
		size_t size() const;
		bool empty() const;

		T& operator[](size_t i);
		Vector& operator=(const Vector& v);

	private:
		T* _begin;
		size_t _size;
		size_t _cap;

		void _copy(const Vector& v);
	};
}

template<class T>
ds::Vector<T>::Vector()
{
	_begin = new T[1];
	_size = 0;
	_cap = 1;
}

template<class T>
ds::Vector<T>::Vector(const Vector& v)
{
	_begin = new T[1];
	_size = 0;
	_cap = 1;

	if (this != &v)
		_copy(v);
}

template<class T>
ds::Vector<T>::Vector(size_t n, const T& val)
{
	_begin = new T[n + 1];
	
	for (size_t i = 0; i < n; i++)
		_begin[i] = val;

	_size = n;
	_cap = n + 1;
}

template<class T>
ds::Vector<T>::~Vector()
{
	delete[] _begin;
}

template<class T>
void ds::Vector<T>::push_back(const T& val)
{
	if (_size == _cap)
	{
		size_t oldSize = _size;

		resize(_size * 2);

		_size = oldSize;
	}

	_begin[_size++] = val;
}

template<class T>
void ds::Vector<T>::erase(size_t index)
{
	for (size_t i = index; i < _size - 1; i++)
		_begin[i] = _begin[i + 1];

	_size--;
}

template<class T>
void ds::Vector<T>::resize(size_t n, const T& val)
{
	const size_t copySize = n < _size ? n : _size;
	T* _buffer = new T[n + 1];
	
	for (size_t i = 0; i < copySize; i++)
		_buffer[i] = _begin[i];

	for (size_t i = copySize; i < n; i++)
		_buffer[i] = val;

	delete [] _begin;

	_begin = _buffer;
	_size = n;
	_cap = n + 1;
}

template<class T>
void ds::Vector<T>::clear()
{
	delete [] _begin;

	_begin = new T[1];
	_size = 0;
	_cap = 1;
}

template<class T>
size_t ds::Vector<T>::size() const
{
	return _size;
}

template<class T>
bool ds::Vector<T>::empty() const
{
	return _size == 0;
}

template<class T>
T& ds::Vector<T>::operator[](size_t i)
{
	return _begin[i];
}

template<class T>
ds::Vector<T>& ds::Vector<T>::operator=(const Vector& v)
{
	if (this != &v)
		_copy(v);

	return *this;
}

template<class T>
void ds::Vector<T>::_copy(const Vector& v)
{
	const size_t n = v.size();
	
	delete[] _begin;
	
	_begin = new T[n + 1];

	for (size_t i = 0; i < n; i++)
		_begin[i] = v._begin[i];
	
	_size = n;
	_cap = n + 1;
}

#endif
