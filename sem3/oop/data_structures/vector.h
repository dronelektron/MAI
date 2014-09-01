#ifndef VECTOR_H
#define VECTOR_H

namespace ds
{
	template<class T>
	class Vector
	{
		public:
			explicit Vector();
			explicit Vector(const Vector& v);
			Vector(size_t n, T val = T());
			~Vector();

			void push_back(T val);
			void resize(size_t n, T val = T());
			size_t size() const;

			T& operator[](size_t i);
			Vector& operator=(const Vector& v);

		private:
			T* _begin;
			size_t _size;

			void _copy(const Vector& v);
	};
}

template<class T>
ds::Vector<T>::Vector()
{
	_begin = nullptr;
	_size = 0;
}

template<class T>
ds::Vector<T>::Vector(const Vector& v)
{
	_copy(v);
}

template<class T>
ds::Vector<T>::Vector(size_t n, T val)
{
	_begin = nullptr;

	if (n > 0)
	{	
		_begin = new T[n];
		
		for (size_t i = 0; i < n; i++)
			_begin[i] = val;
	}

	_size = n;
}

template<class T>
ds::Vector<T>::~Vector()
{
	if (_begin != nullptr)
		delete [] _begin;
}

template<class T>
void ds::Vector<T>::push_back(T val)
{
	resize(_size + 1);

	_begin[_size - 1] = val;
}

template<class T>
void ds::Vector<T>::resize(size_t n, T val)
{
	const size_t copySize = n < _size ? n : _size;
	T* _buffer = nullptr;
	
	if (n > 0)
		_buffer = new T[n];

	for (size_t i = 0; i < copySize; i++)
		_buffer[i] = _begin[i];

	for (size_t i = copySize; i < n; i++)
		_buffer[i] = val;

	if (_begin != nullptr)
		delete [] _begin;

	_begin = _buffer;
	_size = n;
}

template<class T>
size_t ds::Vector<T>::size() const
{
	return _size;
}

template<class T>
T& ds::Vector<T>::operator[](size_t i)
{
	return _begin[i];
}

template<class T>
ds::Vector<T>& ds::Vector<T>::operator=(const Vector& v)
{
	_copy(v);

	return *this;
}

template<class T>
void ds::Vector<T>::_copy(const Vector& v)
{
	if (this != &v)
	{
		const size_t n = v.size();

		_begin = nullptr;

		if (n > 0)
		{
			_begin = new T[n];

			for (size_t i = 0; i < n; i++)
				_begin[i] = v._begin[i];
		}

		_size = v._size;
	}
}

#endif
