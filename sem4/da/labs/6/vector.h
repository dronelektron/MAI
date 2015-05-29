#ifndef VECTOR_H
#define VECTOR_H

#include <exception>
#include <new>
#include <cstdlib>
#include <cstdio>

namespace NDS
{
	template<class T>
	class TVector
	{
	public:
		TVector();
		TVector(const TVector& v);
		TVector(size_t n, const T& val = T());
		~TVector();
		
		void PushBack(const T& val);
		void PopBack();
		void Erase(size_t index);
		void Resize(size_t n, const T& val = T());
		void Clear();
		size_t Size() const;
		bool Empty() const;
		T& Back();
		const T& Back() const;

		T& operator [] (size_t i);
		const T& operator [] (size_t i) const;
		TVector& operator = (const TVector& v);

	private:
		T* mBegin;
		size_t mSize;
		size_t mCap;

		void mCopy(const TVector& v);
	};
}

template<class T>
NDS::TVector<T>::TVector()
{
	try
	{
		mBegin = new T[1];
	}
	catch (const std::bad_alloc& e)
	{
		printf("Error\n");
		std::exit(0);
	}

	mSize = 0;
	mCap = 1;
}

template<class T>
NDS::TVector<T>::TVector(const TVector& v)
{
	try
	{
		mBegin = new T[1];
	}
	catch (const std::bad_alloc& e)
	{
		printf("Error\n");
		std::exit(0);
	}

	mSize = 0;
	mCap = 1;

	if (this != &v)
	{
		mCopy(v);
	}
}

template<class T>
NDS::TVector<T>::TVector(size_t n, const T& val)
{
	try
	{
		mBegin = new T[n + 1];
	}
	catch (const std::bad_alloc& e)
	{
		printf("Error\n");
		std::exit(0);
	}
	
	for (size_t i = 0; i < n; ++i)
	{
		mBegin[i] = val;
	}

	mSize = n;
	mCap = n + 1;
}

template<class T>
NDS::TVector<T>::~TVector()
{
	delete[] mBegin;
}

template<class T>
void NDS::TVector<T>::PushBack(const T& val)
{
	if (mSize == mCap)
	{
		size_t oldSize = mSize;

		Resize(mSize * 2);

		mSize = oldSize;
	}

	mBegin[mSize++] = val;
}

template<class T>
void NDS::TVector<T>::PopBack()
{
	--mSize;
}

template<class T>
void NDS::TVector<T>::Erase(size_t index)
{
	for (size_t i = index; i < mSize - 1; ++i)
	{
		mBegin[i] = mBegin[i + 1];
	}

	--mSize;
}

template<class T>
void NDS::TVector<T>::Resize(size_t n, const T& val)
{
	const size_t copySize = n < mSize ? n : mSize;
	T* buffer = NULL;

	try
	{
		buffer = new T[n + 1];
	}
	catch (const std::bad_alloc& e)
	{
		printf("Error\n");
		std::exit(0);
	}

	for (size_t i = 0; i < copySize; ++i)
	{
		buffer[i] = mBegin[i];
	}

	for (size_t i = copySize; i < n; ++i)
	{
		buffer[i] = val;
	}

	delete [] mBegin;

	mBegin = buffer;
	mSize = n;
	mCap = n + 1;
}

template<class T>
void NDS::TVector<T>::Clear()
{
	delete [] mBegin;

	try
	{
		mBegin = new T[1];
	}
	catch (const std::bad_alloc& e)
	{
		printf("Error\n");
		std::exit(0);
	}

	mSize = 0;
	mCap = 1;
}

template<class T>
size_t NDS::TVector<T>::Size() const
{
	return mSize;
}

template<class T>
bool NDS::TVector<T>::Empty() const
{
	return mSize == 0;
}

template<class T>
T& NDS::TVector<T>::Back()
{
	return mBegin[mSize - 1];
}

template<class T>
const T& NDS::TVector<T>::Back() const
{
	return mBegin[mSize - 1];
}

template<class T>
T& NDS::TVector<T>::operator [] (size_t i)
{
	return mBegin[i];
}

template<class T>
const T& NDS::TVector<T>::operator [] (size_t i) const
{
	return mBegin[i];
}

template<class T>
NDS::TVector<T>& NDS::TVector<T>::operator = (const TVector& v)
{
	if (this != &v)
	{
		mCopy(v);
	}

	return *this;
}

template<class T>
void NDS::TVector<T>::mCopy(const TVector& v)
{
	const size_t n = v.Size();

	delete[] mBegin;

	try
	{
		mBegin = new T[n + 1];
	}
	catch (const std::bad_alloc& e)
	{
		printf("Error\n");
		std::exit(0);
	}
	
	for (size_t i = 0; i < n; ++i)
	{
		mBegin[i] = v.mBegin[i];
	}
	
	mSize = n;
	mCap = n + 1;
}

#endif
