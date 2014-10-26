#include <cstdio>
#include <cstdlib>

typedef unsigned short TKey;
typedef unsigned long long TVal;

struct TData
{
	TKey key;
	TVal val;
};

template<class T>
struct TVector
{
	T* begin;
	size_t size;
	size_t cap;
};

template<class T>
void VectorCreate(TVector<T>& v, size_t size);

template<class T>
void VectorPushBack(TVector<T>& v, TData val);

template<class T>
void VectorDestroy(TVector<T>& v);

template<class T>
void MySwap(T& a, T& b);

void countingSort(TVector<TData>& v);

int main(void)
{
	TVector<TData> v;
	TData data;

	VectorCreate<TData>(v, 0);

	while (scanf("%hu\t%llu", &data.key, &data.val) == 2)
	{
		VectorPushBack<TData>(v, data);
	}

	countingSort(v);
	
	for (size_t i = 0; i < v.size; ++i)
	{
		printf("%hu\t%llu\n", v.begin[i].key, v.begin[i].val);
	}
	
	VectorDestroy<TData>(v);

	return 0;
}

template<class T>
void VectorCreate(TVector<T>& v, size_t size)
{
	v.begin = new T[size + 1];
	v.cap = size + 1;
	v.size = size;
}

template<class T>
void VectorPushBack(TVector<T>& v, TData val)
{
	if (v.size == v.cap)
	{
		v.cap *= 2;
		T* v2 = new T[v.cap];
		
		for (size_t i = 0; i < v.size; ++i)
		{
			v2[i] = v.begin[i];
		}
		
		delete[] v.begin;

		v.begin = v2;
	}

	v.begin[v.size++] = val;
}

template<class T>
void VectorDestroy(TVector<T>& v)
{
	delete[] v.begin;

	v.begin = NULL;
	v.size = 0;
	v.cap = 0;
}

template<class T>
void MySwap(T& a, T& b)
{
	T c = a;
	a = b;
	b = c;
}

void countingSort(TVector<TData>& v)
{
	if (v.size < 2)
	{
		return;
	}

	TKey k = v.begin[0].key;

	for (size_t i = 0; i < v.size; ++i)
	{
		if (v.begin[i].key > k)
		{
			k = v.begin[i].key;
		}
	}

	TVector<TData> v2;
	TVector<size_t> cnt;

	VectorCreate<TData>(v2, v.size);
	VectorCreate<size_t>(cnt, k + 1);

	for (size_t i = 0; i <= k; ++i)
	{
		cnt.begin[i] = 0;
	}

	for (size_t i = 0; i < v.size; ++i)
	{
		++cnt.begin[v.begin[i].key];
	}

	for (size_t i = 1; i <= k; ++i)
	{
		cnt.begin[i] += cnt.begin[i - 1];
	}

	for (size_t i = v.size; i > 0; --i)
	{
		v2.begin[--cnt.begin[v.begin[i - 1].key]] = v.begin[i - 1];
	}

	MySwap<TVector<TData> >(v, v2);

	VectorDestroy(v2);
	VectorDestroy(cnt);
}
