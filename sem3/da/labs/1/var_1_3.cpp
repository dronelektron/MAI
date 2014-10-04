#include <cstdio>
#include <cstdlib>
#include "vector.h"

struct TData
{
	unsigned short key;
	unsigned long long val;
};

void countingSort(NAndy::TVector<TData>& v);

int main()
{
	NAndy::TVector<TData> arr;
	TData item;

	while (scanf("%hu\t%llu", &item.key, &item.val) == 2)
	{
		arr.push_back(item);
	}

	countingSort(arr);

	for (size_t i = 0; i < arr.size(); ++i)
	{
		printf("%hu\t%llu\n", arr[i].key, arr[i].val);
	}

	return 0;
}

void countingSort(NAndy::TVector<TData>& v)
{
	if (v.size() < 2)
	{
		return;
	}

	unsigned short k = v[0].key;

	for (size_t i = 0; i < v.size(); ++i)
	{
		if (v[i].key > k)
		{
			k = v[i].key;
		}
	}

	NAndy::TVector<TData> v2(v);
	NAndy::TVector<size_t> cnt(k + 1, 0);

	for (size_t i = 0; i < v2.size(); ++i)
	{
		++cnt[v2[i].key];
	}

	for (size_t i = 1; i <= k; ++i)
	{
		cnt[i] += cnt[i - 1];
	}

	for (size_t i = v2.size(); i > 0; --i)
	{
		v[cnt[v2[i - 1].key] - 1] = v2[i - 1];
		--cnt[v2[i - 1].key];
	}
}
