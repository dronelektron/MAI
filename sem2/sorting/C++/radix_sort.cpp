#include <iostream>
#include <vector>

void counterSort(std::vector<int> &v, const int shift);
void radixSort(std::vector<int> &v);

// С поддержкой отрицательных чисел
void radixSortSign(std::vector<int> &v);

int main()
{
	int n;
	std::vector<int> v;
	
	std::cin >> n;
	
	v.resize(n);
	
	for (int i = 0; i < n; i++)
		std::cin >> v[i];

	radixSort(v);
	
	for (int i = 0; i < v.size(); i++)
		std::cout << v[i] << " ";
	
	std::cout << std::endl;

	return 0;
}

void counterSort(std::vector<int> &v, const int shift)
{
	const int max = 255;

	std::vector<int> a(v.begin(), v.end());
	std::vector<int> c(max + 1, 0);

	for (int i = 0; i < v.size(); i++)
		c[(v[i] >> (8 * shift)) & max]++;

	for (int i = 1; i < c.size(); i++)
		c[i] += c[i - 1];

	for (int i = v.size() - 1; i >= 0; i--)
	{
		v[c[(a[i] >> (8 * shift)) & max] - 1] = a[i];
		c[(a[i] >> (8 * shift)) & max]--;
	}

	return;
}

void radixSort(std::vector<int> &v)
{
	for (int i = 0; i < sizeof(int); i++)
		counterSort(v, i);
}

void radixSortSign(std::vector<int> &v)
{
	int minusCnt = 0;

	for (int i = 0; i < sizeof(int); i++)
		counterSort(v, i);

	for (int i = v.size() - 1; i >= 0 && v[i] < 0; i--, minusCnt++);
	
	if (minusCnt > 0 && minusCnt < v.size())
	{
		std::vector<int> tmp(v.begin(), v.end());

		std::copy(tmp.end() - minusCnt, tmp.end(), v.begin());
		std::copy(tmp.begin(), tmp.end() - minusCnt, v.begin() + minusCnt);
	}
}
