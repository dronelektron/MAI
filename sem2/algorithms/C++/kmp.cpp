#include <iostream>
#include <vector>
#include <string>

std::vector<int> prefix(const std::string &s);
int KMP(const std::string &s1, const std::string &s2);

int main()
{
	std::string s1, s2;

	std::getline(std::cin, s1);
	std::getline(std::cin, s2);

	std::cout << KMP(s1, s2) << std::endl;
	
	return 0;
}

std::vector<int> prefix(const std::string &s)
{
	const int N = s.length();
	std::vector<int> overlap(N);

	overlap[0] = 0;

	for (int i = 1; i < N; i++)
	{
		int j = overlap[i - 1];

		while (j > 0 && s[i] != s[j])
			j = overlap[j - 1];

		if (s[i] == s[j])
			j++;

		overlap[i] = j;
	}

	return overlap;
}

int KMP(const std::string &s1, const std::string &s2)
{
	const int N = s1.length(), M = s2.length();
	std::vector<int> pref = prefix(s2);

	int j = 0;

	for (int i = 0; i < N; i++)
	{
		while (j > 0 && s1[i] != s2[j])
			j = pref[j - 1];

		if (s1[i] == s2[j])
			j++;

		if (j >= M)
			return i - M + 1;
	}

	return -1;
}
