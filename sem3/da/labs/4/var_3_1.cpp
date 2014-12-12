#include <exception>
#include <new>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "vector.h"
#include "queue.h"

struct TPos
{
	int row;
	int col;
};

int ToLower(int ch);
void PreBmBc(char* x, int m, int* bmBc, int size);
void Suffixes(char* x, int m, int* suff);
void PreBmGs(char* x, int m, int* bmGs, int* suff);
void AG(const TVector<char>& pat);
int Max(int a, int b);

int main()
{
	int ch;
	TVector<char> pattern;
	
	VectorInit<char>(pattern);

	while ((ch = getchar()) != '\n')
	{
		ch = ToLower(ch);

		VectorPushBack<char>(pattern, ch);
	}

	VectorPushBack<char>(pattern, '\0');

	AG(pattern);
	
	VectorDestroy<char>(pattern);
	
	return 0;
}

int ToLower(int ch)
{
	return ch >= 'A' && ch <= 'Z' ? ch + 'a' - 'A' : ch;
}

void PreBmBc(char* x, int m, int* bmBc, int size)
{
	for (int i = 0; i < size; ++i)
	{
		bmBc[i] = m;
	}

	for (int i = 0; i < m - 1; ++i)
	{
		int ind = static_cast<int>(x[i]);

		bmBc[ind] = m - i - 1;
	}
}

void Suffixes(char* x, int m, int* suff)
{
	int f = 0;
	int g;
	int i;

	suff[m - 1] = m;
	g = m - 1;
	
	for (i = m - 2; i >= 0; --i)
	{
		if (i > g && suff[i + m - 1 - f] < i - g)
		{
			suff[i] = suff[i + m - 1 - f];
		}
		else
		{
			if (i < g)
			{
				g = i;
			}

			f = i;
		
			while (g >= 0 && x[g] == x[g + m - 1 - f])
			{
				--g;
			}

			suff[i] = f - g;
		}
	}
}

void PreBmGs(char* x, int m, int* bmGs, int* suff)
{
	int i, j;

	Suffixes(x, m, suff);

	for (i = 0; i < m; ++i)
	{
		bmGs[i] = m;
	}

	j = 0;

	for (i = m - 1; i >= -1; --i)
	{
		if (i == -1 || suff[i] == i + 1)
		{
			for (; j < m - 1 - i; ++j)
			{
				if (bmGs[j] == m)
				{
					bmGs[j] = m - 1 - i;
				}
			}
		}
	}

	for (i = 0; i < m - 1; ++i)
	{
		bmGs[m - 1 - suff[i]] = m - 1 - i;
	}
}

void AG(const TVector<char>& pat)
{
	const int A_SIZE = 128;
	int i;
	int j;
	int k;
	int s;
	int shift;
	int ch;
	int m = pat.size - 1;
	int* bmBc = NULL;
	int* bmGs = NULL;
	int* skip = NULL;
	int* suff = NULL;
	TQueue<char> text;
	TQueue<TPos> pos;
	TPos tmpPos;

	try
	{
		bmBc = new int[A_SIZE];
		bmGs = new int[m];
		skip = new int[m];
		suff = new int[m];
	}
	catch (const std::bad_alloc& e)
	{
		printf("ERROR: No memory\n");

		std::exit(0);
	}

	tmpPos.row = 1;
	tmpPos.col = 1;
	
	QueueInit<char>(text, m);
	QueueInit<TPos>(pos, m);

	PreBmGs(pat.begin, m, bmGs, suff);
	PreBmBc(pat.begin, m, bmBc, A_SIZE);
	memset(skip, 0, m * sizeof(int));

	j = 0;

	bool isWordFound = false;

	while (true)
	{
		while (text.size < m && (ch = getchar()) != EOF)
		{
			if (ch == ' ' || ch == '\t')
			{
				if (isWordFound)
				{
					QueuePush<char>(text, ' ');
					
					isWordFound = false;
					++tmpPos.col;
				}
			}
			else if (ch == '\n')
			{
				if (isWordFound)
				{
					QueuePush<char>(text, ' ');

					isWordFound = false;
				}

				++tmpPos.row;
				tmpPos.col = 1;
			}
			else
			{
				if (!isWordFound)
				{
					QueuePush<TPos>(pos, tmpPos);
				}

				QueuePush<char>(text, ToLower(ch));

				isWordFound = true;
			}
		}

		if (text.size < m)
		{
			break;
		}

		i = m - 1;

		while (i >= 0)
		{
			k = skip[i];
			s = suff[i];

			if (k > 0)
			{
				if (k > s)
				{
					if (i + 1 == s)
					{
						i = -1;
					}
					else
					{
						i -= s;
					}

					break;
				}
				else
				{
					i -= k;

					if (k < s)
					{
						break;
					}
				}
			}
			else
			{
				if (pat.begin[i] == text.begin[(i + j) % m])
				{
					--i;
				}
				else
				{
					break;
				}
			}
		}

		if (i < 0)
		{
			TPos wp = pos.begin[pos.offset];

			printf("%d, %d\n", wp.row, wp.col);
			
			skip[m - 1] = m;
			shift = bmGs[0];
		}
		else
		{
			int ind = static_cast<int>(text.begin[(i + j) % m]);

			skip[m - 1] = m - 1 - i;
			shift = Max(bmGs[i], bmBc[ind] - m + 1 + i);
		}
		
		int z = j;

		j += shift;
		
		for (; z < j; ++z)
		{
			QueuePop<char>(text);

			if (text.begin[z % m] == ' ')
			{
				QueuePop<TPos>(pos);
			}
		}

		memcpy(skip, skip + shift, (m - shift) * sizeof(int));
		memset(skip + m - shift, 0, shift * sizeof(int));
	}

	QueueDestroy<char>(text);
	QueueDestroy<TPos>(pos);

	delete[] bmBc;
	delete[] bmGs;
	delete[] skip;
	delete[] suff;
}

int Max(int a, int b)
{
	return a > b ? a : b;
}
