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

		if (ch == ' ')
		{
			ch = '{';
		}

		VectorPushBack<char>(pattern, ch);
	}

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
		bmBc[x[i] - 'a'] = m - i - 1;
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
	const int A_SIZE = '{' - 'a' + 1;
	int i;
	int k;
	int s;
	int shift;
	int ch;
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
		bmGs = new int[pat.size];
		skip = new int[pat.size];
		suff = new int[pat.size];
	}
	catch (const std::bad_alloc& e)
	{
		printf("ERROR: No memory\n");

		std::exit(0);
	}

	tmpPos.row = 1;
	tmpPos.col = 1;
	
	QueueInit<char>(text, pat.size);
	QueueInit<TPos>(pos, pat.size);

	PreBmGs(pat.begin, pat.size, bmGs, suff);
	PreBmBc(pat.begin, pat.size, bmBc, A_SIZE);
	memset(skip, 0, pat.size * sizeof(int));

	bool isWordFound = false;

	while (true)
	{
		while (text.size < pat.size && (ch = getchar()) != EOF)
		{
			if (ch == ' ' || ch == '\t')
			{
				if (isWordFound)
				{
					QueuePush<char>(text, '{');
					
					isWordFound = false;
					++tmpPos.col;
				}
			}
			else if (ch == '\n')
			{
				if (isWordFound)
				{
					QueuePush<char>(text, '{');

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

		if (text.size < pat.size)
		{
			break;
		}

		i = pat.size - 1;

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
				if (pat.begin[i] == text.begin[(i + text.offset) % pat.size])
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
			
			skip[pat.size - 1] = pat.size;
			shift = bmGs[0];
		}
		else
		{
			int ind = text.begin[(i + text.offset) % pat.size] - 'a';
			
			skip[pat.size - 1] = pat.size - 1 - i;
			shift = Max(bmGs[i], bmBc[ind] - pat.size + 1 + i);
		}
		
		int offset = text.offset;

		for (size_t z = 0; z < shift; ++z)
		{
			QueuePop<char>(text);

			if (text.begin[(offset + z) % pat.size] == '{')
			{
				QueuePop<TPos>(pos);
			}
		}

		memcpy(skip, skip + shift, (pat.size - shift) * sizeof(int));
		memset(skip + pat.size - shift, 0, shift * sizeof(int));
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
