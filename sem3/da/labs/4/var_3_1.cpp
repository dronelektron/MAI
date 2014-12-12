#include <exception>
#include <new>
#include <cstdio>
#include <cstring>
#include <cstdlib>

static const int ASIZE = 128;

struct TPos
{
	int row;
	int col;
};

template<class T>
struct TVector
{
	T* begin;
	int cap;
	int size;
};

template<class T>
void VectorInit(TVector<T>& v)
{
	v.begin = new T[1];
	v.cap = 1;
	v.size = 0;
}

template<class T>
void VectorPushBack(TVector<T>& v, const T& val)
{
	if (v.size == v.cap)
	{
		v.cap *= 2;
		T* v2 = new T[v.cap];

		for (int i = 0; i < v.size; ++i)
		{
			v2[i] = v.begin[i];
		}

		delete v.begin;

		v.begin = v2;
	}

	v.begin[v.size++] = val;
}

template<class T>
void VectorDestroy(TVector<T>& v)
{
	delete[] v.begin;
}

int ToLower(int ch);
void PreBmBc(char* x, int m, int* bmBc);
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

void PreBmBc(char* x, int m, int* bmBc)
{
	for (int i = 0; i < ASIZE; ++i)
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
	const int Q_MAX_SIZE = (pat.size - 1) * 2;
	int i;
	int j;
	int k;
	int s;
	int shift;
	int ch;
	int qSize = 0;
	int qSize2 = 0;
	int qInd2 = 0;
	int m = pat.size - 1;
	int* bmBc = new int[ASIZE];
	int* bmGs = new int[m];
	int* skip = new int[m];
	int* suff = new int[m];
	char* y = new char[Q_MAX_SIZE];
	char* x = pat.begin;
	TPos* q = new TPos[Q_MAX_SIZE];
	TPos tmpPos;

	tmpPos.row = 1;
	tmpPos.col = 1;
	
	qInd2 = 0;
	qSize2 = 0;

	PreBmGs(x, m, bmGs, suff);
	PreBmBc(x, m, bmBc);
	memset(skip, 0, m * sizeof(int));

	j = 0;

	bool isWordFound = false;

	while (true)
	{
		while (qSize < Q_MAX_SIZE && (ch = getchar()) != EOF)
		{
			if (ch == ' ' || ch == '\t')
			{
				if (isWordFound)
				{
					y[(j + qSize++) % Q_MAX_SIZE] = ' ';
					isWordFound = false;

					++tmpPos.col;
				}
			}
			else if (ch == '\n')
			{
				if (isWordFound)
				{
					y[(j + qSize++) % Q_MAX_SIZE] = ' ';
					isWordFound = false;
				}

				++tmpPos.row;
				tmpPos.col = 1;
			}
			else
			{
				if (!isWordFound)
				{
					q[(qInd2 + qSize2++) % Q_MAX_SIZE] = tmpPos;
				}

				y[(j + qSize++) % Q_MAX_SIZE] = ToLower(ch);
				isWordFound = true;
			}
		}

		if (qSize < m)
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
				if (x[i] == y[(i + j) % Q_MAX_SIZE])
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
			printf("%d, %d\n", q[qInd2].row, q[qInd2].col);
			
			skip[m - 1] = m;
			shift = bmGs[0];
		}
		else
		{
			int ind = static_cast<int>(y[(i + j) % Q_MAX_SIZE]);

			skip[m - 1] = m - 1 - i;
			shift = Max(bmGs[i], bmBc[ind] - m + 1 + i);
		}
		
		int z = j;

		j += shift;
		qSize -= shift;
		
		for (; z < j; ++z)
		{
			if (y[z % Q_MAX_SIZE] == ' ')
			{
				qInd2 = (qInd2 + 1) % Q_MAX_SIZE;
				--qSize2;
			}
		}

		memcpy(skip, skip + shift, (m - shift) * sizeof(int));
		memset(skip + m - shift, 0, shift * sizeof(int));
	}

	delete[] bmBc;
	delete[] bmGs;
	delete[] skip;
	delete[] suff;
	delete[] y;
	delete[] q;
}

int Max(int a, int b)
{
	return a > b ? a : b;
}
