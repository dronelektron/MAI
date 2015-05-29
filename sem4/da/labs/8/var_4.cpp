#include <exception>
#include <new>
#include <cstdio>
#include <cstdlib>

typedef double TColType;

struct TRow
{
	TColType* cols;
	int cost;
	int num;
};

struct TMat
{
	TRow* rows;
	int m;
	int n;
};

void MatrixCreate(TMat& mat, int m, int n);
void MatrixDestroy(TMat& mat);
bool Gauss(TMat& mat, bool* used);

int main()
{
	int m;
	int n;
	bool* ans = NULL;
	TMat mat;

	scanf("%d %d", &m, &n);

	MatrixCreate(mat, m, n);

	try
	{
		ans = new bool[m];
	}
	catch (const std::bad_alloc& e)
	{
		printf("No memory\n");

		std::exit(0);
	}

	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			scanf("%lf", &mat.rows[i].cols[j]);
		}

		scanf("%d", &mat.rows[i].cost);

		ans[i] = false;
	}

	if (Gauss(mat, ans))
	{
		int cnt = 0;

		for (int i = 0; i < m && cnt < n; ++i)
		{
			if (ans[i])
			{
				if (cnt == n - 1)
				{
					printf("%d", i + 1);
				}
				else
				{
					printf("%d ", i + 1);
				}

				++cnt;
			}
		}

		printf("\n");
	}
	else
	{
		printf("-1\n");
	}

	MatrixDestroy(mat);

	delete[] ans;

	return 0;
}

void MatrixCreate(TMat& mat, int m, int n)
{
	mat.rows = NULL;
	mat.m = m;
	mat.n = n;

	try
	{
		mat.rows = new TRow[m];

		for (int i = 0; i < m; ++i)
		{
			mat.rows[i].cols = new TColType[n];
			mat.rows[i].cost = 0;
			mat.rows[i].num = i;
		}
	}
	catch (const std::bad_alloc& e)
	{
		printf("No memory\n");

		std::exit(0);
	}
}

void MatrixDestroy(TMat& mat)
{
	for (int i = 0; i < mat.m; ++i)
	{
		delete[] mat.rows[i].cols;
	}

	delete[] mat.rows;

	mat.rows = NULL;
	mat.m = 0;
	mat.n = 0;
}

bool Gauss(TMat& mat, bool* used)
{
	for (int col = 0; col < mat.n; ++col)
	{
		int minRow = -1;
		int minCost = 100;
		
		for (int row = col; row < mat.m; ++row)
		{
			if (mat.rows[row].cols[col] != 0.0 && mat.rows[row].cost < minCost)
			{
				minRow = row;
				minCost = mat.rows[row].cost;
			}
		}

		if (minRow == -1)
		{
			return false;
		}

		TRow tmpRow = mat.rows[col];

		mat.rows[col] = mat.rows[minRow];
		mat.rows[minRow] = tmpRow;

		used[mat.rows[col].num] = true;

		for (int row = col + 1; row < mat.m; ++row)
		{
			double c = mat.rows[row].cols[col] / mat.rows[col].cols[col];

			for (int i = col; i < mat.n; ++i)
			{
				mat.rows[row].cols[i] -= mat.rows[col].cols[i] * c;
			}
		}
	}

	return true;
}
