#include <exception>
#include <new>
#include <cstdlib>
#include <cstdio>

int main()
{
	const long long INF = static_cast<long long>(1) << 60;
	int n;
	int m;
	int** path = NULL;
	int* pathRev = NULL;
	long long* dataPrev = NULL;
	long long* dataCur = NULL;

	scanf("%d %d", &n, &m);

	try
	{
		path = new int*[n];
		pathRev = new int[n];
		dataPrev = new long long[m + 2];
		dataCur = new long long[m];
	}
	catch (const std::bad_alloc& e)
	{
		printf("No memory\n");
		std::exit(0);
	}

	for (int i = 0; i < n; ++i)
	{
		path[i] = new int[m];
	}

	dataPrev[0] = INF;
	dataPrev[m + 1] = INF;

	for (int i = 0; i < m; ++i)
	{
		scanf("%lld", &dataPrev[i + 1]);
	}

	for (int i = 1; i < n; ++i)
	{
		for (int j = 0; j < m; ++j)
		{
			scanf("%lld", &dataCur[j]);
		}

		for (int j = 1; j <= m; ++j)
		{
			long long minCost = INF;

			if (dataPrev[j - 1] < minCost)
			{
				minCost = dataPrev[j - 1];
				path[i][j - 1] = j - 1;
			}

			if (dataPrev[j] < minCost)
			{
				minCost = dataPrev[j];
				path[i][j - 1] = j;
			}
			
			if (dataPrev[j + 1] < minCost)
			{
				minCost = dataPrev[j + 1];
				path[i][j - 1] = j + 1;
			}

			dataCur[j - 1] += minCost;
		}

		for (int j = 0; j < m; ++j)
		{
			dataPrev[j + 1] = dataCur[j];
		}
	}

	long long ans = dataCur[0];
	int end = 1;

	for (int i = 1; i < m; ++i)
	{
		if (dataCur[i] < ans)
		{
			ans = dataCur[i];
			end = i + 1;
		}
	}

	for (int i = 0; i < n; ++i)
	{
		pathRev[i] = end;
		end = path[n - i - 1][end - 1];
	}

	printf("%lld\n", ans);
	printf("(%d,%d)", 1, pathRev[n - 1]);
	
	for (int i = 1; i < n; ++i)
	{
		printf(" (%d,%d)", i + 1, pathRev[n - i - 1]);
	}

	printf("\n");

	for (int i = 0; i < n; ++i)
	{
		delete[] path[i];
	}
	
	delete[] dataCur;
	delete[] dataPrev;
	delete[] pathRev;
	delete[] path;

	return 0;
}
