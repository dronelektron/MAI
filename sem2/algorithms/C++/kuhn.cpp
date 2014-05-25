#include <cstdio>
#include <vector>

bool kuhn(std::vector<std::vector<int> > &g, std::vector<int> &mt, std::vector<bool> &used, int v);

int main()
{
	int n, m, k;
	std::vector<std::vector<int> > g;
	std::vector<int> mt;
	std::vector<bool> used;
  
	scanf("%d %d %d\n", &n, &m, &k);

	g.resize(n);
	mt.resize(m, -1);
	used.resize(n, false);

	for (int i = 0; i < k; i++)
	{
		int a, b;

		scanf("%d %d", &a, &b);

		a--;
		b--;

		g[a].push_back(b);
	}

	for (int v = 0; v < n; v++)
	{
		used.assign(n, false);
		
		kuhn(g, mt, used, v);
	}

	printf("Паросочетания:\n");

	for (int i = 0; i < m; i++)
		if (mt[i] != -1)
			printf("%d %d\n", mt[i] + 1, i + 1);

	return 0;
}

bool kuhn(std::vector<std::vector<int> > &g, std::vector<int> &mt, std::vector<bool> &used, int v)
{
	if (used[v])
		return false;

	used[v] = true;

	for (size_t i = 0; i < g[v].size(); i++)
	{
		int to = g[v][i];

		if (mt[to] == -1 || kuhn(g, mt, used, mt[to]))
		{
			mt[to] = v;

			return true;
		}
	}

	return false;
}
