#include <cstdio>
#include <vector>
#include <algorithm>

int main()
{
	int n, m;
	std::vector<std::pair<int, std::pair<int, int> > > g;
	std::vector<int> trees;

	scanf("%d %d", &n, &m);

	g.resize(m);
	trees.resize(n);

	for (int i = 0; i < m; i++)
	{
		scanf("%d %d %d", &g[i].second.first, &g[i].second.second, &g[i].first);

		g[i].second.first--;
		g[i].second.second--;
	}

	int w = 0;

	for (int i = 0; i < n; i++)
		trees[i] = i;

	std::sort(g.begin(), g.end());

	for (int i = 0; i < m; i++)
	{
		int from = g[i].second.first;
		int to = g[i].second.second;
		int cost = g[i].first;

		if (trees[from] != trees[to])
		{
			w += cost;
			
			int unionOld = trees[to];

			for (int k = 0; k < n; k++)
				if (trees[k] == unionOld)
					trees[k] = trees[from];
		}
	}

	printf("Минимальная сумма весов ребр: %d\n", w);

	return 0;
}
