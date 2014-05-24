#include <cstdio>
#include <vector>

struct Edge
{
	int to;
	int cost;
};

int main()
{
	const int INF = 2000000000;
	int n, m;
	std::vector<std::vector<Edge> > g;
	std::vector<int> dist;
	std::vector<bool> used;

	scanf("%d %d", &n, &m);

	g.resize(n);
	dist.resize(n, INF);
	used.resize(n, false);

	for (int i = 0; i < m; i++)
	{
		int a;
		Edge e;

		scanf("%d %d %d", &a, &e.to, &e.cost);

		a--;
		e.to--;

		g[a].push_back(e);
	}

	int from;

	scanf("%d", &from);

	from--;

	dist[from] = 0;

	for (int i = 0; i < n; i++)
	{
		int cur = -1;

		for (int j = 0; j < n; j++)
			if (!used[j] && (cur == -1 || dist[j] < dist[cur]))
				cur = j;

		if (dist[cur] == INF)
			break;

		used[cur] = true;

		for (int j = 0; j < g[cur].size(); j++)
			dist[g[cur][j].to] = std::min(dist[g[cur][j].to], dist[cur] + g[cur][j].cost);
	}

	for (int i = 0; i < n; i++)
		if (dist[i] == INF)
			printf("До вершины %d: недостижима\n", i + 1);
		else
			printf("До вершины %d: %d\n", i + 1, dist[i]);

	return 0;
}
