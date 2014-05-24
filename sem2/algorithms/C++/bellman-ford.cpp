#include <cstdio>
#include <vector>

struct Edge
{
	int from;
	int to;
	int cost;	
};

int main()
{
	const int INF = 2000000000;
	int n, m;
	std::vector<Edge> g;
	std::vector<int> dist;

	scanf("%d %d", &n, &m);

	g.resize(m);
	dist.resize(n, INF);

	for (int i = 0; i < m; i++)
	{
		scanf("%d %d %d", &g[i].from, &g[i].to, &g[i].cost);

		g[i].from--;
		g[i].to--;
	}

	int from;

	scanf("%d", &from);

	from--;

	dist[from] = 0;

	for (int i = 0; i < n - 1; i++)
		for (int j = 0; j < m; j++)
			if (dist[g[j].from] < INF)
				dist[g[j].to] = std::min(dist[g[j].to], dist[g[j].from] + g[j].cost);

	for (int i = 0; i < n; i++)
		if (dist[i] == INF)
			printf("До вершины %d: недостижима\n", i + 1);
		else
			printf("До вершины %d: %d\n", i + 1, dist[i]);

	return 0;
}
