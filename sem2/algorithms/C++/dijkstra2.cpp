#include <cstdio>
#include <vector>
#include <set>

int main()
{
	const int INF = 2000000000;
	int n, m;
	std::vector<std::vector<std::pair<int, int> > > g;
	std::vector<int> dist;
	std::vector<bool> used;
	std::set<std::pair<int, int> > s;

	scanf("%d %d", &n, &m);

	g.resize(n);
	dist.resize(n, INF);
	used.resize(n, false);

	for (int i = 0; i < m; i++)
	{
		int a, b, c;

		scanf("%d %d %d", &a, &b, &c);

		a--;
		b--;

		g[a].push_back(std::make_pair<int, int>(b, c));
	}

	int from;

	scanf("%d", &from);

	from--;

	dist[from] = 0;

	s.insert(std::make_pair<int, int>(0, from));

	while (!s.empty())
	{
		int cur = s.begin()->second;

		s.erase(s.begin());

		for (int j = 0; j < g[cur].size(); j++)
		{
			if (dist[cur] + g[cur][j].second < dist[g[cur][j].first])
			{
				s.erase(std::make_pair<int, int>(dist[g[cur][j].first], g[cur][j].first));

				dist[g[cur][j].first] = dist[cur] + g[cur][j].second;

				s.insert(std::make_pair<int, int>(dist[g[cur][j].first], g[cur][j].first));
			}
		}
	}

	for (int i = 0; i < n; i++)
		if (dist[i] == INF)
			printf("До вершины %d: недостижима\n", i + 1);
		else
			printf("До вершины %d: %d\n", i + 1, dist[i]);

	return 0;
}
