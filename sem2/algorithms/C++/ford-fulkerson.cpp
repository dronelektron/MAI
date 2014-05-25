#include <cstdio>
#include <vector>
#include <queue>

int hasPath(const std::vector<std::vector<int> > &c, const std::vector<std::vector<int> > &f, std::vector<int> &p, int from, int to);

int main()
{
	int n, m, from, to;
	std::vector<std::vector<int> > c;
	std::vector<std::vector<int> > f;
	std::vector<int> p;

	scanf("%d %d", &n, &m);
	scanf("%d %d", &from, &to);

	from--;
	to--;

	c.resize(n, std::vector<int>(n, 0));
	f.resize(n, std::vector<int>(n, 0));
	p.resize(n, -1);

	for (int i = 0; i < m; i++)
	{
		int start, end, cost;

		scanf("%d %d %d", &start, &end, &cost);

		start--;
		end--;

		c[start][end] = cost;
	}

	int path;

	while ((path = hasPath(c, f, p, from, to)) > 0)
	{
		int cur = to;
		int prev = p[to];

		while (prev != -1)
		{
			f[prev][cur] += path;
			f[cur][prev] = -f[prev][cur];
			cur = prev;
			prev = p[cur];
		}

		p.assign(n, -1);
	}

	int sum = 0;

	for (int i = 0; i < n; i++)
		sum += f[from][i];

	printf("%d\n", sum);

	return 0;
}

int hasPath(const std::vector<std::vector<int> > &c, const std::vector<std::vector<int> > &f, std::vector<int> &p, int from, int to)
{
	const int INF = 2000000000;
	int maxEdge = INF;
	std::queue<int> q;
	std::vector<bool> used(c.size(), false);
	
	q.push(from);
	used[from] = true;

	while (!q.empty())
	{
		int cur = q.front();

		q.pop();	

		for (int i = 0; i < c[cur].size(); i++)
		{
			if (!used[i] && c[cur][i] && c[cur][i] - f[cur][i] > 0)
			{
				q.push(i);

				used[i] = true;
				maxEdge = std::min(maxEdge, c[cur][i] - f[cur][i]);
				p[i] = cur;
			}
		}
	}

	if (maxEdge == INF || p[to] == -1)
		maxEdge = 0;

	return maxEdge;
}
