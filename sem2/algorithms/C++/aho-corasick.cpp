#include <iostream>
#include <string>
#include <vector>
#include <queue>
#include <map>

typedef std::map<char, int> Node;

enum kData
{
	LINK = 0,
	LEAF = 10
};

struct VertexInfo
{
	int v;
	int p;
	char ch;
};

class AhoCorasick
{
public:
	AhoCorasick();

	void addString(const std::string &s);
	void buildLinks();
	int go(int v, char c);
	void check(int v, int i);
	void findAll(const std::string &s);

private:
	std::vector<Node> _t;
};

int main()
{
	AhoCorasick ac;

	ac.addString("string");
	ac.addString("search");

	ac.buildLinks();

	ac.findAll("Testing string for searching");

	return 0;
}

AhoCorasick::AhoCorasick()
{
	_t.push_back(Node());
}

void AhoCorasick::addString(const std::string &s)
{
	int v = 0;

	for (size_t i = 0; i < s.length(); i++)
	{
		if (_t[v].count(s[i]) == 0)
		{
			_t.push_back(Node());

			_t[v][s[i]] = _t.size() - 1;
		}

		v = _t[v][s[i]];
	}

	_t[v][LEAF] = s.length();
}

void AhoCorasick::buildLinks()
{
	std::queue<VertexInfo> q;

	VertexInfo vi;

	vi.v = 0;
	vi.p = 0;
	vi.ch = '$';

	q.push(vi);
	
	while (!q.empty())
	{
		vi = q.front();

		int cur = vi.v;
		int p = vi.p;
		char ch = vi.ch;

		q.pop();

		if (cur != 0 && p != 0)
		{
			int link = _t[p].count(LINK) > 0 ? _t[p][LINK] : 0;

			if (_t[link].count(ch) > 0)
				_t[cur][LINK] = _t[link][ch];
		}

		for (Node::iterator it = _t[cur].begin(); it != _t[cur].end(); it++)
		{
			if (it->first == LEAF || it->first == LINK)
				continue;

			vi.p = cur;
			vi.v = it->second;
			vi.ch = it->first;

			q.push(vi);
		}
	}
}

int AhoCorasick::go(int v, char c)
{
	if (_t[v].count(c) > 0)
		return _t[v][c];

	if (v == 0)
		return 0;
	
	return go(_t[v].count(LINK) > 0 ? _t[v][LINK] : 0, c);
}

void AhoCorasick::check(int v, int i)
{
	for (size_t u = v; u != 0; u = _t[u].count(LINK) > 0 ? _t[u][LINK] : 0)
		if (_t[u].count(LEAF) > 0)
			std::cout << "Matched: " << (i - _t[u][LEAF] + 1) << std::endl;
		else
			break;
}

void AhoCorasick::findAll(const std::string &s)
{
	int v = 0;

	for (size_t i = 0; i < s.length(); i++)
	{
		v = go(v, s[i]);

		check(v, i);
	}
}
