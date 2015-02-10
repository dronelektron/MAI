#include <cstdio>
#include <cstring>
#include <vector>
#include <queue>

class AhoCorasick
{
public:
	struct Node
	{
		bool flag;
		char ch;
		int par;
		int suff;
		int gsuff;
		int nodes[26];
	};

	AhoCorasick();

	void addString(const char* str);
	void buildLinks();
	void find(const char* str);

private:
	std::vector<Node> _trie;

	Node _createNode(int p, char ch);
	void _check(int v, int i);
	int _go(int v, char ch);
};

int main()
{
	AhoCorasick ac;
	
	ac.addString("d");
	ac.addString("cd");
	ac.addString("bcd");
	ac.addString("abcd");
	
	ac.buildLinks();
	ac.find("abcdcdcdxabxacd");
	
	return 0;	
}

AhoCorasick::AhoCorasick()
{
	_trie.push_back(_createNode(-1, '$'));
}

void AhoCorasick::addString(const char* str)
{
	int n = strlen(str);
	int num = 0;

	for (int i = 0; i < n; ++i)
	{
		int ind = str[i] - 'a';

		if (_trie[num].nodes[ind] == -1)
		{
			_trie.push_back(_createNode(num, str[i]));
			_trie[num].nodes[ind] = _trie.size() - 1;
		}

		num = _trie[num].nodes[ind];
	}

	_trie[num].flag = true;
}

void AhoCorasick::buildLinks()
{
	std::queue<int> q;

	for (int i = 0; i < 26; ++i)
		if (_trie[0].nodes[i] != -1)
			q.push(_trie[0].nodes[i]);

	while (!q.empty())
	{
		int cur = q.front();
		int ind = _trie[cur].ch - 'a';
		int p = _trie[cur].par;
		int suff = _trie[p].suff;
		
		q.pop();
		
		for (int i = 0; i < 26; ++i)
			if (_trie[cur].nodes[i] != -1)
				q.push(_trie[cur].nodes[i]);

		_trie[cur].suff = 0;

		while (suff != -1)
		{
			if (_trie[suff].nodes[ind] != -1)
			{
				_trie[cur].suff = _trie[suff].nodes[ind];

				break;
			}

			suff = _trie[suff].suff;
		}

		int gs = _trie[cur].suff;

		while (gs != 0)
		{
			if (_trie[gs].flag)
			{
				_trie[cur].gsuff = gs;

				break;
			}

			gs = _trie[gs].suff;
		}
	}
}

void AhoCorasick::find(const char* str)
{
	int n = strlen(str);
	int v = 0;

	for (int i = 0; i < n; ++i)
	{
		v = _go(v, str[i]);

		_check(v, i);
	}
}

AhoCorasick::Node AhoCorasick::_createNode(int p, char ch)
{
	Node node;

	node.ch = ch;
	node.par = p;
	node.suff = -1;
	node.gsuff = -1;
	node.flag = false;

	memset(node.nodes, 255, sizeof(node.nodes));

	return node;
}

void AhoCorasick::_check(int v, int i)
{
	for (int u = v; u != -1; u = _trie[u].gsuff)
		if (_trie[u].flag)
			printf("Matched: %d\n", i);
}

int AhoCorasick::_go(int v, char ch)
{
	int ind = ch - 'a';

	if (_trie[v].nodes[ind] != -1)
		return _trie[v].nodes[ind];

	if (v <= 0)
		return 0;

	return _go(_trie[v].suff, ch);
}
