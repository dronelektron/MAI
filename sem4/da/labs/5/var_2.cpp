#include <cstdio>
#include "suffix_array.h"

struct TStr
{
	char* buffer;
	size_t len;
	size_t cap;
};

void StrInit(TStr& str);
void StrDestroy(TStr& str);
void StrPush(TStr& str, char ch);

int main()
{
	int patCnt = 1;
	int ch;
	TStr str;

	StrInit(str);
	
	while ((ch = getchar()) != '\n')
	{
		StrPush(str, static_cast<char>(ch));
	}

	StrPush(str, '$');
	StrPush(str, '\0');

	TST st(str.buffer);
	TSA sa(st);
	
	while (true)
	{
		str.len = 0;
		ch = getchar();

		if (ch == EOF)
		{
			break;
		}

		StrPush(str, static_cast<char>(ch));

		while ((ch = getchar()) != '\n')
		{
			StrPush(str, static_cast<char>(ch));
		}

		StrPush(str, '\0');
		
		printf("%d: ", patCnt++);

		sa.Find(str.buffer);
	}

	StrDestroy(str);
	
	/*
	NDS::TMap<char, int> m;
	NDS::TMap<char, int>::TData data;

	data.key = 'z';
	data.val = 573;

	m.Insert(data);
	
	data.key = 'b';
	data.val = 3683496;

	m.Insert(data);

	data.key = 'g';
	data.val = 17236;

	m.Insert(data);

	data.key = 'a';
	data.val = 123;

	m.Insert(data);

	data.key = 'm';
	data.val = 6468;

	m.Insert(data);

	data.key = 'x';
	data.val = 566;

	m.Insert(data);

	data.key = 's';
	data.val = 887;

	m.Insert(data);

	for (NDS::TMap<char, int>::TIterator it = m.Begin(); it != m.End(); ++it)
	{
		printf("%c - %d\n", it->key, it->val);
	}
	
	NDS::TMap<char, int>::TNode* it = m.Find('n');

	if (it != NULL)
	{
		printf("Find: %c - %d\n", it->data.key, it->data.val);
	}
	*/
	return 0;
}

void StrInit(TStr& str)
{
	str.buffer = new char[1];
	str.len = 0;
	str.cap = 1;
}

void StrDestroy(TStr& str)
{
	delete[] str.buffer;

	str.len = 0;
	str.cap = 0;
}

void StrPush(TStr& str, char ch)
{
	if (str.len == str.cap)
	{
		str.cap *= 2;

		char* str2 = new char[str.cap];

		for (size_t i = 0; i < str.len; ++i)
		{
			str2[i] = str.buffer[i];
		}

		delete[] str.buffer;

		str.buffer = str2;
	}

	str.buffer[str.len++] = ch;
}
