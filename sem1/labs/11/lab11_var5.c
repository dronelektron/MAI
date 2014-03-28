/*
Лабораторная работа 11
Студента: Барковского А.А.
Группа: 80-107Б
*/

#include <stdio.h>
#include <limits.h>

int isLetter(int ch);
int isSpace(int ch);
int toLower(int ch);

int main(void)
{
	int ch, lastCh = -1, cnt = 0, isInput = 0, f = 0;

	while ((ch = getchar()) != EOF)
	{
		if (!isSpace(ch))
		{
			if (f) continue;

			if (isLetter(ch))
			{
				isInput = 1;

				ch = toLower(ch);

				if (lastCh != -1 && ch < lastCh) f = 1;

				lastCh = ch;
			}
		}
		else
		{
			if (isInput && !f) cnt++;

			isInput = 0;
			lastCh = -1;
			f = 0;
		}
	}

	printf("Количество слов с лексикографически возрастающими буквами: %d\n", cnt);

	return 0;
}

int isLetter(int ch)
{
	return ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z'));
}

int isSpace(int ch)
{
	return (ch == ' ' || ch == ',' || ch == '\n' || ch == '\t');
}

int toLower(int ch)
{
	return (ch >= 'A' && ch <= 'Z' ? ch + 'a' - 'A' : ch);
}
