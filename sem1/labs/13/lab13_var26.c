/*
Лабораторная работа 13
Студента: 
Группа: 80-107Б
*/

#include <stdio.h>

void addToSet(unsigned int *s, int c);
int setSubtract(unsigned int s, unsigned int a);
int isKeyExists(unsigned int s, int c);

int toLower(int c);
int isSpace(int c);
int isLetter(int c);
int charBit(int c);

int main(void)
{
	int ch, letterFound = 0, isWordFound = 0, isSkip = 0, isInput = 0, num = 0, numFirst = 0;
	unsigned int soglSet = 0, glasSet = 0;

	addToSet(&glasSet, 'a');
	addToSet(&glasSet, 'o');
	addToSet(&glasSet, 'i');
	addToSet(&glasSet, 'u');
	addToSet(&glasSet, 'y');
	addToSet(&glasSet, 'e');

	addToSet(&soglSet, 's');
	addToSet(&soglSet, 'z');

	while ((ch = getchar()) != EOF)
	{
		ch = toLower(ch);

		if (!isSpace(ch))
		{
			if (isSkip) continue;

			if (!isInput)
			{
				num++;
				isInput = 1;
			}

			if (isLetter(ch))
			{
				if (!letterFound)
				{
					if (!isKeyExists(glasSet, ch))
					{
						if (!isKeyExists(soglSet, ch)) isSkip = 1;
						else
						{
							letterFound = 1;
							isWordFound = 1;
						}
					}
				}
				else if (!isKeyExists(glasSet, ch) && !isKeyExists(soglSet, ch)) isWordFound = 0;
			}
		}
		else
		{
			if (isWordFound && !numFirst) numFirst = num;

			isInput = 0;
			letterFound = 0;
			isSkip = 0;
		}
	}

	if (numFirst > 0) printf("Слово #%d - содержит только свистящие согласные\n", numFirst);
	else printf("Не найдено слова только со свистящими согласными\n");

	return 0;
}

void addToSet(unsigned int *s, int c)
{
	*s |= charBit(c);
}

int setSubtract(unsigned int s, unsigned int a)
{
	return (s & ~a);
}

int isKeyExists(unsigned int s, int c)
{
	return ((s & charBit(c)) != 0);
}

int toLower(int c)
{
	return (c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c);
}

int isSpace(int c)
{
	return (c == ' ' || c == ',' || c == '\n' || c == '\t');
}

int isLetter(int c)
{
	return (c >= 'a' && c <= 'z');
}

int charBit(int c)
{
	return (1 << (c - 'a'));
}
