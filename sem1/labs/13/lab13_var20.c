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
	int ch, letter, letterFound = 0, isWordFound = 0, isInput = 0, num = 0, numFirst = 0;
	unsigned int soglSet = 0, letterSet = 0, glasSet = 0;

	for (int i = 0; i < 26; i++) addToSet(&letterSet, 'a' + i);

	addToSet(&glasSet, 'a');
	addToSet(&glasSet, 'o');
	addToSet(&glasSet, 'e');
	addToSet(&glasSet, 'i');
	addToSet(&glasSet, 'u');
	addToSet(&glasSet, 'y');
	
	soglSet = setSubtract(letterSet, glasSet);

	while ((ch = getchar()) != EOF)
	{
		ch = toLower(ch);

		if (!isSpace(ch))
		{
			if (!isInput)
			{
				num++;
				isInput = 1;
			}

			if (isLetter(ch) && isKeyExists(soglSet, ch))
			{
				if (!letterFound)
				{
					letter = ch;
					letterFound = 1;
					isWordFound = 1;
				}
				else if (ch != letter) isWordFound = 0;
			}
		}
		else
		{
			if (isWordFound && !numFirst) numFirst = num;

			isInput = 0;
			letterFound = 0;
		}
	}

	if (numFirst > 0) printf("Слово #%d - содержит 1 или более одинаковых согласных\n", numFirst);
	else printf("Не найдено слова с 1-й или более одинаковой согласной\n");

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
