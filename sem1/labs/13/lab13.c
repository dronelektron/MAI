/*
Лабораторная работа 13
Студента: Барковского А.А.
Группа: 80-107Б
*/

#include <stdio.h>

void addToSet(unsigned int *s, int c);
void removeFromSet(unsigned int *s, int c);
int setSubtract(unsigned int s, unsigned int a);
int isEmptySet(unsigned int s);
int isKeyExists(unsigned int s, int c);

int toLower(int c);
int isSpace(int c);
int isLetter(int c);
int charBit(int c);

int main(void)
{
	char letter;
	int ch, flag = 0, letterFound = 0, isInput = 0, num = 0, numFirst = 0;
	unsigned int tmpSet = 0, letterSet = 0, glasSet = 0;

	for (int i = 0; i < 26; i++) addToSet(&letterSet, 'a' + i);

	addToSet(&glasSet, 'a');
	addToSet(&glasSet, 'o');
	addToSet(&glasSet, 'e');
	addToSet(&glasSet, 'i');
	addToSet(&glasSet, 'u');
	addToSet(&glasSet, 'y');
	
	tmpSet = letterSet = setSubtract(letterSet, glasSet);

	while ((ch = getchar()) != EOF)
	{
		ch = toLower(ch);

		if (isLetter(ch))
		{
			isInput = 1;
			letterFound = 1;

			if (isKeyExists(glasSet, ch)) continue;

			if (isKeyExists(tmpSet, ch)) removeFromSet(&tmpSet, ch);
			else flag = 1;
		}
		else if (isSpace(ch))
		{
			if (isInput) num++;	

			if (!flag && isInput && !numFirst && letterFound)
			{
				numFirst = num;
				letterFound = 0;
			}

			flag = 0;
			isInput = 0;
			letterFound = 0;
			tmpSet = letterSet;
		}
		else isInput = 1;
	}

	if (numFirst > 0) printf("Слово #%d - не содержит одинаковых согласных\n", numFirst);

	return 0;
}

void addToSet(unsigned int *s, int c)
{
	*s |= charBit(c);
}

void removeFromSet(unsigned int *s, int c)
{
	*s &= ~charBit(c);
}

int setSubtract(unsigned int s, unsigned int a)
{
	return (s & ~a);
}

int isEmptySet(unsigned int s)
{
	return (s != 0);
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
