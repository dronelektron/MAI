/*
Лабораторная работа 13
Студента: 
Группа: 
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
	int i, ch, flag = 0, letterFound = 0, isInput = 0, num = 0, numFirst = 0;
	unsigned int letterSet = 0, glasSet = 0, tmpSet = 0;

	for (i = 0; i < 26; i++) addToSet(&letterSet, 'a' + i);

	addToSet(&glasSet, 'a');
	addToSet(&glasSet, 'o');
	addToSet(&glasSet, 'e');
	addToSet(&glasSet, 'i');
	addToSet(&glasSet, 'u');
	addToSet(&glasSet, 'y');

	letterSet = setSubtract(letterSet, glasSet);
	tmpSet = letterSet;

	while ((ch = getchar()) != EOF)
	{
		ch = toLower(ch);

		if (isLetter(ch))
		{
			isInput = 1;

			if (isKeyExists(glasSet, ch)) continue;

			if (isKeyExists(tmpSet, ch)) removeFromSet(&tmpSet, ch);
			else if (!letterFound)
			{
				letterFound = 1;
				numFirst = num + 1;
			}
		}
		else if (isSpace(ch))
		{
			if (isInput) num++;	

			isInput = 0;
			tmpSet = letterSet;
		}
		else isInput = 1;
	}

	if (letterFound) printf("Слово #%d - содержит одинаковые согласные\n", numFirst);
	else printf("Не найдено ни одного слова с повторяющимися согласными\n");

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
