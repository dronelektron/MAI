/*
Лабораторная работа 13
Студента: 
Группа: 80-107Б
*/

#include <stdio.h>

void addToSet(unsigned int *s, int c);
void removeFromSet(unsigned int *s, int c);
void clearSet(unsigned int *s);
unsigned int setSubtract(unsigned int s, unsigned int a);
unsigned int setCross(unsigned int s1, unsigned int s2);
int isEmptySet(unsigned int s);
int isKeyExists(unsigned int s, int c);

int toLower(int c);
int isSpace(int c);
int isLetter(int c);
int charBit(int c);

int main(void)
{
	char letter;
	int ch, isInput = 0, num = 0, numCurrent = 0;
	unsigned int set1 = 0, set2 = 0, soglSet = 0, glasSet = 0, letterSet = 0;
	unsigned int res = 0;

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
				set1 = set2;

				clearSet(&set2);
			}

			if (isLetter(ch) && isKeyExists(soglSet, ch)) addToSet(&set2, ch);
		}
		else
		{
			if (setCross(set1, set2) && !numCurrent) numCurrent = num;

			isInput = 0;
		}
	}

	if (numCurrent > 1) printf("Слова #%d и #%d - имеют пересекающийся набор согласных\n", numCurrent - 1, numCurrent);
	else printf("Нет соседних слов с пересекающимся набором согласных\n");

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

void clearSet(unsigned int *s)
{
	*s = 0;
}

unsigned int setSubtract(unsigned int s, unsigned int a)
{
	return (s & ~a);
}

unsigned int setCross(unsigned int s1, unsigned int s2)
{
	return (s1 & s2);
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
