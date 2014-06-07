/*
36. *Определить, имеются ли два пассажира,
багаж которых совпадает по числу вещей и
различается по весу не более чем на p кг.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "person.h"

int main(int argc, char *argv[])
{
	int cnt = 0, isFound = 0, mass;
	Person p, p1, p2;
	FILE *file = NULL;
	FILE *file2 = NULL;

	if (argc < 3)
	{
		printf("Usage: %s filename [flag]\nFlags:\n-f - print database\n-p VALUE - set parameter\n", argv[0]);

		return 1;
	}
	
	file = fopen(argv[1], "rb");
	file2 = fopen(argv[1], "rb");

	if (file == NULL)
	{
		printf("Error. Can't open a file\n");

		return 1;
	}

	if (!strcmp(argv[2], "-f"))
	{
		printf("+----------------+------+-----+------------------+--------------+-------------------+--------------------+\n");
		printf("|    Фамилия     | Вещи | Вес | Пункт назначения | Время вылета | Наличие пересадок | Информация о детях |\n");
		printf("+----------------+------+-----+------------------+--------------+-------------------+--------------------+\n");

		while (fread(&p, sizeof(p), 1, file) == 1)
		{
			printf("|%16s|%6d|%5d|%18s|%14s|%19d|%20d|\n",
				p.fam,
				p.items,
				p.weight,
				p.dest,
				p.start,
				p.trans,
				p.children
			);
			
			printf("+----------------+------+-----+------------------+--------------+-------------------+--------------------+\n");		
		}
	}
	else if (!strcmp(argv[2], "-p"))
	{
		if (argc < 4)
		{
			printf("Usage: %s filename -p VALUE\n", argv[0]);

			return 1;
		}

		mass = atoi(argv[3]);

		while (fread(&p1, sizeof(p1), 1, file) == 1)
		{
			while (fread(&p2, sizeof(p2), 1, file2) == 1)
			{
				if (strcmp(p1.fam, p2.fam) && p1.items == p2.items && abs(p1.weight - p2.weight) <= mass)
				{
					printf("Пассажиры: %s и %s\n", p1.fam, p2.fam);

					isFound = 1;

					break;
				}
			}

			if (isFound)
				break;

			fseek(file2, 0, SEEK_SET);
		}

		if (!isFound)
			printf("Пассажиры с одинаковым количеством вещей и разностью весов багажа не более %d кг - не найдены.\n", mass);
	}

	fclose(file);
	fclose(file2);

	return 0;
}
