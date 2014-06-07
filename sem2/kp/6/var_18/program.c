/*
18. *Выяснить в какой группе разность между
максимальным и минимальным средним баллом студентов максимальна.
*/

#include <stdio.h>
#include <string.h>
#include "person.h"
#include "bst.h"

int main(int argc, char *argv[])
{
	int sum, diff, max = 0;
	Person p;
	FILE *file = NULL;
	Node *bstRoot = NULL;
	Node *tmp = NULL;

	if (argc < 3)
	{
		printf("Использование: %s файл флаг\nФлаги:\n-f - печать базы данных\n-p - выполнить задание\n", argv[0]);

		return 1;
	}

	file = fopen(argv[1], "rb");

	if (file == NULL)
	{
		printf("Произошла ошибка при открытии файла\n");

		return 1;
	}

	if (!strcmp(argv[2], "-f"))
	{
		printf("+----------------+----------+---------+--------+-------------+--------------+---------------+\n");
		printf("|    Фамилия     | Инициалы |   Пол   | Группа | Информатика | Лин. алгебра | Дискр. матем. |\n");
		printf("+----------------+----------+---------+--------+-------------+--------------+---------------+\n");

		while (fread(&p, sizeof(p), 1, file) == 1)
		{
			printf("|%16s|%10s|%9s|%8d|%13d|%14d|%15d|\n",
				p.fam,
				p.ini,
				p.sex == MALE ? "Male" : "Female",
				p.group,
				p.informat,
				p.linal,
				p.diskr
			);
		
			printf("+----------------+----------+---------+--------+-------------+--------------+---------------+\n");
		}
	}
	else if (!strcmp(argv[2], "-p"))
	{
		while (fread(&p, sizeof(p), 1, file) == 1)
		{	
			tmp = bstInsert(&bstRoot, p.group);

			sum = p.informat + p.linal + p.diskr;

			if (sum > tmp->avgMax) tmp->avgMax = sum;
			if (sum < tmp->avgMin) tmp->avgMin = sum;

			diff = tmp->avgMax - tmp->avgMin;

			if (diff > max) max = diff;
		}

		printf("Группы:\n");
		printMaxGroups(bstRoot, max);
	}

	bstDelete(&bstRoot);
	fclose(file);

	return 0;
}
