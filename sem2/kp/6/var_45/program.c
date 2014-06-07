/*
45. *Найти среднее число учениц в классах школы.
*/

#include <stdio.h>
#include <string.h>
#include <math.h>
#include "person.h"
#include "bst.h"

void bstCalc(Node **node, int *sum, int *cnt);

int main(int argc, char *argv[])
{
	int sum = 0, cnt = 0;
	Person p;
	Key key;
	FILE *file = NULL;
	Node *root = NULL;
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
		printf("+----------------+----------+-------+-------+-------+---------+----------------+----------------+\n");
		printf("|    Фамилия     | Инициалы |  Пол  | Класс | Буква |   ВУЗ   |     Работа     |      Полк      |\n");
		printf("+----------------+----------+-------+-------+-------+---------+----------------+----------------+\n");

		while (fread(&p, sizeof(p), 1, file) == 1)
		{
			printf("|%16s|%10s|%7s|%7d|%7c|%9s|%16s|%16s|\n",
				p.fam,
				p.ini,
				p.pol == 'm' ? "Male" : "Female",
				p.klass,
				p.bukva,
				p.vuz,
				p.work,
				p.polk
			);
			
			printf("+----------------+----------+-------+-------+-------+---------+----------------+----------------+\n");
		}
	}
	else if (!strcmp(argv[2], "-p"))
	{
		while (fread(&p, sizeof(p), 1, file) == 1)
		{
			if (p.pol != 'f')
				continue;

			key.klass = p.klass;
			key.bukva = p.bukva;
			tmp = bstInsert(&root, key);
			tmp->cnt++;
		}

		bstCalc(&root, &sum, &cnt);

		printf("Среднее число учениц: %d\n", (int)round((double)sum / cnt));
	}

	bstDestroy(&root);
	fclose(file);

	return 0;
}

void bstCalc(Node **node, int *sum, int *cnt)
{
	if (*node == NULL)
		return;

	bstCalc(&(*node)->right, sum, cnt);
	bstCalc(&(*node)->left, sum, cnt);

	(*sum) += (*node)->cnt;
	(*cnt)++;
}
