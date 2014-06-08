/*
16. Найти фамилии лучших студенток курса
(не имеющих отметок ниже четырех и по сумме баллов не уступающих другим студентам своей группы)
*/

#include <stdio.h>
#include <string.h>
#include "person.h"

int main(int argc, char *argv[])
{
	int sum, max = 0;
	Person p;
	FILE *file = NULL;

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
			sum = p.informat + p.linal + p.diskr;

			if (p.sex == FEMALE && sum > max)
				max = sum;
		}

		printf("Лучшие студентки:\n");

		if (max < 12)
			printf("Не найдено\n");
		else
		{
			fseek(file, 0, SEEK_SET);

			while (fread(&p, sizeof(p), 1, file) == 1)
			{	
				sum = p.informat + p.linal + p.diskr;

				if (p.sex == FEMALE && sum == max)
					printf("%s\n", p.fam);
			}
		}
	}

	fclose(file);

	return 0;
}
