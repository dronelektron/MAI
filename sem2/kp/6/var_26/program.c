/*
26. *Найти абитуриентов-немедалистов, суммарный балл которых выше среднего.
*/

#include <stdio.h>
#include <string.h>
#include <math.h>
#include "person.h"

int main(int argc, char *argv[])
{
	int sum = 0, cnt = 0, avg = 0;
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
		printf("+----------------+----------+---------+-------+--------+------------+--------+----------+\n");
		printf("|    Фамилия     | Инициалы |   Пол   | Школа | Медаль | Математика | Физика | Рус. яз. |\n");
		printf("+----------------+----------+---------+-------+--------+------------+--------+----------+\n");

		while (fread(&p, sizeof(p), 1, file) == 1)
		{
			printf("|%16s|%10s|%9s|%7d|%8s|%12d|%8d|%10d|\n",
				p.fam,
				p.ini,
				p.pol == 'm' ? "Male" : "Female",
				p.nomer,
				p.gold == 'y' ? "Yes" : "No",
				p.matem,
				p.fizika,
				p.rus
			);
		
			printf("+----------------+----------+---------+-------+--------+------------+--------+----------+\n");
		}
	}
	else if (!strcmp(argv[2], "-p"))
	{
		while (fread(&p, sizeof(p), 1, file) == 1)
		{	
			sum += p.matem + p.fizika + p.rus;
			cnt++;
		}

		avg = (int)round((double)sum / cnt);

		printf("Абитуриенты без медалей и у которых суммарный балл выше среднего:\n");

		fseek(file, 0, SEEK_SET);

		while (fread(&p, sizeof(p), 1, file) == 1)
		{	
			sum = p.matem + p.fizika + p.rus;

			if (p.gold == 'n' && sum > avg)
				printf("%s\n", p.fam);
		}
	}
	
	fclose(file);

	return 0;
}
