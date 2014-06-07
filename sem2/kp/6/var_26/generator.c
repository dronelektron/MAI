#include <stdio.h>
#include "person.h"

int readPerson(Person *p);

int main(int argc, char *argv[])
{
	Person p;
	FILE *file = NULL;

	if (argc != 2)
	{
		printf("Usage: %s filename\n", argv[0]);

		return 1;
	}

	file = fopen(argv[1], "wb");

	if (file == NULL)
	{
		printf("Произошла ошибка при открытии файла\n");

		return 1;
	}

	while (readPerson(&p))
		fwrite(&p, sizeof(p), 1, file);

	fclose(file);

	return 0;
}

int readPerson(Person *p)
{
	int ret = scanf("%s %s %c %d %c %d %d %d",
		p->fam,
		p->ini,
		&p->pol,
		&p->nomer,
		&p->gold,
		&p->matem,
		&p->fizika,
		&p->rus
	);
	
	return (ret == 8);
}
