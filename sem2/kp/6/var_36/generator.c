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
		printf("Error. Can't open a file\n");

		return 1;
	}

	while (readPerson(&p))
		fwrite(&p, sizeof(p), 1, file);

	fclose(file);

	return 0;
}

int readPerson(Person *p)
{
	int ret = scanf("%s %d %d %s %s %d %d",
		p->fam,
		&p->items,
		&p->weight,
		p->dest,
		p->start,
		&p->trans,
		&p->children
	);
	
	return (ret == 7);
}
