#ifndef PERSON_H
#define PERSON_H

typedef struct _Person
{
	char fam[17];	// Фамилия
	int items;		// Количество вещей
	int weight;		// Общий вес вещей
	char dest[17];	// Пункт назначения
	char start[6];	// Время вылета
	int trans;		// Наличие пересадок
	int children;	// Сведения о детях
} Person;

#endif
