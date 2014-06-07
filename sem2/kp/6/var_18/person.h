#ifndef PERSON_H
#define PERSON_H

typedef enum _kSex
{
	MALE = 0,	// Мужской
	FEMALE		// Женский
} kSex;

typedef struct _Person
{
	char fam[17];	// Фамилия
	char ini[9];	// Инициалы
	kSex sex;		// Пол
	int group;		// Группа
	int informat;	// Информатика
	int linal;		// Лин. алгебра
	int diskr;		// Дискр. математика
} Person;

#endif
