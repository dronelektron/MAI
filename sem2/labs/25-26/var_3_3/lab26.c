#include <stdio.h>
#include "sort.h"

void getLine(char *str, const int size);

int main(void)
{
	const int N = 10;
	int action;
	char tmpCh;
	Udt udt;
	Item item;

	udtCreate(&udt, N);

	do
	{
		printf("Меню\n");
		printf("1) Добавить элемент слева\n");
		printf("2) Добавить элемент справа\n");
		printf("3) Удалить элемент слева\n");
		printf("4) Удалить элемент справа\n");
		printf("5) Размер дека\n");
		printf("6) Сортировка\n");
		printf("7) Печать\n");
		printf("8) Выход\n");
		printf("Выберите действие: ");
		scanf("%d", &action);

		switch (action)
		{
			case 1:
			{
				printf("Введите ключ: ");
				scanf("%f", &item._key);
				scanf("%c", &tmpCh);
				printf("Введите Строку: ");
				getLine(item._str, sizeof(item._str));

				if (udtPushFront(&udt, item))
					printf("Элемент с ключом %f и строкой '%s' добавлен успешно\n", item._key, item._str);
				else
					printf("Дек полон\n");
			}
			break;

			case 2:
			{
				printf("Введите ключ: ");
				scanf("%f", &item._key);
				scanf("%c", &tmpCh);
				printf("Введите Строку: ");
				getLine(item._str, sizeof(item._str));

				if (udtPushBack(&udt, item))
					printf("Элемент с ключом %f и строкой '%s' добавлен успешно\n", item._key, item._str);
				else
					printf("Дек полон\n");
			}
			break;

			case 3:
			{
				if (udtSize(&udt) > 0)
				{
					item = udtTopFront(&udt);

					udtPopFront(&udt);

					printf("Элемент с ключом %f и строкой '%s' удален успешно\n", item._key, item._str);
				}
				else
					printf("Дек пуст\n");
			}
			break;

			case 4:
			{
				if (udtSize(&udt) > 0)
				{
					item = udtTopBack(&udt);

					udtPopBack(&udt);

					printf("Элемент с ключом %f и строкой '%s' удален успешно\n", item._key, item._str);
				}
				else
					printf("Дек пуст\n");
			}
			break;

			case 5:
			{
				printf("Размер дека: %d (Реальный размер: %d)\n", udtSize(&udt), N);
			}
			break;

			case 6:
			{
				udtQuickSort(&udt);
			}
			break;

			case 7:
			{
				if (udtSize(&udt) > 0)
				{
					printf("Дек:\n");

					udtPrint(&udt);
				}
				else
					printf("Дек пуст\n");
			}
			break;

			case 8: break;

			default:
			{
				printf("Ошибка. Такого пункта меню не существует\n");
			}
			break;
		}
	}
	while (action != 8);
	
	udtDestroy(&udt);

	return 0;
}

void getLine(char *str, const int size)
{
	int cnt = 0, ch;

	while ((ch = getchar()) != '\n' && cnt < size - 1)
		str[cnt++] = ch;

	str[cnt] = '\0';
}
