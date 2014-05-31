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
		printf("1) Добавить элемент\n");
		printf("2) Удалить элемент\n");
		printf("3) Размер очереди\n");
		printf("4) Сортировка\n");
		printf("5) Печать\n");
		printf("6) Выход\n");
		printf("Выберите действие: ");
		scanf("%d", &action);

		switch (action)
		{
			case 1:
			{
				printf("Введите ключ: ");
				scanf("%lf", &item._key);
				scanf("%c", &tmpCh);
				printf("Введите Строку: ");
				getLine(item._val, sizeof(item._val));

				if (udtPush(&udt, item))
					printf("Элемент с ключом %lf и строкой '%s' добавлен успешно\n", item._key, item._val);
				else
					printf("Очередь заполнена\n");
			}
			break;

			case 2:
			{
				if (udtSize(&udt) > 0)
				{
					item = udtFront(&udt);

					udtPop(&udt);

					printf("Элемент с ключом %lf и строкой '%s' удален успешно\n", item._key, item._val);
				}
				else
					printf("Очередь пуста\n");
			}
			break;
			
			case 3:
			{
				printf("Размер очереди: %d (Реальный размер: %d)\n", udtSize(&udt), N);
			}
			break;

			case 4:
			{
				udtSort(&udt);
			}
			break;

			case 5:
			{
				if (udtSize(&udt) > 0)
				{
					printf("Очередь:\n");

					udtPrint(&udt);
				}
				else
					printf("Очередь пуст\n");
			}
			break;

			case 6: break;

			default:
			{
				printf("Ошибка. Такого пункта меню не существует\n");
			}
			break;
		}
	}
	while (action != 6);
	
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
