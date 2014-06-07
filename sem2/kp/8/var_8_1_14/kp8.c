#include <stdio.h>
#include "list.h"

void listSwapElems(Iterator *it1, Iterator *it2);

int main(void)
{
	const int N = 10;
	int i, isFound, action, pos, arg;
	List list;
	Iterator it1, it2;

	listCreate(&list, N);

	do
	{
		printf("Меню:\n");
		printf("1) Вставить элемент\n");
		printf("2) Удалить элемент\n");
		printf("3) Печать списка\n");
		printf("4) Подсчет длины списка\n");
		printf("5) Выполнить задание над списком\n");
		printf("6) Выход\n");
		printf("Выберите действие: ");
		scanf("%d", &action);

		switch (action)
		{
			case 1:
			{
				printf("Введите позицию элемента: ");
				scanf("%d", &pos);
				printf("Введите значение элемента (1 - true, 0 - false): ");
				scanf("%d", &arg);

				if (arg != 0 && arg != 1)
					printf("Ошибка. Введено недопустимое значение\n");
				else
					listInsert(&list, pos - 1, arg);

				break;
			}

			case 2:
			{
				printf("Введите номер элемента: ");
				scanf("%d", &pos);

				listRemove(&list, pos - 1);

				break;
			}

			case 3:
			{
				if (listEmpty(&list))
					printf("Список пуст\n");
				else
					listPrint(&list);

				break;
			}

			case 4:
			{
				printf("Длина списка: %d\n", listSize(&list));

				break;
			}

			case 5:
			{
				it1 = itFirst(&list);
				it2 = it1;

				itNext(&it2);

				for (i = 0; i < listSize(&list) / 2; i++)
				{
					listSwapElems(&it1, &it2);

					itNext(&it1);
					itNext(&it1);
					itNext(&it2);
					itNext(&it2);
				}

				break;
			}

			case 6: break;

			default:
			{
				printf("Ошибка. Такого пункта меню не существует\n");

				break;
			}
		}
	}
	while (action != 6);

	listDestroy(&list);

	return 0;
}

void listSwapElems(Iterator *it1, Iterator *it2)
{
	LIST_TYPE tmp = itFetch(it1);

	itStore(it1, itFetch(it2));
	itStore(it2, tmp);
}
