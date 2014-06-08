#include <stdio.h>
#include "list.h"

int main(void)
{
	const int N = 10;
	int i, isFound, action, pos, arg, cnt;
	List list;
	Iterator it;

	listCreate(&list, N);

	do
	{
		printf("Меню:\n");
		printf("1) Вставить элемент\n");
		printf("2) Удалить элемент\n");
		printf("3) Печать списка\n");
		printf("4) Размер списка\n");
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
				printf("Введите значение: ");
				scanf("%d", &arg);
				
				if (arg != 0 && arg != 1)
					printf("Ошибка. Введено недопустимое значение\n");
				else
				{
					isFound = 0;

					it = itFirst(&list);

					while (it._index != END)
					{
						if (itFetch(&it) == arg)
						{
							isFound = 1;

							break;
						}

						itNext(&it);
					}

					if (isFound)
					{
						cnt = 0;
						it = itFirst(&list);

						while (it._index != END)
						{
							if (itFetch(&it) != arg)
							{
								listRemove(&list, cnt);

								it = itFirst(&list);
								cnt = 0;
								isFound = 1;
							}
							else
							{
								itNext(&it);

								cnt++;
							}
						}

						printf("Из списка были удалены все элементы, предшествующие и последующие заданному значению\n");
					}
					else
						printf("Элемент не найден\n");
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
