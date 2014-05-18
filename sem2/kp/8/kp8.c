#include <stdio.h>
#include "list.h"

int main(void)
{
	const int N = 10;
	int i, isFound, action, pos, arg;
	List list;
	Iterator it;

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
			}
			break;

			case 2:
			{
				printf("Введите номер элемента: ");
				scanf("%d", &pos);

				listRemove(&list, pos - 1);
			}
			break;

			case 3:
			{
				if (listEmpty(&list))
					printf("Список пуст\n");
				else
					listPrint(&list);
			}
			break;

			case 4:
			{
				printf("Длина списка: %d\n", listSize(&list));
			}
			break;

			case 5:
			{
				printf("Введите значение: ");
				scanf("%d", &arg);
				
				if (arg != 0 && arg != 1)
					printf("Ошибка. Введено недопустимое значение\n");
				else
				{
					it = itFirst(&list);

					isFound = 0;

					for (i = 0; i < listSize(&list); i++)
					{
						if (itFetch(&it) == arg)
						{
							while (!listEmpty(&list))
								listRemove(&list, 0);

							isFound = 1;

							break;
						}

						itNext(&it);
					}

					if (isFound)
						printf("Список был очищен, так как в нем было найдено введенное значение\n");
					else
						printf("Список не был очищен, так как в нем не найдено введенное значение\n");
				}
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

	listDestroy(&list);

	return 0;
}
