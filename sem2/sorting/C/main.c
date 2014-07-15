#include <stdio.h>

#include "sort1.h"
#include "sort2.h"
#include "sort3.h"
#include "sort4.h"
#include "sort5.h"
#include "sort6.h"
#include "sort7.h"
#include "sort8.h"
#include "sort9.h"
#include "sort10.h"
#include "sort11.h"
#include "sort12.h"

int main(void)
{
	const int N = 10;
	int arr[N];
	int action;
	void (*sorts[])(int *, const int) = {sort1, sort2, sort3, sort4, sort5, sort6, sort7, sort8, sort9, sort10, sort11, sort12};

	generateArray(arr, N, -10, 10);

	while (1)
	{
		printf("Меню:\n");
		printf("1) Линейный выбор с обменом\n");
		printf("2) Линейный выбор с подсчетом\n");
		printf("3) Метод пузырька\n");
		printf("4) Шейкер-сортировка\n");
		printf("5) Метод простой вставки\n");
		printf("6) Метод двоичной вставки\n");
		printf("7) Метод Шелла\n");
		printf("8) Турнирная сортировка\n");
		printf("9) Пирамидальная сортировка с просеиванием\n");
		printf("10) Простое двухпоточное слияние\n");
		printf("11) Быстрая сортировка Хоара (рекурсивный вариант)\n");
		printf("12) Быстрая сортировка Хоара (нерекурсивный вариант)\n");
		printf("--------------------------------\n");
		printf("13) Генерация массива\n");
		printf("14) Перемешка массива\n");
		printf("15) Печать массива\n");
		printf("16) Выход\n");

		printf("Выберите действие: ");
		scanf("%d", &action);

		if (action < 1 || action > 16)
		{
			printf("Ошибка. Такого пункта меню не существует\n");

			continue;
		}

		if (action < 13)
			sorts[action - 1](arr, N);

		if (action == 13)
			generateArray(arr, N, -10, 10);

		if (action == 14)
			shuffleArray(arr, N);

		if (action == 15)
			printArray(arr, N);

		if (action == 16)
			break;
	}

	return 0;
}
