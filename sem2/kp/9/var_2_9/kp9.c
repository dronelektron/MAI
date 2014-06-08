#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct _Key
{
	short int a;
	short int b;
} Key;

typedef struct _Row
{
	Key key;
	char val[81];
} Row;

void printTable(const Row *rows, const int size);
int binSearch(const Row *rows, const int size, const Key key);
void sort(Row *rows, const int size);
void scramble(Row *rows, const int size);
void reverse(Row *rows, const int size);

void getRow(FILE *stream, char *str, const int size);
void swapRows(Row *rows, const int a, const int b);
int comparator(const Key k1, const Key k2);
int randomAB(const int a, const int b);
int isSorted(const Row *rows, const int size);

int main(void)
{
	const int N = 50;
	int i, cnt, action;
	char ch;
	Row data[N];
	Key key;
	FILE *file = fopen("input.txt", "r");

	if (file == NULL)
	{
		printf("Ошибка при открытии файла\n");

		return 0;
	}

	i = 0;

	while (i < N && fscanf(file, "%hd %hd", &data[i].key.a, &data[i].key.b) == 2)
	{
		fscanf(file, "%c", &ch);
		getRow(file, data[i].val, sizeof(data[i].val));

		i++;
	}

	fclose(file);

	cnt = i;

	do
	{
		printf("Меню\n");
		printf("1) Печать\n");
		printf("2) Двоичный поиск\n");
		printf("3) Сортировка\n");
		printf("4) Перемешивание\n");
		printf("5) Реверс\n");
		printf("6) Выход\n");
		printf("Выберите действие\n");
		scanf("%d", &action);

		switch (action)
		{
			case 1:
			{
				printTable(data, cnt);

				break;
			}

			case 2:
			{
				if (!isSorted(data, cnt))
					printf("Ошибка. Таблица не отсортирована\n");
				else
				{
					printf("Введите ключ: ");
					scanf("%hd %hd", &key.a, &key.b);

					i = binSearch(data, cnt, key);

					if (i > -1)
						printf("Найдена строка: %s\n", data[i].val);
					else
						printf("Строка с таким ключом не найдена\n");
				}

				break;
			}

			case 3:
			{
				sort(data, cnt);

				break;
			}

			case 4:
			{
				scramble(data, cnt);

				break;
			}

			case 5:
			{
				reverse(data, cnt);

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

	return 0;
}

void printTable(const Row *rows, const int size)
{
	int i;

	printf("+---------+------------------------------------------------------------+\n");
	printf("|  Ключ   |                          Значение                          |\n");
	printf("+---------+------------------------------------------------------------+\n");

	for (i = 0; i < size; i++)
		printf("|%4hd %4hd|%60s|\n", rows[i].key.a, rows[i].key.b, rows[i].val);

	printf("+---------+------------------------------------------------------------+\n");
}

int binSearch(const Row *rows, const int size, const Key key)
{
	int start = 0, end = size - 1, mid;

	if (size <= 0)
		return -1;

	while (start < end)
	{
		mid = start + (end - start) / 2;

		if (comparator(rows[mid].key, key) == 0)
			return mid;
		else if (comparator(rows[mid].key, key) < 1)
			start = mid + 1;
		else
			end = mid;
	}

	if (comparator(rows[end].key, key) == 0)
		return end;

	return -1;
}

void sort(Row *rows, const int size)
{
	int i, j;
	int count[size];
	Row b[size];

	for (i = 0; i < size; i++)
		count[i] = 0;

	for (i = 0; i < size - 1; i++)
		for (j = i + 1; j < size; j++)
			if (comparator(rows[i].key, rows[j].key) < 1)
				count[j]++;
			else
				count[i]++;

	for (i = 0; i < size; i++)
		b[count[i]] = rows[i];

	for (i = 0; i < size; i++)
		rows[i] = b[i];
}

void scramble(Row *rows, const int size)
{
	int i, j, z;

	srand((unsigned int)time(0));

	for (z = 0; z < size; z++)
	{
		i = randomAB(0, size - 1);
		j = randomAB(0, size - 1);

		swapRows(rows, i, j);
	}
}

void reverse(Row *rows, const int size)
{
	int i, j;

	for (i = 0, j = size - 1; i < j; i++, j--)
		swapRows(rows, i, j);
}

void getRow(FILE *stream, char *str, const int size)
{
	int cnt = 0, ch;

	while ((ch = getc(stream)) != '\n' && cnt < size - 1)
		str[cnt++] = ch;

	str[cnt] = '\0';
}

void swapRows(Row *rows, const int a, const int b)
{
	Row tmp;

	tmp = rows[a];
	rows[a] = rows[b];
	rows[b] = tmp;
}

int comparator(const Key k1, const Key k2)
{
	if (k1.a > k2.a)
		return 1;

	if (k1.a < k2.a)
		return -1;

	if (k1.b > k2.b)
		return 1;

	if (k1.b < k2.b)
		return -1;

	return 0;
}

int randomAB(const int a, const int b)
{
	return a + rand() % (b - a + 1);
}

int isSorted(const Row *rows, const int size)
{
	int i;

	for (i = 0; i < size - 1; i++)
		if (comparator(rows[i].key, rows[i + 1].key) > 0)
			return 0;

	return 1;
}
