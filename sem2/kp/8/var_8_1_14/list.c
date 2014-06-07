#include "list.h"

void listCreate(List *list, const int size)
{
	int i;

	if (size <= 0)
		return;

	list->_arr = (Item *)malloc(sizeof(Item) * size);

	for (i = 0; i < size - 1; i++)
		list->_arr[i]._next = i + 1;

	list->_arr[size - 1]._next = -1;

	list->_first = 0;
	list->_hole = 0;
	list->_capacity = size;
	list->_size = 0;
}

int findPrev(List *list, const int index)
{
	int i, prev = list->_first;

	if (list->_size <= 1)
		return prev;

	if (index == 0 || index == list->_size)
		while (list->_arr[prev]._next != list->_first)
			prev = list->_arr[prev]._next;
	else
		for (i = 0; i < index - 1; i++)
			prev = list->_arr[prev]._next;

	return prev;
}

void listInsert(List *list, const int index, const LIST_TYPE value)
{
	int i, prev, nextHole;

	if (list->_size == list->_capacity)
	{
		printf("Ошибка. Список полон\n");

		return;
	}

	if (list->_size)
	{
		if (index < 0 || index > list->_size)
		{
			printf("Ошибка. Позиция не найдена\n");

			return;
		}
	}
	else if (index != 0)
	{
		printf("Ошибка. Позиция не найдена\n");

		return;
	}

	prev = findPrev(list, index);
	nextHole = list->_arr[list->_hole]._next;
	list->_arr[list->_hole]._data = value;
	list->_arr[list->_hole]._next = list->_arr[prev]._next;

	if (index == 0)
		list->_first = list->_hole;

	list->_arr[prev]._next = list->_hole;
	list->_hole = nextHole;
	list->_size++;
	
	printf("Элемент %s вставлен в список\n", value == 1 ? "true" : "false");
}

void listRemove(List *list, const int index)
{
	int prev, cur;

	if (listEmpty(list))
	{
		printf("Список пуст\n");

		return;
	}
	else if (index < 0 || index >= list->_size)
	{
		printf("Ошибка. Позиция не найдена\n");

		return;
	}

	prev = findPrev(list, index);
	cur = list->_arr[prev]._next;

	printf("Элемент %s удален из списка\n", list->_arr[cur]._data == 1 ? "true" : "false");
	
	list->_arr[prev]._next = list->_arr[cur]._next;

	if (index == 0)
		list->_first = list->_arr[prev]._next;
	
	list->_arr[cur]._data = 0;
	list->_arr[cur]._next = list->_hole;
	list->_hole = cur;

	list->_size--;
}

int listSize(const List *list)
{
	return list->_size;
}

int listEmpty(const List *list)
{
	return list->_size == 0;
}

void listPrint(const List *list)
{
	int i, cur = list->_first;
	
	for (i = 0; i < list->_size; i++)
	{
		printf("%s ", list->_arr[cur]._data == 1 ? "true" : "false");

		cur = list->_arr[cur]._next;
	}

	printf("\n");
}

void listDestroy(List *list)
{
	if (list->_arr != NULL)
	{
		free(list->_arr);

		list->_arr = NULL;
	}

	list->_first = 0;
	list->_hole = 0;
	list->_capacity = 0;
	list->_size = 0;
}

Iterator itFirst(const List *list)
{
	Iterator it;

	it._begin = list->_arr;
	it._index = list->_first;

	return it;
}

void itNext(Iterator *it)
{
	it->_index = it->_begin[it->_index]._next;	
}

LIST_TYPE itFetch(const Iterator *it)
{
	return it->_begin[it->_index]._data;
}

void itStore(Iterator *it, const LIST_TYPE value)
{
	it->_begin[it->_index]._data = value;
}
