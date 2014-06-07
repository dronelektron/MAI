#ifndef LIST_H
#define LIST_H

#include <stdio.h>
#include <stdlib.h>

typedef int LIST_TYPE;

typedef struct _Item
{
	LIST_TYPE _data;
	int _next;
} Item;

typedef struct _List
{
	Item *_arr;
	int _first;
	int _hole;
	int _capacity;
	int _size;
} List;

typedef struct _Iterator
{
	Item *_begin;
	int _index;
} Iterator;

void listCreate(List *list, const int size);
int findPrev(List *list, const int index);
void listInsert(List *list, const int index, const LIST_TYPE value);
void listRemove(List *list, const int index);
int listSize(const List *list);
int listEmpty(const List *list);
void listPrint(const List *list);
void listDestroy(List *list);

Iterator itFirst(const List *list);
void itNext(Iterator *it);
LIST_TYPE itFetch(const Iterator *it);
void itStore(Iterator *it, const LIST_TYPE value);

#endif
