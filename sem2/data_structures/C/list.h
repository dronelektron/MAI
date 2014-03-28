#ifndef LIST_H
#define LIST_H

#include <stdlib.h>

typedef struct _ItemList
{
	int _data;
	struct _ItemList *_prev;
	struct _ItemList *_next;
} ItemList;

typedef struct _List
{
	int _size;
	struct _ItemList *_head;
} List;

typedef struct _ItemListIterator
{
	struct _ItemList *_node;
} ItemListIterator;

void listCreate(List *list);
ItemListIterator listFirst(const List *list);
ItemListIterator listLast(const List *list);
int listEmpty(const List *list);
int listSize(const List *list);
ItemListIterator listInsert(List *list, ItemListIterator *it, const int value);
ItemListIterator listDelete(List *list, ItemListIterator *it);
void listDestroy(List *list);

int listIteratorEqual(const ItemListIterator *it1, const ItemListIterator *it2);
int listIteratorNotEqual(const ItemListIterator *it1, const ItemListIterator *it2);
ItemListIterator *listIteratorNext(ItemListIterator *it);
ItemListIterator *listIteratorPrev(ItemListIterator *it);
int listIteratorFetch(const ItemListIterator *it);
void listIteratorStore(const ItemListIterator *it, const int value);

#endif
