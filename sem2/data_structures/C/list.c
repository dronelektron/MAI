#include "list.h"

void listCreate(List *list)
{
	list->_head = (ItemList *)malloc(sizeof(ItemList));
	list->_head->_prev = list->_head->_next = list->_head;
	list->_size = 0;
}

ItemListIterator listFirst(const List *list)
{
	ItemListIterator it;

	it._node = list->_head->_next;

	return it;
}

ItemListIterator listLast(const List *list)
{
	ItemListIterator it;

	it._node = list->_head;

	return it;
}

int listEmpty(const List *list)
{
	ItemListIterator first = listFirst(list);
	ItemListIterator last = listLast(list);

	return listIteratorEqual(&first, &last);
}

int listSize(const List *list)
{
	return list->_size;
}

ItemListIterator listInsert(List *list, ItemListIterator *it, const LIST_TYPE value)
{
	ItemListIterator res;

	res._node = (ItemList *)malloc(sizeof(ItemList));

	if (!res._node)
		return listLast(list);

	res._node->_data = value;
	res._node->_next = it->_node;
	res._node->_prev = it->_node->_prev;
	res._node->_prev->_next = res._node;
	it->_node->_prev = res._node;
	list->_size++;

	return res;
}

ItemListIterator listDelete(List *list, ItemListIterator *it)
{
	ItemListIterator res = listLast(list);

	if (listIteratorEqual(it, &res))
		return res;

	res._node = it->_node->_next;
	res._node->_prev = it->_node->_prev;
	it->_node->_prev->_next = res._node;
	list->_size--;

	free(it->_node);

	it->_node = NULL;

	return res;
}

void listDestroy(List *list)
{
	ItemList *item = list->_head->_next;
	ItemList *tmp = NULL;

	while (item != list->_head)
	{
		tmp = item;
		item = item->_next;

		free(tmp);
	}

	free(list->_head);

	list->_head = NULL;
	list->_size = 0;
}

int listIteratorEqual(const ItemListIterator *it1, const ItemListIterator *it2)
{
	return it1->_node == it2->_node;
}

int listIteratorNotEqual(const ItemListIterator *it1, const ItemListIterator *it2)
{
	return !listIteratorEqual(it1, it2);
}

ItemListIterator *listIteratorNext(ItemListIterator *it)
{
	it->_node = it->_node->_next;

	return it;
}

ItemListIterator *listIteratorPrev(ItemListIterator *it)
{
	it->_node = it->_node->_prev;

	return it;
}

int listIteratorFetch(const ItemListIterator *it)
{
	return it->_node->_data;
}

void listIteratorStore(const ItemListIterator *it, const LIST_TYPE value)
{
	it->_node->_data = value;
}
