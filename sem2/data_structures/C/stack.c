#include "stack.h"

void stackCreate(Stack *s)
{
	s->_size = 0;
	s->_top = NULL;
}

int stackEmpty(const Stack *s)
{
	return s->_top == NULL;
}

int stackSize(const Stack *s)
{
	return s->_size;
}

int stackPush(Stack *s, const int value)
{
	ItemStack *item = (ItemStack *)malloc(sizeof(ItemStack));

	if (!item)
		return 0;

	item->_data = value;
	item->_prev = s->_top;
	s->_top = item;
	s->_size++;

	return 1;
}

int stackPop(Stack *s)
{
	ItemStack *item = NULL;

	if (!s->_size)
		return 0;

	item = s->_top;
	s->_top = s->_top->_prev;
	s->_size--;

	free(item);

	return 1;
}

int stackTop(const Stack *s)
{
	return s->_top->_data;
}

void stackDestroy(Stack *s)
{
	ItemStack *item = NULL;

	while (s->_top)
	{
		item = s->_top;
		s->_top = s->_top->_prev;

		free(item);
	}

	s->_size = 0;
	s->_top = NULL;
}
