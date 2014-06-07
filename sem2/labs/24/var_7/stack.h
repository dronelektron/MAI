#ifndef STACK_H
#define STACK_H

#include <stdlib.h>

typedef struct _Token
{
	char _varOp;
	double _num;
} Token;

typedef Token STACK_TYPE;

typedef struct _ItemStack
{
	STACK_TYPE _data;
	struct _ItemStack *_prev;
} ItemStack;

typedef struct _Stack
{
	int _size;
	struct _ItemStack *_top;
} Stack;

void stackCreate(Stack *s);
int stackEmpty(const Stack *s);
int stackSize(const Stack *s);
int stackPush(Stack *s, const STACK_TYPE value);
int stackPop(Stack *s);
STACK_TYPE stackTop(const Stack *s);
void stackDestroy(Stack *s);

#endif
