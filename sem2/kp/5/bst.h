#ifndef BST_H
#define BST_H

#include <stdio.h>
#include <stdlib.h>

#include "person.h"

typedef struct _Node
{
	int key;

	int avgMin;
	int avgMax;

	struct _Node *left;
	struct _Node *right;
} Node;

Node *bstFind(Node *node, const int key);
Node *bstInsert(Node **node, const int key);
void bstDelete(Node **node);
void printMaxGroups(Node *node, const int max);

#endif
