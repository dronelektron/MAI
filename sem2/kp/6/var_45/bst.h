#ifndef BST_H
#define BST_H

#include <stdlib.h>

typedef struct _Key
{
	int klass;
	char bukva;
} Key;

typedef struct _Node
{
	Key key;
	int cnt;
	struct _Node *left;
	struct _Node *right;
} Node;

Node *bstInsert(Node **node, const Key key);
int bstKeyGreater(const Key key1, const Key key2);
int bstKeyLess(const Key key1, const Key key2);
void bstDestroy(Node **node);

#endif
