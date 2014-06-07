#include "bst.h"

Node *bstInsert(Node **node, const Key key)
{
	if (*node == NULL)
	{
		*node = (Node *)malloc(sizeof(Node));
		(*node)->key = key;
		(*node)->cnt = 0;
		(*node)->left = NULL;
		(*node)->right = NULL;

		return *node;
	}
	else if (bstKeyGreater(key, (*node)->key))
		return bstInsert(&(*node)->right, key);
	else if (bstKeyLess(key, (*node)->key))
		return bstInsert(&(*node)->left, key);
	
	return *node;
}

int bstKeyGreater(const Key key1, const Key key2)
{
	if (key1.klass == key2.klass)
		return key1.bukva > key2.bukva;

	return key1.klass > key2.klass;
}

int bstKeyLess(const Key key1, const Key key2)
{
	if (key1.klass == key2.klass)
		return key1.bukva < key2.bukva;

	return key1.klass < key2.klass;
}

void bstDestroy(Node **node)
{
	if (*node == NULL)
		return;

	bstDestroy(&(*node)->left);
	bstDestroy(&(*node)->right);

	free(*node);

	*node = NULL;
}
