#include "bst.h"

BstNode *bstInsert(BstNode **node, const BST_TYPE key)
{
	if (*node == NULL)
	{
		*node = (BstNode *)malloc(sizeof(BstNode));

		(*node)->key = key;
		(*node)->left = NULL;
		(*node)->right = NULL;
	}
	else if (key > (*node)->key)
		return bstInsert(&(*node)->right, key);
	else if (key < (*node)->key)
		return bstInsert(&(*node)->left, key);

	return *node;
}

BstNode *bstFind(BstNode **node, const BST_TYPE key)
{
	if (*node == NULL)
		return NULL;
	else if (key > (*node)->key)
		return bstFind(&(*node)->right, key);
	else if (key < (*node)->key)
		return bstFind(&(*node)->left, key);

	return *node;
}

void bstRemove(BstNode **node, const BST_TYPE key)
{
	BstNode *tmpNode = NULL;

	if (*node == NULL)
		return;

	if (key > (*node)->key)
		bstRemove(&(*node)->right, key);
	else if (key < (*node)->key)
		bstRemove(&(*node)->left, key);
	else
	{
		if ((*node)->left == NULL && (*node)->right == NULL)
		{
			free(*node);

			*node = NULL;
		}
		else if ((*node)->left != NULL && (*node)->right == NULL)
		{
			tmpNode = (*node)->left;
			**node = *tmpNode;

			free(tmpNode);
		}
		else if ((*node)->left == NULL && (*node)->right != NULL)
		{
			tmpNode = (*node)->right;
			**node = *tmpNode;

			free(tmpNode);
		}
		else
		{
			tmpNode = (*node)->left;

			while (tmpNode->right != NULL)
				tmpNode = tmpNode->right;

			(*node)->key = tmpNode->key;

			bstRemove(&(*node)->left, tmpNode->key);
		}
	}
}

void bstDestroy(BstNode **node)
{
	if (*node == NULL)
		return;

	bstDestroy(&(*node)->left);
	bstDestroy(&(*node)->right);

	free(*node);

	*node = NULL;
}
