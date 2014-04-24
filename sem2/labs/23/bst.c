#include "bst.h"

BstNode *bstInsert(BstNode **node, const kLetters key)
{
	if (*node == NULL)
	{
		*node = (BstNode *)malloc(sizeof(BstNode));

		(*node)->_key = key;
		(*node)->_left = NULL;
		(*node)->_right = NULL;

		return *node;
	}
	else if ((*node)->_key == key)
		return *node;
	else if (key < (*node)->_key)
		return bstInsert(&(*node)->_left, key);
	else
		return bstInsert(&(*node)->_right, key);
}

BstNode *bstFind(BstNode **node, const kLetters key)
{
	if (*node == NULL)
		return NULL;

	if ((*node)->_key == key)
		return *node;
	
	if (key < (*node)->_key)
		return bstFind(&(*node)->_left, key);
	else
		return bstFind(&(*node)->_right, key);
}

int bstRemove(BstNode **node, const kLetters key)
{
	BstNode *repl = NULL, *parent = NULL, *tmp = *node;

	while (tmp != NULL && tmp->_key != key)
	{
		parent = tmp;

		if (key < tmp->_key)
			tmp = tmp->_left;
		else
			tmp = tmp->_right;
	}

	if (tmp == NULL)
		return 0;

	if (tmp->_left != NULL && tmp->_right == NULL)
	{
		if (parent != NULL)
		{
			if (parent->_left == tmp)
				parent->_left = tmp->_left;
			else
				parent->_right = tmp->_left;
		}
		else
			*node = tmp->_left;

		free(tmp);

		tmp = NULL;
	}
	else if (tmp->_left == NULL && tmp->_right != NULL)
	{
		if (parent != NULL)
		{
			if (parent->_left == tmp)
				parent->_left = tmp->_right;
			else
				parent->_right = tmp->_right;
		}
		else
			*node = tmp->_right;

		free(tmp);

		tmp = NULL;
	}
	else if (tmp->_left != NULL && tmp->_right != NULL)
	{
		repl = tmp->_right;

		if (repl->_left == NULL)
			tmp->_right = repl->_right;	
		else
		{
			while (repl->_left != NULL)
			{
				parent = repl;
				repl = repl->_left;
			}

			parent->_left = repl->_right;
		}

		tmp->_key = repl->_key;

		free(repl);

		repl = NULL;
	}
	else
	{
		if (parent != NULL)
		{
			if (parent->_left == tmp)
				parent->_left = NULL;
			else
				parent->_right = NULL;
		}
		else
			*node = NULL;

		free(tmp);

		tmp = NULL;
	}

	return 1;
}

void bstDestroy(BstNode **node)
{
	if (*node == NULL)
		return;

	if ((*node)->_left != NULL)
		bstDestroy(&(*node)->_left);
	
	if ((*node)->_right != NULL)
		bstDestroy(&(*node)->_right);

	free(*node);

	*node = NULL;
}
