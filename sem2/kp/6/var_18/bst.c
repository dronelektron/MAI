#include "bst.h"

Node *bstFind(Node *node, const int key)
{
	if (node == NULL)
		return NULL;

	if (node->key == key)
		return node;
	
	if (key >= node->key)
		return bstFind(node->right, key);
	else
		return bstFind(node->left, key);
}

Node *bstInsert(Node **node, const int key)
{
	if (*node == NULL)
	{
		*node = (Node *)malloc(sizeof(Node));

		(*node)->key = key;
		(*node)->left = NULL;
		(*node)->right = NULL;

		// Минимальный и максимальный средний балл для трех предметов
		(*node)->avgMin = 15; // 5 + 5 + 5
		(*node)->avgMax = 6; // 2 + 2 + 2

		return *node;
	}
	else if ((*node)->key == key)
		return *node;
	else if (key >= (*node)->key)
		return bstInsert(&(*node)->right, key);
	else
		return bstInsert(&(*node)->left, key);
}

void bstDelete(Node **node)
{
	if (*node == NULL)
		return;

	if ((*node)->left != NULL)
		bstDelete(&(*node)->left);
	
	if ((*node)->right != NULL)
		bstDelete(&(*node)->right);

	free(*node);

	*node = NULL;
}

void printMaxGroups(Node *node, const int max)
{
	if (node == NULL)
		return;

	if (node->left != NULL)
		printMaxGroups(node->left, max);

	if (node->avgMax - node->avgMin == max)
		printf("%d\n", node->key);

	if (node->right != NULL)
		printMaxGroups(node->right, max);
}
