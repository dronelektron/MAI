#include "tree.h"

TreeNode *treeAddNode(TreeNode **node, const kLetters value)
{
	TreeNode *tmpBro = NULL, *tmpNode = (TreeNode *)malloc(sizeof(TreeNode));

	tmpNode->_data = value;
	tmpNode->_parent = NULL;
	tmpNode->_olderBro = NULL;
	tmpNode->_bro = NULL;
	tmpNode->_son = NULL;

	if (*node == NULL)
		*node = tmpNode;
	else if ((*node)->_son == NULL)
	{
		tmpNode->_parent = *node;
		(*node)->_son = tmpNode;
	}
	else
	{
		tmpBro = (*node)->_son;

		while (tmpBro->_bro != NULL)
			tmpBro = tmpBro->_bro;

		tmpNode->_parent = *node;
		tmpNode->_olderBro = tmpBro;
		tmpBro->_bro = tmpNode;
	}

	return tmpNode;
}

TreeNode *treeFindNode(TreeNode **node, const kLetters value)
{
	TreeNode *tmpNode = NULL;

	if ((*node)->_data == value)
		tmpNode = *node;
	else if ((*node)->_bro != NULL)
		tmpNode = treeFindNode(&(*node)->_bro, value);

	if (tmpNode == NULL && (*node)->_son != NULL)
		tmpNode = treeFindNode(&(*node)->_son, value);

	return tmpNode;
}

int treeRemoveNode(TreeNode **node, const kLetters value)
{
	TreeNode *tmpNode = treeFindNode(node, value);

	if (tmpNode == NULL)
		return 0;

	if (tmpNode->_parent == NULL)
	{
		treeDestroy(node);

		return 1;
	}

	if (tmpNode->_olderBro == NULL)
		tmpNode->_parent->_son = tmpNode->_bro;
	else
		tmpNode->_olderBro->_bro = tmpNode->_bro;
		
	if (tmpNode->_bro != NULL)
		tmpNode->_bro->_olderBro = tmpNode->_olderBro;

	if (tmpNode->_son != NULL)
		treeDestroy(&tmpNode->_son);

	free(tmpNode);

	tmpNode = NULL;

	return 1;
}

void treeDestroy(TreeNode **node)
{
	if (*node == NULL)
		return;

	if ((*node)->_bro != NULL)
		treeDestroy(&(*node)->_bro);

	if ((*node)->_son != NULL)
		treeDestroy(&(*node)->_son);

	free(*node);

	*node = NULL;
}
