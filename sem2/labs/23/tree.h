#ifndef TREE_H
#define TREE_H

#include <stdlib.h>

typedef struct _TreeNode
{
	int _data;
	struct _TreeNode *_parent;
	struct _TreeNode *_olderBro;
	struct _TreeNode *_bro;
	struct _TreeNode *_son;
} TreeNode;

TreeNode *treeAddNode(TreeNode **node, const int value);
TreeNode *treeFindNode(TreeNode **node, const int value);
int treeRemoveNode(TreeNode **node, const int value);
void treeDestroy(TreeNode **node);

#endif
