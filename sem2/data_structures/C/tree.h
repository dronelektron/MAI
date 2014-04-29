#ifndef TREE_H
#define TREE_H

#include <stdlib.h>

typedef int TREE_TYPE;

typedef struct _TreeNode
{
	TREE_TYPE _data;
	struct _TreeNode *_parent;
	struct _TreeNode *_olderBro;
	struct _TreeNode *_bro;
	struct _TreeNode *_son;
} TreeNode;

TreeNode *treeAddNode(TreeNode **node, const TREE_TYPE value);
TreeNode *treeFindNode(TreeNode **node, const TREE_TYPE value);
int treeRemoveNode(TreeNode **node, const TREE_TYPE value);
void treeDestroy(TreeNode **node);

#endif
