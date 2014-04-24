#ifndef TREE_H
#define TREE_H

#include <stdlib.h>

typedef enum _kLetters
{
	A = 0, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z
} kLetters;

typedef struct _TreeNode
{
	kLetters _data;
	struct _TreeNode *_parent;
	struct _TreeNode *_olderBro;
	struct _TreeNode *_bro;
	struct _TreeNode *_son;
} TreeNode;

TreeNode *treeAddNode(TreeNode **node, const kLetters value);
TreeNode *treeFindNode(TreeNode **node, const kLetters value);
int treeRemoveNode(TreeNode **node, const kLetters value);
void treeDestroy(TreeNode **node);

#endif
